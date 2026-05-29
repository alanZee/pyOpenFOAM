"""Tests for external coupled boundary condition."""

import pytest
import torch
import tempfile
from pathlib import Path

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.external_coupled import ExternalCoupledBC


class TestExternalCoupledBC:
    """Test the externalCoupled boundary condition."""

    def test_registration(self):
        assert "externalCoupled" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "externalCoupled", simple_patch,
            {"commsDir": "/tmp/test", "nSample": 10},
        )
        assert isinstance(bc, ExternalCoupledBC)

    def test_type_name(self, simple_patch):
        bc = ExternalCoupledBC(simple_patch)
        assert bc.type_name == "externalCoupled"

    def test_default_properties(self, simple_patch):
        bc = ExternalCoupledBC(simple_patch)
        assert bc.comms_dir == "/tmp/externalCoupling"
        assert bc.n_sample == 0
        assert bc.transform_model == "linear"
        assert bc.external_values is None

    def test_custom_properties(self, simple_patch):
        bc = ExternalCoupledBC(simple_patch, {
            "commsDir": "/my/dir",
            "nSample": 50,
            "transformModel": "nearestCell",
        })
        assert bc.comms_dir == "/my/dir"
        assert bc.n_sample == 50
        assert bc.transform_model == "nearestCell"

    def test_load_from_file(self, simple_patch):
        """Load values from a temporary file."""
        bc = ExternalCoupledBC(simple_patch, {"value": 0.0})
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / simple_patch.name
            with open(data_file, "w") as f:
                for v in [100.0, 200.0, 300.0]:
                    f.write(f"{v:.10e}\n")
            result = bc.load(path=data_file)
            assert result.shape == (3,)
            assert torch.allclose(
                result,
                torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64),
            )
            assert bc.external_values is not None

    def test_load_missing_file_uses_fallback(self, simple_patch):
        """Missing file falls back to value coefficient."""
        bc = ExternalCoupledBC(simple_patch, {"value": 42.0})
        result = bc.load(path="/nonexistent/path/data")
        assert torch.allclose(
            result,
            torch.full((3,), 42.0, dtype=torch.float64),
        )

    def test_save(self, simple_patch):
        """Save boundary values to file."""
        bc = ExternalCoupledBC(simple_patch)
        field = torch.tensor([
            0.0, 0.0, 0.0,
            10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 0.0,
            100.0, 200.0, 300.0, 0.0, 0.0,
        ], dtype=torch.float64)
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / simple_patch.name
            bc.save(field, path=data_file)
            # Read back
            lines = data_file.read_text().strip().split("\n")
            assert len(lines) == 3
            assert float(lines[0]) == pytest.approx(100.0)
            assert float(lines[1]) == pytest.approx(200.0)
            assert float(lines[2]) == pytest.approx(300.0)

    def test_apply_with_external_values(self, simple_patch):
        """Apply uses loaded external values."""
        bc = ExternalCoupledBC(simple_patch)
        bc._external_values = torch.tensor(
            [10.0, 20.0, 30.0], dtype=torch.float64,
        )
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(20.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(30.0, dtype=torch.float64))

    def test_apply_without_external_values_fallback(self, simple_patch):
        """Without loaded values, falls back to zero-gradient."""
        bc = ExternalCoupledBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 5.0
        field[1] = 6.0
        field[2] = 7.0
        bc.apply(field)
        assert field[10] == pytest.approx(5.0)
        assert field[11] == pytest.approx(6.0)
        assert field[12] == pytest.approx(7.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = ExternalCoupledBC(simple_patch)
        bc._external_values = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float64,
        )
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert field[5] == pytest.approx(1.0)
        assert field[6] == pytest.approx(2.0)
        assert field[7] == pytest.approx(3.0)

    def test_matrix_contributions_with_values(self, simple_patch):
        """Penalty method with loaded external values."""
        bc = ExternalCoupledBC(simple_patch)
        bc._external_values = torch.tensor(
            [100.0, 200.0, 300.0], dtype=torch.float64,
        )
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([200.0, 400.0, 600.0], dtype=torch.float64))

    def test_matrix_contributions_without_values(self, simple_patch):
        """Zero contribution when no external values loaded."""
        bc = ExternalCoupledBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)
        assert torch.allclose(diag, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))
