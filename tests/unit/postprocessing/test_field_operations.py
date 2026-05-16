"""Tests for FieldOperations function object."""

from __future__ import annotations

import pytest
import torch

from pyfoam.postprocessing.field_operations import FieldOperations, create_field_operation


class TestFieldOperations:
    def test_init_defaults(self):
        fo = FieldOperations()
        assert fo.name == "fieldOperation"
        assert fo._operation == "grad"

    def test_init_with_config(self):
        config = {
            "operation": "curl",
            "field": "U",
            "resultName": "vorticity",
        }
        fo = FieldOperations("vort", config)
        assert fo._operation == "curl"
        assert fo._field_name == "U"
        assert fo._result_name == "vorticity"

    def test_init_invalid_operation(self):
        with pytest.raises(ValueError, match="Unknown operation"):
            FieldOperations("test", {"operation": "invalid"})

    def test_valid_operations(self):
        for op in ["grad", "div", "curl", "mag", "magSqr", "vorticity", "enstrophy"]:
            fo = FieldOperations("test", {"operation": op})
            assert fo._operation == op

    def test_initialise(self, fv_mesh, sample_fields):
        fo = FieldOperations("gradU", {"operation": "grad", "field": "U"})
        fo.initialise(fv_mesh, sample_fields)
        assert fo.mesh is fv_mesh

    def test_execute_grad(self, fv_mesh, sample_fields):
        fo = FieldOperations("gradU", {"operation": "grad", "field": "U"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert fo.result_data is not None
        assert fo.result_data.shape == (2, 3, 3)  # (n_cells, 3, 3) for gradient of vector

    def test_execute_curl(self, fv_mesh, sample_fields):
        fo = FieldOperations("curlU", {"operation": "curl", "field": "U"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert fo.result_data is not None
        assert fo.result_data.shape == (2, 3)  # (n_cells, 3) for curl of vector

    def test_execute_mag(self, fv_mesh, sample_fields):
        fo = FieldOperations("magU", {"operation": "mag", "field": "U"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert fo.result_data is not None
        assert fo.result_data.shape == (2,)  # (n_cells,) for magnitude of vector

    def test_execute_mag_sqr(self, fv_mesh, sample_fields):
        fo = FieldOperations("magSqrU", {"operation": "magSqr", "field": "U"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert fo.result_data is not None
        assert fo.result_data.shape == (2,)

    def test_execute_enstrophy(self, fv_mesh, sample_fields):
        fo = FieldOperations("enstrophy", {"operation": "enstrophy", "field": "U"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert fo.result_data is not None
        assert fo.result_data.shape == (2,)

    def test_execute_missing_field(self, fv_mesh, sample_fields):
        fo = FieldOperations("gradP", {"operation": "grad", "field": "nonexistent"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert fo.result_data is None

    def test_result_name(self):
        fo = FieldOperations("test", {"operation": "grad", "field": "U"})
        assert fo.result_name == "grad(U)"

    def test_custom_result_name(self):
        fo = FieldOperations("test", {"operation": "grad", "field": "U", "resultName": "gradU"})
        assert fo.result_name == "gradU"


class TestCreateFieldOperation:
    def test_create(self):
        fo = create_field_operation("gradU", "grad", "U")
        assert fo.name == "gradU"
        assert fo._operation == "grad"
        assert fo._field_name == "U"

    def test_create_with_kwargs(self):
        fo = create_field_operation("gradU", "grad", "U", writeField=True)
        assert fo._write_field is True


class TestFieldOperationsRegistration:
    def test_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        from pyfoam.postprocessing import field_operations
        assert "fieldOperation" in FunctionObjectRegistry.list_registered()
