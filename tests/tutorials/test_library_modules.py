"""
Tutorial validation: library module smoke tests.

验证所有新实现的库模块的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


class TestPolyTopoChangeSmoke:
    """polyTopoChange 模块 smoke 测试。"""

    def test_import(self):
        from pyfoam.poly_topo_change import PolyTopoChange, TopoSet, BoxToCell, CylinderToCell
        assert PolyTopoChange is not None
        assert TopoSet is not None
        assert BoxToCell is not None
        assert CylinderToCell is not None

    def test_poly_topo_change_basic(self):
        from pyfoam.poly_topo_change import PolyTopoChange
        topo = PolyTopoChange(n_cells=100)
        topo.add_cell([0, 1, 2, 3])
        assert topo.n_pending == 1


class TestSurfMeshSmoke:
    """surfMesh 模块 smoke 测试。"""

    def test_import(self):
        from pyfoam.surf_mesh import SurfMesh, SurfZone, SurfScalarField, SurfVectorField
        assert SurfMesh is not None
        assert SurfZone is not None
        assert SurfScalarField is not None
        assert SurfVectorField is not None

    def test_surf_mesh_basic(self):
        from pyfoam.surf_mesh import SurfMesh
        pts = torch.tensor([[0,0,0],[1,0,0],[0.5,1,0]], dtype=CFD_DTYPE)
        faces = [torch.tensor([0, 1, 2])]
        mesh = SurfMesh(points=pts, faces=faces)
        assert mesh.n_points == 3
        assert mesh.n_faces == 1


class TestPhysicalPropertiesSmoke:
    """physicalProperties 模块 smoke 测试。"""

    def test_import(self):
        from pyfoam.physical_properties import PhysicalProperties, ConstantViscosity, PolynomialViscosity
        assert PhysicalProperties is not None
        assert ConstantViscosity is not None
        assert PolynomialViscosity is not None

    def test_physical_properties_basic(self):
        from pyfoam.physical_properties import PhysicalProperties
        props = PhysicalProperties(nu=1e-5, rho=1.0)
        assert props.nu == pytest.approx(1e-5)
        assert props.Pr > 0


class TestSpecieTransferSmoke:
    """specieTransfer 模块 smoke 测试。"""

    def test_import(self):
        from pyfoam.specie_transfer import SpecieTransferModel, SimpleDiffusionModel
        assert SpecieTransferModel is not None
        assert SimpleDiffusionModel is not None

    def test_simple_diffusion_basic(self):
        from pyfoam.specie_transfer import SimpleDiffusionModel
        model = SimpleDiffusionModel(D_mass=1e-5)
        assert model.molecular_diffusivity == pytest.approx(1e-5)


class TestFvMeshFrameworkSmoke:
    """fvMesh 框架 smoke 测试。"""

    def test_import(self):
        from pyfoam.fv_mesh_framework import MeshMover, DeformingMeshMover, MeshStitcher, MeshTopoChanger, MeshDistributor
        assert MeshMover is not None
        assert DeformingMeshMover is not None
        assert MeshStitcher is not None
        assert MeshTopoChanger is not None
        assert MeshDistributor is not None


class TestFvAgglomerationSmoke:
    """fvAgglomeration 模块 smoke 测试。"""

    def test_import(self):
        from pyfoam.fv_agglomeration import PairGamgAgglomeration
        assert PairGamgAgglomeration is not None

    def test_pair_agglomeration_basic(self):
        from pyfoam.fv_agglomeration import PairGamgAgglomeration
        owner = torch.tensor([0, 1, 2], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1, 2, 3], dtype=INDEX_DTYPE)
        agg = PairGamgAgglomeration(n_cells=4, owner=owner, neighbour=neighbour)
        assert agg.n_levels == 0  # too small to agglomerate


class TestRandomProcessesSmoke:
    """randomProcesses 模块 smoke 测试。"""

    def test_import(self):
        from pyfoam.random_processes import FFT, Kmesh, TurbGen, OUProcess, NoiseFFT
        assert FFT is not None
        assert Kmesh is not None
        assert TurbGen is not None
        assert OUProcess is not None
        assert NoiseFFT is not None

    def test_fft_roundtrip(self):
        from pyfoam.random_processes import FFT
        x = torch.randn(8, 8, 8, dtype=CFD_DTYPE)
        X = FFT.forward_transform(x, (8, 8, 8))
        x_recovered = FFT.reverse_transform(X, (8, 8, 8)).real
        assert torch.allclose(x, x_recovered, atol=1e-10)

    def test_kmesh_basic(self):
        from pyfoam.random_processes import Kmesh
        km = Kmesh(nn=(8, 8, 8))
        assert km.n_total == 512
        assert km.kmax > 0

    def test_turb_gen_basic(self):
        from pyfoam.random_processes import TurbGen, Kmesh
        km = Kmesh(nn=(8, 8, 8))
        gen = TurbGen(kmesh=km, Ea=1.0, k0=4.0, seed=42)
        U = gen.velocity_field()
        assert U.shape == (512, 3)
        assert torch.isfinite(U).all()

    def test_ou_process_basic(self):
        from pyfoam.random_processes import OUProcess
        ou = OUProcess(n_modes=50, alpha=1.0, sigma=0.1)
        field = ou.step(dt=0.01)
        assert field.shape == (50, 3)
        assert torch.isfinite(field).all()

    def test_noise_fft_basic(self):
        from pyfoam.random_processes import NoiseFFT
        t = torch.linspace(0, 1, 1024, dtype=CFD_DTYPE)
        p = torch.sin(2 * 3.14159 * 100 * t)
        noise = NoiseFFT(p, dt=1/1024, window_size=512)
        freqs, spl = noise.spl_spectrum()
        assert len(freqs) > 0
        assert len(spl) > 0
