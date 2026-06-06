"""
Tutorial validation: tool smoke tests.

验证工具程序的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestToolSmoke:
    """工具程序 smoke 测试。"""

    def test_check_mesh_import(self):
        """checkMesh 可导入。"""
        from pyfoam.tools import check_mesh
        assert check_mesh is not None

    def test_set_fields_import(self):
        """setFields 可导入。"""
        from pyfoam.tools import set_fields
        assert set_fields is not None

    def test_decompose_par_import(self):
        """decomposePar 可导入。"""
        from pyfoam.tools import decompose_par
        assert decompose_par is not None

    def test_reconstruct_par_import(self):
        """reconstructPar 可导入。"""
        from pyfoam.tools import reconstruct_par
        assert reconstruct_par is not None

    def test_foam_to_vtk_import(self):
        """foamToVTK 可导入。"""
        from pyfoam.tools import foam_to_vtk
        assert foam_to_vtk is not None

    def test_foam_to_ensight_import(self):
        """foamToEnsight 可导入。"""
        from pyfoam.tools import foam_to_ensight
        assert foam_to_ensight is not None

    def test_foam_to_fluent_import(self):
        """foamToFluent 可导入。"""
        from pyfoam.tools import foam_to_fluent
        assert foam_to_fluent is not None

    def test_foam_dictionary_import(self):
        """foamDictionary 可导入。"""
        from pyfoam.tools import foam_dictionary
        assert foam_dictionary is not None

    def test_foam_list_times_import(self):
        """foamListTimes 可导入。"""
        from pyfoam.tools import foam_list_times
        assert foam_list_times is not None

    def test_transform_points_import(self):
        """transformPoints 可导入。"""
        from pyfoam.tools import transform_points
        assert transform_points is not None

    def test_refine_mesh_import(self):
        """refineMesh 可导入。"""
        from pyfoam.tools import refine_mesh
        assert refine_mesh is not None

    def test_merge_meshes_import(self):
        """mergeMeshes 可导入。"""
        from pyfoam.tools import merge_meshes
        assert merge_meshes is not None

    def test_stitch_mesh_import(self):
        """stitchMesh 可导入。"""
        from pyfoam.tools import stitch_mesh
        assert stitch_mesh is not None

    def test_create_baffles_import(self):
        """createBaffles 可导入。"""
        from pyfoam.tools import create_baffles
        assert create_baffles is not None

    def test_snappy_hex_mesh_import(self):
        """snappyHexMesh 可导入。"""
        from pyfoam.tools import snappy_hex_mesh
        assert snappy_hex_mesh is not None

    def test_noise_import(self):
        """noise 可导入。"""
        from pyfoam.tools import noise_analysis
        assert noise_analysis is not None

    def test_surface_features_import(self):
        """surfaceFeatures 可导入。"""
        from pyfoam.tools import surface_features
        assert surface_features is not None

    def test_surface_convert_import(self):
        """surfaceConvert 可导入。"""
        from pyfoam.tools import surface_convert
        assert surface_convert is not None

    def test_map_fields_import(self):
        """mapFields 可导入。"""
        from pyfoam.tools import map_fields
        assert map_fields is not None

    def test_subset_mesh_import(self):
        """subsetMesh 可导入。"""
        from pyfoam.tools import subset_mesh
        assert subset_mesh is not None
