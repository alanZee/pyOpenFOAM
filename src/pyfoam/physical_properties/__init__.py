"""physicalProperties — 物性参数统一框架。"""
from pyfoam.physical_properties.physical_properties import PhysicalProperties
from pyfoam.physical_properties.viscosity_models import (
    ViscosityModel,
    ConstantViscosity,
    PolynomialViscosity,
)

__all__ = [
    "PhysicalProperties",
    "ViscosityModel",
    "ConstantViscosity",
    "PolynomialViscosity",
]
