"""
Gaussian Splat Generator — Blender Extension entry point.

Blender discovers register/unregister from this package init.
All implementation lives in blender_gausian_splat.py.
"""

from .blender_gausian_splat import register, unregister

__all__ = ["register", "unregister"]
