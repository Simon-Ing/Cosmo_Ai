# (C) 2026: Hans Georg Schaathun <georg@schaathun.net> 

from . import CosmoSimPy as cs

"""
The module provides dictionaries of the different model identifiers,
both lens, simulator, and source models.  They are used as lookup tables
for the different strings used in configuration files and command line 
arguments.
"""

ModelSpec = cs.ModelSpec
SourceSpec = cs.SourceSpec
PsiSpec = cs.PsiSpec
LightProfileSpec = cs.LightProfileSpec
Point = cs.Point 

lensDict = {
        "PM" : PsiSpec.PM,
        "PointMass" : PsiSpec.PM,
        "SIS" : PsiSpec.SIS,
        "SIE" : PsiSpec.SIE,
        "Cluster" : PsiSpec.Cluster,
        }
lensValues = {
        "PointMass" : PsiSpec.PM,
        "SIS" : PsiSpec.SIS,
        "SIE" : PsiSpec.SIE,
        }
modelValues = {
        "Raytrace" : ModelSpec.Raytrace,
        "Roulette" : ModelSpec.Roulette,
        }
modelDict = {
        "Raytrace" : ModelSpec.Raytrace,
        "Roulette" : ModelSpec.Roulette,
        "RouletteRegenerator" : ModelSpec.RouletteRegenerator,
        }

# Note that the source names are looked up in two dictionaries,
# both the `sourceDict` and the `lightProfileDict`.
sourceDict = {
        "Spherical" : SourceSpec.Sphere,
        "Ellipsoid" : SourceSpec.Ellipse,
        "SersicSphere" : SourceSpec.Sphere,
        "SersicEllipsoid" : SourceSpec.Ellipse,
        "Image (Einstein)" : SourceSpec.Image,
        "Triangle" : SourceSpec.Triangle,
        "s" : SourceSpec.Sphere,
        "e" : SourceSpec.Ellipse,
        "t" : SourceSpec.Triangle,
        }
sourceValues = {
        "Spherical" : SourceSpec.Sphere,
        "Ellipsoid" : SourceSpec.Ellipse,
        "Image (Einstein)" : SourceSpec.Image,
        "Triangle" : SourceSpec.Triangle,
        }
lightProfileDict = {
        "Gaussian" : LightProfileSpec.Gaussian,
        "Sersic" : LightProfileSpec.Sersic,
        "Spherical" : LightProfileSpec.Gaussian,
        "Ellipsoid" : LightProfileSpec.Gaussian,
        "SersicSphere" : LightProfileSpec.Sersic,
        "SersicEllipsoid" : LightProfileSpec.Sersic,
        # "g" : LightProfileSpec.Gaussian,
        # "s" : LightProfileSpec.Sersic,
        }


