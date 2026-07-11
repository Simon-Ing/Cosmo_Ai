# (C) 2026: Hans Georg Schaathun <georg@schaathun.net> 

"""
This defines the `ArgumentParser` for the CosmoSim scripts.
Several scripts use the same arguments, and thus share this module.

The argument list is not complete.  For ease of maintenance, we should
abandon support for command line arguments and refer to CSV and TOML 
files instead.

`paramap` is a mapping from the command line options into the
structure of the TOML config files.  The key is the parameter
name used in CSV and on the command line, while the value is
the key used in TOML and nested `dict`s.
"""

import argparse
from cascadict import CascaDict
from copy import deepcopy

class Parameters:
    """
    The Parameters class wraps the commandline arguments as well as a CSV 
    row as a dict-like object.

    The constructor can take either an `ArgumentParser` object (`args`) 
    as well as a nested `dict` (`cfg`) as given by a TOML file.  
    Only selected parameters are used from the
    TOML format, and they will never override `args`/`cfg` parameters.

    For executable scripts, one can use `CosmoParser` object directly as
    the `args` argument, in addition to a TOML file if available.
    For API use, one should normally use the `cfg` argument.

    To add a row from the dataset, use the `serRow()` method.  Values
    from the row will always override other parameters.
    """
    def __init__(self,fileconfig=None,cliconfig=None,verbosity=0):

        self._cliconfig = cliconfig
        self._fileconfig = fileconfig
        self.verbosity = verbosity

        cfg = CascaDict( skel )
        if fileconfig:
           cfg = cfg.cascade( fileconfig  )
        if cliconfig:
           cfg = cfg.cascade( cliconfig )
        self._base = cfg
        self.reset()
    def copy(self):
        return deepcopy(self)
    def reset(self):
        self.config = self._base.cascade( {})
    def __str__(self): 
        return self.config.__str__()
    def setRow(self,row):
        if self.verbosity > 1: print( row )
        self._row = row
        c = getConfig( self._row, verbose=self.verbosity ) 
        if self.verbosity: print( c )
        self.config = self._base.cascade( c )
    def get(self,key,default=None,verbose=0):
        if isinstance( key, str ):
            return self.get( paramap[key], default, verbose )
        elif isinstance( key, tuple ):
            cf = self.config
            for k in key:
                cf = cf.get( k, None )
                if cf is None: break
            if cf is None:
                return default
            else: 
                return cf
        else:
            print( key )
            raise Exception( "Key should be string or tuple." )
    def __getitem__(self,key):
        return self.config.get(key)
    def __setitem__(self,key,v):
        self._row[key] = v

skel = { "simulator" : { "imagesize": 512, "model" : "Raytrace" }
        , "source" : { "sigma" : 10, "sigma2" : 20, "theta" : 45 }
        , "lens" : { "mode" : "SIS", "einsteinradius" : 20, "ellipseratio" : 0.6, 
                    "orientation" : 45 }
         , "dataset" : {}
         , "annotation" : {}
         , "management" : { "maxcount" : None ,"filename" : "test.png" },
         }

paramap = {
        "lens" : ( "lens", "mode" ),
        "model" : ( "simulator", "model" ),
        "sampled" : ( "simulator", "sampled" ),
        "nterms" : ( "simulator", "nterms" ),
        "imagesize" : ( "simulator", "imagesize" ),
        "cropsize" : ( "simulator", "cropsize" ),
        "xireference" : ( "simulator", "xireference" ), # new
        "centred" : ( "simulator", "centred" ), # new
        "source" : ( "source", "mode" ), # used differently
        "lightprofile" : ( "source", "lightprofile" ), 
        "n_sersic" : ( "source", "n_sersic" ), 
        "luminosity" : ( "source", "luminosity" ), 

        "amplitudes" : ( "lens", "amplitudefile" ),
        "einsteinradius" : ( "lens", "einsteinradius" ),
        "orientation" : ( "lens", "orientation" ),
        "ratio" : ( "lens", "ellipseratio" ), # relabelled
        "ellipseratio" : ( "lens", "ellipseratio" ), # duplicate
        "cluster" : ( "lens", "cluster" ), 

        "sigma" :  ( "source", "sigma" ),
        "sigma2" : ( "source", "sigma2" ),
        "theta" : ( "source", "theta" ),
        "x" : ( "position", "x" ),
        "y" : ( "position", "y" ),
        "phi" : ( "position", "phi" ),

        "directory" : ( "dataset", "directory" ) ,
        "reflines" : ( "annotation", "reflines" ),
        "criticalcurves" : ( "annotation", "criticalcurves" ),
        "criticalcurves" : ( "annotation", "criticalcurves" ),

        "maxcount" : ( "management", "maxcount" ),
        "showmask" : ( "management", "showmask" ),
        "mask" : ( "management", "mask" ),
        "join" : ( "management", "join" ),
        "family" : ( "management", "family" ),
        "psiplot" : ( "management", "psiplot" ),
        "kappaplot" : ( "management", "kappaplot" ),
        "apparent" : ( "management", "apparent" ),
        "actual" : ( "management", "actual" ),
        "filename" : ( "management", "filename" ),
        "index" : ( "management", "index" ),
        }

def getConfig( data, verbose=0 ):
      cfg = skel.copy()
      if verbose: print( "[getConfig]", data )
      flat = dict(data)
      if not "filename" in flat:
          try:
             flat["filename"] = data.name
          except:
              if verbose: print( "No filename given" )

      for k in flat:
          if k in paramap:
              key = paramap[k]
              d = cfg
              for subkey in key[:-1]:
                  try:
                     d = d[subkey]
                  except KeyError:
                      d[subkey] = {}
                      d = d[subkey]
              d[key[-1]] = flat[k]
              if verbose: print( f"[getConfig] {k} -> {key} ({flat[k]})" )
          else:
              if verbose: print( f"[getConfig] {k} -> not found ({flat[k]})" )
      return cfg
class CosmoParser(argparse.ArgumentParser):
  """Argument Parser for CosmoSim.
  The class provides standardised CLI options for many scripts
  in the CosmoSim suite.  
  """
  def getArgs(self,*a,**kw):
      """Parse the arguments and return the resulting `args` object.
      """
      if not hasattr(self,"_args"):
          self._args = super().parse_args(*a,**kw)
      return self._args
  def getConfig(self,*a,verbose=0,**kw):
      """Return the configuration as a nested `dict` structure.
      The arguments are parsed if required.
      """
      if not hasattr(self,"_args"):
          self._args = super().parse_args(*a,**kw)
      d = self._args.__dict__
      ks = [ k for k in d if d[k] is None ]
      for k in ks: del d[k]
      return getConfig( d, verbose=verbose )
  def __init__(self,*a,**kw):
    super().__init__(*a,**kw)

    # Model selection
    self.add_argument('-l', '--lens',
            help="lens model")
    self.add_argument('--cluster', help="cluster lens")
    self.add_argument('-L', '--model',
            help="simulation model")
    self.add_argument('-S', '--source',
            default="Spherical", help="source model")
    self.add_argument('-G', '--sampled',
                      action=argparse.BooleanOptionalAction, 
                      help="Sample the lens model")

    # Model Parameters
    self.add_argument('-x', '--x', type=float, default=0, help="x coordinate")
    self.add_argument('-y', '--y', type=float, default=0, help="y coordinate")
    self.add_argument('-T', '--phi', type=float, help="polar coordinate angle (phi)")

    self.add_argument('-s', '--sigma', type=float, default=20, help="source size (sigma)")
    self.add_argument('-2', '--sigma2', type=float, default=10, help="secondary source size (sigma2)")
    self.add_argument('-t', '--theta', type=float, default=45, help="source rotation angle (theta)")

    self.add_argument('-E', '--einsteinradius', type=float, default=20, help="Einstein radius")
    self.add_argument('-r', '--ratio', type=float, help="Ratio (usually Elliptic eccentricity)")
    self.add_argument('--orientation', default=0, type=float, help="Orientation of the lens")

    # Other parameters
    self.add_argument('-n', '--nterms', type=int,
                      help="Number of Roulettes terms")
    self.add_argument('-Z', '--imagesize', type=int, default=512, help="image size for calculations")
    self.add_argument('-z', '--cropsize', type=int, help="Final image size")

    # Output configuration 
    self.add_argument('-R', '--reflines',
                      action=argparse.BooleanOptionalAction,
                      help="Add reference (axes) lines")
    self.add_argument( '--criticalcurves',
                      action=argparse.BooleanOptionalAction,
                      help="Add critical curves to the distorted image")
    self.add_argument('-C', '--centred',
                      action=argparse.BooleanOptionalAction, 
                      help="centre image")
    self.add_argument('--maskradius',
            help="Set explicit masking radius")
    self.add_argument('-M', '--mask',
                      action=argparse.BooleanOptionalAction, 
            help="Mask out the convergence circle")
    self.add_argument('-m', '--showmask',
                      action=argparse.BooleanOptionalAction, 
            help="Mark the convergence circle")
    self.add_argument('-O', '--maskscale',default="0.9",
            help="Scaling factor for the mask radius")
    self.add_argument('-F', '--amplitudes',help="Amplitudes file")
    self.add_argument('--mldata',
                      action=argparse.BooleanOptionalAction, 
                      help="Make roulette output for ML without redundant colums")
    self.add_argument('-a', '--actual',
                      action=argparse.BooleanOptionalAction, 
                      help="write actual image")

    # Output file names
    self.add_argument('-D', '--directory',default="./",
            help="directory path (for output files)")

    # Using CSV data sets
    self.add_argument('-o', '--outfile',
            help="Output CSV file")
    self.add_argument('-i', '--csvfile',
            help="Dataset to generate (CSV file)")
    self.add_argument('--toml',
            help="TOML file defining a random dataset to generate.")

    self.add_argument('--maxcount',
            help="Maximum number of images to process")
    self.add_argument('--xireference',
                      action=argparse.BooleanOptionalAction,
            help="Use apparent position as reference for roulette amplitudes")
    # Command mode
    self.add_argument('--rnd', action='store_true',
            default=False, help="Generate random dataset")
    self.add_argument('--version', action='store_true',
            default=False, help="Show version number")
    self.add_argument('--roulette', type=str,
            help="Input file for roulette resimulation")

    self.add_argument('-v','--verbose', action="count",default=0,
            help="Debug output")

