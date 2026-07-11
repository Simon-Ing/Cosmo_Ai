#! /usr/bin/env python3
# (C) 2023,2026: Hans Georg Schaathun <georg@schaathun.net>

"""
Generate an image for given parameters.
"""

import sys
import pandas as pd

from .Image import drawAxes
from .datagen import crop

from .CosmoSimPy import RouletteRegenerator
from .CLI.Resim import Resim

def processResim(frame,param,maxcount=None,verbose=0):
    """
    Bulk simulation from a dataset.
    The `frame` is normally a pandas DataFrame, with a row for
    each lensing system.
    """
    count = 1
    print( "[processResim]", verbose )
    print( "[processResim]", param )
    for index,row in frame.iterrows():
        if verbose: print( "[roulettegen.py] Processing", index )
        param.setRow( row )
        print( f"[processResim] {index}", param )
        imsim = Resim( row, param, verbose=verbose )
        imsim.saveImage()

        if param.get( "actual" ): imsim.getActual()

        if maxcount is not None:
            count += 1
            if count > maxcount: break


def rgen(args,param):
    """
    This is the main procedure of the script, simulating a dataset of
    roulette amplitudes based on a CLI args argument.
    """
    verbose = args.verbose
    if args.roulette:
        if verbose: print( "Load CSV file:", args.roulette )
        try:
            frame = pd.read_csv(args.roulette)
        except Exception as e:
            print( "Fails to open file:", args.roulette )
            raise e

    else:
        raise Exception( "No CSV file given." )

    maxcount = param.get( ("management", "maxcount" ), None )
    processResim(frame,param,maxcount,verbose=verbose)

    if verbose: print( "[roulettegen.py] Done" )

if __name__ == "__main__":
    sys.exit( "[roulettegen.py] Deprecated." )
