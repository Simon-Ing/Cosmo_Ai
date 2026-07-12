#! /usr/bin/env python3

import pandas as pd

src = pd.read_csv( "roulette.csv" )

print( "tweak base set")
df = src.head(10)
df.to_csv( "roulette-base.csv" )

print( "tweak first set")
df["beta[5][2]"] = df["beta[5][2]"] * 10
df.to_csv( "roulette-beta.csv" )
df["beta[5][2]"] = 0
df.to_csv( "roulette-beta0.csv" )

print( "tweak second set")
df = src.head(10)
df["alpha[5][2]"] = df["alpha[5][2]"] * 10
df.to_csv( "roulette-alpha.csv" )
df["alpha[5][2]"] = 0
df.to_csv( "roulette-alpha0.csv" )

print( "tweak third set")
df = src.head(10)
lab = [ f"alpha[5][{s}]" for s in [ 2, 4, 6 ] ]
df[lab] = 10*df[lab]
df.to_csv( "roulette-alphas.csv" )
blab = [ f"beta[5][{s}]" for s in [ 2, 4, 6 ] ]
df[blab] = 10*df[blab]
df.to_csv( "roulette-ab.csv" )
df[lab+blab] = 0.01*df[lab+blab]
df.to_csv( "roulette-min.csv" )

print( "done tweaking datasets")
