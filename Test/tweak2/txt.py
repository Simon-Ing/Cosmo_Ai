#! /usr/bin/env python3

import pandas as pd

src = pd.read_csv( "roulette.csv" )

df = src[["filename","chi","einsteinR","x","y","sigma","sigma2","theta"]]

for i,row in df.iterrows():
    fn = row["filename"].split(".")[0] + ".txt"
    with open(fn,"w") as f:
        f.write(row.to_string())
        f.close()
