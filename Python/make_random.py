from random import randint
import json

data = {}

data["images"] = [[[randint(0, 255) for i in range(500)] for ii in range(500)] for iii in range(10)]
data["params"] = [[randint(0, 50) for j in range(3)] for jj in range(10)]

with open("test_dataset.json", "w") as outfile:
    json.dump(data, outfile)
