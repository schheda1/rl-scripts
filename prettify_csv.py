import sys

import pandas as pd

csv = sys.argv[1]
df = pd.read_csv(csv, sep=";")
pretty_string = df.to_string(index=False)
with open(csv, "w") as f:
    f.write(pretty_string + "\n")
