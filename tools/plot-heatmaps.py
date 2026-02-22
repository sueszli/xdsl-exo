import os
import sys
from glob import glob
from math import log2
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import gmean

BENCHMARK_REPEATS = 16

if len(sys.argv) < 3:
    print("Usage: python plot-heatmaps.py <directory> <output>")
    sys.exit(1)

pwd = os.getcwd()
directory = pwd / Path(sys.argv[1])
procedures = glob("*.processed.csv", root_dir=directory)
output = pwd / Path(sys.argv[2])

print(f"Found {len(procedures)} files in {directory}")

heatmap_df = pd.DataFrame()

for proc in procedures:
    procedure = proc.replace(".processed.csv", "")
    filepath = os.path.join(directory, proc)

    df = pd.read_csv(filepath)

    del df["mean"]
    del df["stddev"]

    # df["geomean"] = df[(f"cputime_{i}" for i in range(BENCHMARK_REPEATS))].apply(gmean)

    exomlir = df[df["compiler"] == "exomlir"]
    del exomlir["compiler"]
    exomlir.set_index("size", inplace=True)
    exomlir["geomean"] = exomlir.apply(gmean, axis=1)

    exocc = df[df["compiler"] == "exocc"]
    del exocc["compiler"]
    exocc.set_index("size", inplace=True)
    exocc["geomean"] = exocc.apply(gmean, axis=1)

    heatmap_df[procedure] = exocc["geomean"] / exomlir["geomean"]


# sort columns by procedure
heatmap_df = heatmap_df.reindex(
    sorted(heatmap_df.columns),
    axis=1,
)

# create heatmap
plt.figure(figsize=(10, 6))

cmap = sns.diverging_palette(10, 133, as_cmap=True)

sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    cbar_kws={"label": "Speedup"},
    linewidths=0,
    center=1.0,
    yticklabels=tuple(f"$2^{{{int(log2(n))}}}$" for n in heatmap_df.index),
)

plt.title(f"{directory.name} Geomean of runtime of exocc / exomlir")
plt.xlabel("Size")
plt.ylabel("Procedure")
plt.yticks(rotation=0)
plt.tight_layout()

os.makedirs(os.path.dirname(output), exist_ok=True)

plt.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.1)

print(f"Saved heatmap to {output}")
