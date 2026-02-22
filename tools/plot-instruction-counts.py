import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pwd = os.getcwd()

os.makedirs(f"{pwd}/build/plots", exist_ok=True)

# setup schema
df = pd.read_csv(f"{pwd}/build/instrcount.csv")
df["instcount"] = df["instcount"].astype(int)
df["level"] = df["level"].astype(str)

sns.set_theme(style="whitegrid")

# xdsl colors
compiler_colors = ["#33a02c", "#1f78b4"]  # green, blue
compilers = sorted(df["compiler"].unique())
palette = {compiler: color for compiler, color in zip(compilers, compiler_colors)}

for level in sorted(df["level"].unique()):
    df_level = df[df["level"] == level].sort_values(by="proc")

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df_level, x="proc", y="instcount", hue="compiler", palette=palette)
    ax.set_title(f"Instruction Count - {level}")
    ax.set_xlabel("Procedure")
    ax.set_ylabel("Instruction Count")
    ax.tick_params(axis="x", rotation=45)
    plt.legend(title="Compiler")
    plt.tight_layout()

    output_path = f"{pwd}/build/plots/{level}/instcount.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300)

    plt.close()
