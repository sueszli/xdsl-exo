import os

import numpy as np
import pandas as pd
import plotext as plt
import polars as pl
from plotnine import *


def main():
    if not os.path.exists("benchmarks/results.csv"):
        return
    df = pl.read_csv("benchmarks/results.csv")
    df_p = pl.concat([df.select([pl.col("kernel"), pl.col("n"), pl.col("exo_speedup").alias("s"), pl.lit("Exo").alias("v")]), df.select([pl.col("kernel"), pl.col("n"), pl.col("neon_speedup").alias("s"), pl.lit("NEON").alias("v")])])
    df_p = df_p.with_columns([pl.col("s").map_elements(lambda s: s if s >= 1 else -1.0 / s if s > 0 else -10.0, return_dtype=pl.Float64).alias("ts")])
    pdf = df_p.to_pandas()
    pdf["nn"] = pdf["n"].apply(lambda x: eval(str(x).replace("x", "*")) if "x" in str(x) else int(x))
    pdf["n_cat"] = pd.Categorical(pdf["n"], categories=pdf[["n", "nn"]].drop_duplicates().sort_values("nn")["n"].tolist(), ordered=True)

    out = "benchmarks/plots"
    os.makedirs(out, exist_ok=True)

    def save(p, n):
        p.save(os.path.join(out, f"{n}.png"))
        p.save(os.path.join(out, f"{n}.pdf"))

    # fmt: off
    thm = theme_minimal() + theme(legend_position="top", figure_size=(16, 10), axis_text_x=element_text(rotation=45, hjust=1))
    clr = scale_fill_manual(values=["#E69F00", "#56B4E9"])

    p1 = (ggplot(pdf, aes("n_cat", "ts", fill="v")) + geom_bar(stat="identity", position="dodge") + clr + facet_wrap("~kernel", scales="free_x") + thm + labs(title="Relative Performance", x="N", y="Transformed Speedup") + geom_hline(yintercept=0))
    save(p1, "plot1_all_speedups")

    p2 = (ggplot(pdf.assign(ps=np.where(pdf.s >= 1, pdf.s, 0)), aes("n_cat", "ps", fill="v")) + geom_bar(stat="identity", position="dodge") + clr + facet_wrap("~kernel", scales="free_x") + thm + labs(title="Positive Speedups", x="N", y="Speedup"))
    save(p2, "plot2_positive_only")

    p3 = (ggplot(pdf[pdf.nn > 0], aes("nn", "s", color="v", group="v")) + geom_line(size=1.5) + geom_point(size=3) + facet_wrap("~kernel", scales="free") + scale_color_manual(values=["#E69F00", "#56B4E9"]) + scale_x_log10() + theme_minimal() + theme(legend_position="top", figure_size=(16, 10)) + labs(title="Convergence", x="N (Log)", y="Speedup"))
    save(p3, "plot3_convergence")
    # fmt: on

    with open(os.path.join(out, "plots_ascii.txt"), "w") as f:
        f.write("XNUMPY BENCHMARK RESULTS (ASCII)\n==============================\n\n")
        for k in pdf.kernel.unique():
            plt.clf()
            plt.theme("pro")
            for v in pdf.v.unique():
                d = pdf[(pdf.kernel == k) & (pdf.v == v)].sort_values("nn")
                plt.plot(d.nn.tolist(), d.s.tolist(), label=v, marker="dot")
            plt.title(f"Speedup: {k}")
            plt.xscale("log")
            f.write(f"--- {k} ---\n" + plt.build() + "\n\n")


if __name__ == "__main__":
    main()
