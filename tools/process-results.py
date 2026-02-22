import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

if len(sys.argv) != 3:
    print("Usage: python preprocess-csv.py <input> <output>")
    sys.exit(1)

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

SIZE_REGEX = re.compile(r"/(\d+)")

with open(input_path) as fin:
    with open(output_path, "w") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "compiler",
                "size",
                *(f"cputime_{i}" for i in range(16)),
                "mean",
                "stddev",
            ],
        )
        writer.writeheader()
        data = defaultdict(list)

        current_size = None
        results = None
        colcounter = 0

        for row in reader:
            name = row["name"]
            size = int(SIZE_REGEX.search(name).group(1))

            if current_size is None:
                current_size = size
                results = {
                    "compiler": "exomlir" if "exomlir" in name else "exocc",
                    "size": size,
                    "mean": None,
                    "stddev": None,
                }
            elif current_size != size:
                # start new row
                writer.writerow(results)
                results = {
                    "compiler": "exomlir" if "exomlir" in name else "exocc",
                    "size": size,
                    "mean": None,
                    "stddev": None,
                }
                current_size = size
                colcounter = 0

            # get cpu time
            if not name.endswith("_mean") and not name.endswith("_stddev") and not name.endswith("_median") and not name.endswith("_cv"):
                results[f"cputime_{colcounter}"] = float(row["cpu_time"])
                colcounter += 1
                continue

            if name.endswith("_mean"):
                results["mean"] = float(row["cpu_time"])

            if name.endswith("_stddev"):
                results["stddev"] = float(row["cpu_time"])

        # write last row
        writer.writerow(results)
