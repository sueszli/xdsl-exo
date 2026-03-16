import csv
import inspect
import json
from pathlib import Path

WEIGHTS_PATH = Path(__file__).parent / "weights.json"


def dump_weights(state_dict) -> None:
    # serialize extracted weights to JSON file
    WEIGHTS_PATH.write_text(json.dumps({k: [[v.data for v in row] for row in mat] for k, mat in state_dict.items()}))


def assert_weights_match(state_dict, atol: float = 1e-5) -> None:
    # assert cur == ref element-wise within atol. report worst violation on failure
    assert WEIGHTS_PATH.exists(), f"weights file not found: {WEIGHTS_PATH}"
    ref = json.load(open(WEIGHTS_PATH))
    cur = {k: [[v.data for v in row] for row in mat] for k, mat in state_dict.items()}

    assert set(ref) == set(cur), f"key mismatch: ref={set(ref)-set(cur)} cur={set(cur)-set(ref)}"

    max_diff, max_loc, violations, total = 0.0, "", 0, 0
    for k in ref:
        assert len(ref[k]) == len(cur[k]) and len(ref[k][0]) == len(cur[k][0]), f"shape mismatch '{k}': {len(ref[k])}x{len(ref[k][0])} vs {len(cur[k])}x{len(cur[k][0])}"
        for i, (rr, cr) in enumerate(zip(ref[k], cur[k])):
            for j, (r, c) in enumerate(zip(rr, cr)):
                d = abs(r - c)
                total += 1
                violations += d > atol
                if d > max_diff:
                    max_diff, max_loc = d, f"{k}[{i}][{j}]"

    assert violations == 0, f"weights mismatch (atol={atol}): {violations}/{total} params exceed tolerance, max diff={max_diff:.2e} at {max_loc}"


def save_times(step_times: list[float]) -> None:
    name = Path(inspect.stack()[1].filename).stem
    path = Path(__file__).parent / "times" / f"{name}.csv"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "time_ms"])
        w.writerows([[i + 1, f"{t * 1000:.3f}"] for i, t in enumerate(step_times)])
