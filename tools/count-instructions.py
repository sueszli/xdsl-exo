import os
import re
import sys
from pathlib import Path


def eprint(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


# get opt path and directory
if len(sys.argv) < 3:
    eprint("Usage: python count-instructions.py <opt> <directory> [ext]")
    sys.exit(1)

opt = Path(os.getcwd()) / Path(sys.argv[1])
directory = Path(os.getcwd()) / Path(sys.argv[2])
ext = sys.argv[3] if len(sys.argv) > 3 else "bc"

eprint(f"opt: {opt}")
eprint(f"target: {directory}")

print("compiler,level,proc,instcount")

INSTR_COUNT_REGEX = re.compile(r"(\d+) instcount\s+- Number of instructions \(of all types\)")

for bytecode in directory.glob(f"**/*.{ext}"):
    eprint(f"{bytecode}")

    result = os.popen(f"{opt} -passes=strip,instcount -disable-output -disable-verify -stats {bytecode} 2>&1").read()

    # get the first match
    match = INSTR_COUNT_REGEX.search(result)
    if match:
        instcount = match.group(1)
    else:
        eprint(f"Error: {bytecode} does not contain instcount")
        continue

    compiler = bytecode.parent.parent.name
    level = bytecode.parent.name
    proc = bytecode.stem

    print(f"{compiler},{level},{proc},{instcount}")
