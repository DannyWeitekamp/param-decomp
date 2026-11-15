"""Parse an SPD stdout log (tqdm.write lines) into {step: {metric: value}} JSON.

Usage: python parse_log.py <output.log> <out.json>
"""
import json
import re
import sys

STEP_RE = re.compile(r"^--- Step (\d+) ---$")
KV_RE = re.compile(r"^((?:train|eval)/[^:]+): (-?[0-9.eE+-]+|nan|inf)$")

def parse(path: str) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    step = None
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").split("\r")[-1]
            if m := STEP_RE.match(line):
                step = int(m.group(1))
                out.setdefault(step, {})
            elif step is not None and (m := KV_RE.match(line)):
                out[step][m.group(1)] = float(m.group(2))
    return out

if __name__ == "__main__":
    d = parse(sys.argv[1])
    json.dump(d, open(sys.argv[2], "w"))
    print(f"{len(d)} steps parsed")
