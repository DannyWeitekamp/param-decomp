"""Compare train losses of two runs' metrics.jsonl (last occurrence of each step wins).

Usage: python vpd/tools/compare_ab.py <A/metrics.jsonl> <B/metrics.jsonl> <resume_step>
"""
import json, sys
def load(p):
    d = {}
    for line in open(p):
        r = json.loads(line); d.setdefault(r["step"], {}).update(r)
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
resume_step = int(sys.argv[3])
keys = ["train/loss/total", "train/loss/StochasticReconSubsetLoss", "train/loss/PersistentPGDReconLoss",
        "train/loss/ImportanceMinimalityLoss", "train/loss/FaithfulnessLoss"]
print(f"{'step':>5} " + " ".join(f"{k.split('/')[-1][:22]:>24}" for k in keys))
for s in sorted(set(a) & set(b)):
    if "train/loss/total" not in a[s] or "train/loss/total" not in b[s]: continue
    print(f"{s:>5}{'*' if s >= resume_step else ' '}" + " ".join(f"{a[s][k]:>11.5g}/{b[s][k]:<11.5g}" for k in keys))
for lo, hi, name in [(0, resume_step - 1, "pre-resume"), (resume_step, 10**9, "post-resume")]:
    ss = [s for s in set(a) & set(b) if lo <= s <= hi and "train/loss/total" in a[s] and "train/loss/total" in b[s]]
    if not ss: continue
    for k in keys:
        ma = sum(a[s][k] for s in ss) / len(ss); mb = sum(b[s][k] for s in ss) / len(ss)
        print(f"{name:>12} mean over {len(ss)} logs {k.split('/')[-1]:>28}: A={ma:.6g} B={mb:.6g} rel.diff={(mb-ma)/ma:+.2%}")
