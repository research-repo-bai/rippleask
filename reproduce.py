"""Reproduce the annotation-audit numbers in the paper from labels.csv (no API calls).

Usage: python reproduce.py
"""
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

rows = list(csv.DictReader(open(Path(__file__).parent / "labels.csv")))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


cases = {r["case"] for r in rows}
print(f"annotations: {len(rows)}  cases: {len(cases)}")

inp = Counter(r["input_label"] for r in rows)
resolved = inp["moot"] + inp["consequential"]
lo, hi = wilson(inp["moot"], resolved)
print(f"\ninput labels: {dict(inp)}")
print(f"moot share among resolved: {inp['moot']}/{resolved} = {inp['moot']/resolved:.0%}  95% CI [{lo:.0%}, {hi:.0%}]")

print("\nby kind (moot / resolved):")
bk = defaultdict(lambda: Counter())
for r in rows:
    bk[r["kind"]][r["input_label"]] += 1
for kind, c in sorted(bk.items(), key=lambda kv: -(kv[1]["moot"] + kv[1]["consequential"])):
    res = c["moot"] + c["consequential"]
    print(f"  {kind:26} {c['moot']}/{res}")

tested = [r for r in rows if r["perturbation_tested"] == "yes" and r["input_label"] == "moot"]
latent_all = [r for r in tested if r["perturbation_label"] == "latent"]
controls = [r for r in rows if r["perturbation_tested"] == "yes" and r["input_label"] == "consequential"]
print(f"\nperturbation study: {len(tested)} moot annotations with a valid run; {len(controls)} consequential controls")
print(f"latent (all valid): {len(latent_all)}/{len(tested)} = {len(latent_all)/len(tested):.0%}")
maj = [r for r in tested if r["reviewer_votes"] not in ("", None) and int(r["reviewer_votes"]) >= 2]
maj_lat = [r for r in maj if r["perturbation_label"] == "latent"]
lo, hi = wilson(len(maj_lat), len(maj))
print(f"latent (majority-reviewed): {len(maj_lat)}/{len(maj)} = {len(maj_lat)/len(maj):.0%}  95% CI [{lo:.0%}, {hi:.0%}]")
