import json
from collections import Counter

with open("data/uncertainty_stats.json") as f:
    data = json.load(f)

valid = [
    r for r in data
    if "fen" in r
    and "U" in r
]

valid.sort(
    key=lambda r: float(r["U"]),
    reverse=True,
)

for n in [100, 1000, 10000]:
    subset = valid[:n]

    sides = Counter(
        r["fen"].split()[1]
        for r in subset
    )

    print(f"\nTOP {n} BY U")
    print("=" * 40)
    print(
        f"White: {sides['w']} "
        f"({100 * sides['w'] / n:.1f}%)"
    )
    print(
        f"Black: {sides['b']} "
        f"({100 * sides['b'] / n:.1f}%)"
    )