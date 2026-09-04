import json
from pathlib import Path

DATA_DIR = Path("data/selfplay_jsons")

FILES = [
    DATA_DIR / "uncertainty_stats_1-10.json",
    DATA_DIR / "uncertainty_stats_11-20.json",
    DATA_DIR / "uncertainty_stats_21-30.json",
    DATA_DIR / "uncertainty_stats_31-40.json",
    DATA_DIR / "uncertainty_stats_41-50.json",
    DATA_DIR / "uncertainty_stats_51-60.json",
]

OUTPUT = DATA_DIR / "uncertainty_stats_1-60.json"

merged = []

for path in FILES:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"{path} ne contient pas une liste JSON "
            f"(type trouvé : {type(data).__name__})"
        )

    print(f"{path.name}: {len(data):,} records")

    merged.extend(data)

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(
        merged,
        f,
        ensure_ascii=False,
        indent=2
    )

print()
print("=" * 50)
print("MERGE COMPLETED")
print("=" * 50)
print(f"Files merged : {len(FILES)}")
print(f"Total records: {len(merged):,}")
print(f"Output       : {OUTPUT}")