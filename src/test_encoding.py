import json
from encoding import encode_fen

# Prend la première ligne du dataset
with open("data/processed/positions.jsonl") as f:
    sample = json.loads(next(f))

print("Sample:")
print(sample)

# Encodage
x = encode_fen(sample["fen"])

print("\nTensor shape:")
print(x.shape)

print("\nNumber of pieces:")
for i in range(12):
    print(i, x[i].sum().item())

print("\nGlobal planes:")
for i in range(12, 19):
    print(i, x[i].min().item(), x[i].max().item())

from encoding import encode_fen

import chess

fen = "8/8/8/8/8/8/4P3/4K3 w - - 0 1"

x = encode_fen(fen)

print(x[0]) # pion blanc