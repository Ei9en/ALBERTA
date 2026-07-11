from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from src.models.resnet import ChessResNet

model = ChessResNet()

x = torch.randn(8, 19, 8, 8)

y = model(x)

print(y.shape)
print("Parameters:", sum(p.numel() for p in model.parameters()))