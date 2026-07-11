import chess
from encoding import encode_fen

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

x = encode_fen(fen)

print(x.shape)