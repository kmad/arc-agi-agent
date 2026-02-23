"""Take a screenshot of the current game board and save as PNG."""

import sys
import numpy as np
from arc_agi import Arcade, OperationMode
from game_state import frame_to_image_bytes

game_id = sys.argv[1] if len(sys.argv) > 1 else "ls20"
output = sys.argv[2] if len(sys.argv) > 2 else "board.png"

arcade = Arcade(operation_mode=OperationMode.OFFLINE)
env = arcade.make(game_id)
obs = env.reset()
frame = np.array(obs.frame)[0]

png_bytes = frame_to_image_bytes(frame)
with open(output, "wb") as f:
    f.write(png_bytes)

print(f"Saved {output} ({len(png_bytes)} bytes) - {game_id} initial board")
print(f"Grid values present: {np.unique(frame)}")
