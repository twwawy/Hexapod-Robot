#!/usr/bin/env python3
"""Train the combined flat walking-and-turning residual-RL curriculum."""

from train_rough_terrain import main


if __name__ == "__main__":
    main(default_task="command")
