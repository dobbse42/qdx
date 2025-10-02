import sys
# sys.path.append(r'../')
from qdx.code_finder import CodeFinder
import os
import jax
import json
import time
import pickle
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def main():
    print("Hello from qdx!")
    print(f'Jax backend: {jax.default_backend()}')

    config = {
        "ENV_TYPE": "STANDARD", # Possibilities: "STANDARD", "MAX", "DELTA", "NOISE-AWARE"
        "N": 14,
        "K": 2,
        "D": 3,
        "MAX_STEPS": 30,
        "WHICH_GATES": ["cx", "h", "s"],
        "GRAPH": "equal-bipartite",
        "SOFTNESS": 0,
        "P_I": 0.9,
        "LAMBDA": 1,
        "SEED": 42,
        "LR": 5e-4,
        "NUM_ENVS": 64,
        "NUM_STEPS": 8,
        "TOTAL_TIMESTEPS": 2e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": .99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.1,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.05,
        "ACTIVATION": "relu",
        "HIDDEN_DIM": 200,
        "ANNEAL_LR": True,
        "NUM_AGENTS": 4,
        "COMPUTE_METRICS": True,
        "PRUNING-TYPE": "correction-only",
    }


    finder = CodeFinder(config)
    # Training is now more costly. It should train in approx 60 sec.
    params, metrics = finder.train()

    returns = metrics["returned_episode_returns"]
    lengths = metrics["returned_episode_lengths"]
    print(f"{returns}")
    print(f"{lengths}")

    data = finder.evaluate()
    print(f"{data[0]}")



if __name__ == "__main__":
    main()
