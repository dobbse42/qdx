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
import stim


def visualize_circuit(data):
    gates = data[0]['gates']
    actions = []

    for g in gates:
        gate_name = g.split("(")[0].split(".")[1]
        qubit_ids = g.split("(")[1].split(")")[0]
        instruction = '.append("%s", [%s])' % (gate_name, qubit_ids)
        actions.append(instruction)

    circ = stim.Circuit()
    for action in actions:
        eval('circ%s' % action)

    diagram = circ.diagram('timeline-svg')
    with open('diagram.svg', 'w') as f:
        f.write(str(diagram))

    return


def main():
    print("Hello from qdx!")
    print(f'Jax backend: {jax.default_backend()}')

    config = {
        "ENV_TYPE": "STANDARD", # Possibilities: "STANDARD", "MAX", "DELTA", "NOISE-AWARE"
        "N": 5,
        "K": 1,
        "D": 3,
        "MAX_STEPS": 20,
        "WHICH_GATES": ["cx", "h", "s"],
        "GRAPH": "All-to-All",
        "SOFTNESS": 1,
        "P_I": 0.9,
        "LAMBDA": 10,
        "SEED": 42,
        "LR": 1e-3,
        "NUM_ENVS": 8,
        "NUM_STEPS": 20,
        "TOTAL_TIMESTEPS": 2e6,
        "UPDATE_EPOCHS": 3,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.02,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.25,
        "ACTIVATION": "relu",
        "HIDDEN_DIM": 32,
        "ANNEAL_LR": True,
        "NUM_AGENTS": 4,
        "COMPUTE_METRICS": True,
        "PRUNING-TYPE": "correction-only",
    }


    # config = {
    #     "ENV_TYPE": "STANDARD", # Possibilities: "STANDARD", "MAX", "DELTA", "NOISE-AWARE"
    #     "N": 7,
    #     "K": 1,
    #     "D": 3,
    #     "MAX_STEPS": 30,
    #     "WHICH_GATES": ["cx", "h"],
    #     "GRAPH": "All-to-All", # Possibilities: "All-to-All, NN-1, NN-2, equal-bipartite"
    #     "SOFTNESS": 1,
    #     "P_I": 0.9,
    #     "LAMBDA": 10,
    #     "SEED": 42,
    #     "LR": 5e-4,
    #     "NUM_ENVS": 16,
    #     "NUM_STEPS": 20,
    #     "TOTAL_TIMESTEPS": 2e6,
    #     "UPDATE_EPOCHS": 4,
    #     "NUM_MINIBATCHES": 8,
    #     "GAMMA": .99,
    #     "GAE_LAMBDA": 0.95,
    #     "CLIP_EPS": 0.1,
    #     "ENT_COEF": 0.01,
    #     "VF_COEF": 0.5,
    #     "MAX_GRAD_NORM": 0.05,
    #     "ACTIVATION": "relu",
    #     "HIDDEN_DIM": 200,
    #     "ANNEAL_LR": True,
    #     "NUM_AGENTS": 32,
    #     "COMPUTE_METRICS": True,
    #     "PRUNING-TYPE": "correction-only",
    # }

    print(f"config: {config}")
    finder = CodeFinder(config)
    # Training is now more costly. It should train in approx 60 sec.
    params, metrics = finder.train()

    returns = metrics["returned_episode_returns"]
    lengths = metrics["returned_episode_lengths"]
    print(f"{returns}")
    print(f"{lengths}")

    data = finder.evaluate()
    print(f"{data[0]}")
    print(f"data: {data}")

    visualize_circuit(data)




if __name__ == "__main__":
    main()
