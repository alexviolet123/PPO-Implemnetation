import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from HAGRPO import HAGRPO
from Environment import CartPoleEnvironment
import seaborn as sns

sns.set(style="darkgrid", font_scale=1.0, rc={
    "figure.figsize": (8, 5),
    "axes.titlesize": 18,   # 18 -> 20
    "axes.labelsize": 16,   
    "legend.fontsize": 14,  # 14 -> 16
    "xtick.labelsize": 12,  # 12 -> 14
    "ytick.labelsize": 12   # 12 -> 14
})

# General settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_iterations = 1000
reward_threshold = 500

# get hybrid weight

for G in [8,16]:
    hagrpo = HAGRPO(device=device, env=CartPoleEnvironment(), num_features=4, num_actions=2, group_size=G)
    _, weight_list = hagrpo.train()
    weights = np.array(weight_list)
    # srs = np.array(sr_list)
    # svs = np.array(sv_list)
    iterations = np.arange(len(weights)) 
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, weights, label='Hybrid Weight $\\hat{w}(r)$', linewidth=2)
    # plt.plot(iterations, srs, label='SR Weight \( \hat{w}(s) \)', linewidth=2)
    # plt.plot(iterations, svs, label='SV Weight \( \hat{w}(v) \)', linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel('Weight Value')
    plt.title('Hybrid Weight Dynamics')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'./Cartpole-v1/weight_hybrid{G}.pdf', dpi=300)