import numpy as np
import matplotlib.pyplot as plt
import torch
from PPO import PPO
from GRPO import GRPO
from HAGRPO import HAGRPO
from Environment import CartPoleEnvironment
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Gs = [2, 4, 8, 16, 32, 64]
for G in Gs:
    plt.figure(figsize=(12, 8))
    grpo = GRPO(device=device, 
                env=CartPoleEnvironment(), 
                num_features=4, 
                num_actions=2,
                group_size=G
    )
    grpo.train()
    grpo_rewards = grpo.rewards_per_iteration
    plt.plot(np.arange(len(grpo_rewards)), grpo_rewards, label=f'Group Size {G}')
    plt.title(f'GRPO with Group Size {G}')
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.grid()
    plt.legend()
plt.savefig(f'Effective_group_size.pdf')
