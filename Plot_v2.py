import numpy as np
import matplotlib.pyplot as plt
import torch
from PPO import PPO
from GRPO import GRPO
from HAGRPO import HAGRPO
from Environment import CartPoleEnvironment, LunarLanderEnvironment
import gym
import seaborn as sns
from tqdm import tqdm

sns.set(style="darkgrid", font_scale=1.0, rc={
    "figure.figsize": (8, 5),
    "axes.titlesize": 22,   # 原18 -> 20
    "axes.labelsize": 20,   # 保持不变
    "legend.fontsize": 14,  # 原14 -> 16
    "xtick.labelsize": 16,  # 原12 -> 14
    "ytick.labelsize": 16   # 原12 -> 14
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Plot PPO GRPO HAGRPO rewards
def plot_results(ppo_rewards, grpo_rewards, hagrpo_rewards, num_iterations, G=16):
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(ppo_rewards)), ppo_rewards, label='PPO', color='blue')
    plt.plot(np.arange(len(grpo_rewards)), grpo_rewards, label='GRPO', color='orange')
    plt.plot(np.arange(len(hagrpo_rewards)), hagrpo_rewards, label='HAGRPO', color='green')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.title('Comparison of PPO, GRPO, and HAGRPO on LunarLander-v2')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.show()
    plt.savefig(f'ppo_grpo_hagrpo_rewards_{G}.pdf')


repeat = 20
ppo_steps = []
grpo_steps = []
hagrpo_steps = []
for i in tqdm(range(repeat)):
    ppo = PPO(
        env=gym.make('LunarLander-v2'),
        num_features=8,
        num_actions=4,
        lam=0.98,
        gamma=0.999
    )
    grpo = GRPO(device=device, 
                env=LunarLanderEnvironment(), 
                num_features=8, 
                num_actions=4,
                group_size=16
    )
    hagrpo = HAGRPO(device=device, 
                    env=LunarLanderEnvironment(), 
                    num_features=8,
                    num_actions=4,
                    group_size=16
    )
    ppo_rewards, ppo_step, _ = ppo.run_model()
    grpo_step ,_ = grpo.train()
    grpo_rewards = grpo.rewards_per_iteration
    hagrpo_step, _ = hagrpo.train()
    hagrpo_rewards = hagrpo.rewards_per_iteration

np.save("ppo_steps_v2_g16.npy", ppo_steps)
np.save("grpo_steps_v2_g16.npy", grpo_steps)
np.save("hagrpo_steps_v2_g16.npy", hagrpo_steps)

# calculate mean
ppo_steps = np.load("ppo_steps_v2_g16.npy")
grpo_steps = np.load("grpo_steps_v2_g16.npy")
hagrpo_steps = np.load("hagrpo_steps_v2_g16.npy")
ppo_mean = np.mean(ppo_steps)
grpo_mean = np.mean(grpo_steps)
hagrpo_mean = np.mean(hagrpo_steps)
print(f"PPO mean steps v2 g 16 : {ppo_mean}")
print(f"GRPO mean steps v2 g 16: {grpo_mean}")
print(f"HAGRPO mean steps v2 g 16: {hagrpo_mean}")

