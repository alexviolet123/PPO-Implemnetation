import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from PPO import PPO
from GRPO import GRPO
from HAGRPO import HAGRPO
from HAGRPO_fix_weight import HAGRPO_fix
from Environment import CartPoleEnvironment
import seaborn as sns
import gym
import pickle

sns.set(style="darkgrid", font_scale=2.0, rc={
    "figure.figsize": (12, 7),
    "axes.titlesize": 32,     # 图标题
    "axes.labelsize": 28,     # 坐标轴标签
    "legend.fontsize": 24,    # 图例
    "xtick.labelsize": 22,    # x 轴刻度
    "ytick.labelsize": 22     # y 轴刻度
})

# General settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
repeat = 5
num_iterations = 1000
reward_threshold = 500


def run_algorithm(algo_name, G=None):
    rewards_all = []
    steps_all = []

    for _ in tqdm(range(repeat), desc=f"{algo_name} G={G if G else '--'}"):
        if algo_name == "PPO":
            algo = PPO(env=gym.make("CartPole-v1"), num_features=4, num_actions=2, gamma=0.98, lam=1)
            rewards, steps, _ = algo.run_model()
        elif algo_name == "GRPO":
            algo = GRPO(device=device, env=CartPoleEnvironment(), num_features=4, num_actions=2, group_size=G)
            steps, _ = algo.train()
            rewards = algo.rewards_per_iteration
        elif algo_name == "HAGRPO":
            algo = HAGRPO(device=device, env=CartPoleEnvironment(), num_features=4, num_actions=2, group_size=G)
            steps, _, _ = algo.train()
            rewards = algo.rewards_per_iteration
        elif algo_name == "HAGRPO_fix":
            algo = HAGRPO_fix(device=device, env=CartPoleEnvironment(), num_features=4, num_actions=2, group_size=G)
            steps = algo.train()
            rewards = algo.rewards_per_iteration
        else:
            raise ValueError("Unknown algorithm")

        rewards_all.append(np.array(rewards[:num_iterations]))
        steps_all.append(steps)

    return np.array(rewards_all), np.array(steps_all)


def pad_and_plot(reward_dict, title, filename, pad_value=500):
    all_rewards = sum(reward_dict.values(), [])
    max_len = max(len(r) for r in all_rewards)

    def pad(arr):
        return np.concatenate([arr, np.full(max_len - len(arr), pad_value)]) if len(arr) < max_len else arr[:max_len]

    plt.figure(figsize=(10, 6))
    for label, rewards_list in reward_dict.items():
        rewards_padded = np.array([pad(r) for r in rewards_list])
        mean = np.mean(rewards_padded, axis=0)
        stderr = np.std(rewards_padded, axis=0) / np.sqrt(len(rewards_list))
        ci95 = 1.96 * stderr
        plt.plot(mean, label=label)
        plt.fill_between(np.arange(max_len), mean - ci95, mean + ci95, alpha=0.1)

    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)


Gs = [6,8,10,12,14,16]
reward_dict = {}
step_dict = {}

# Run the PPO algorithm
ppo_rewards, ppo_steps = run_algorithm("PPO")

# Store the results in dictionaries
reward_dict = {}
step_dict = {}

reward_dict["PPO"] = list(ppo_rewards)
step_dict["PPO"] = ppo_steps

# Save the dictionaries to a file
with open('ppo_data.pkl', 'wb') as f:
    pickle.dump((reward_dict, step_dict), f)

for G in Gs:
    # Run GRPO
    grpo_rewards, grpo_steps = run_algorithm("GRPO", G=G)
    reward_dict["GRPO"] = list(grpo_rewards)
    step_dict["GRPO"] = grpo_steps

    # Run HAGRPO
    hagrpo_rewards, hagrpo_steps = run_algorithm("HAGRPO", G=G)
    reward_dict["HAGRPO"] = list(hagrpo_rewards)
    step_dict["HAGRPO"] = hagrpo_steps

    # Run HAGRPO_fix
    # hagrpo_fix_rewards, hagrpo_fix_steps = run_algorithm("HAGRPO_fix", G=G)
    # reward_dict["HAGRPO_fix"] = list(hagrpo_fix_rewards)
    # step_dict["HAGRPO_fix"] = hagrpo_fix_steps

    # Plot rewards
    pad_and_plot(reward_dict, f"Reward Comparison (G={G})", f"reward_comparison_g{G}.pdf")
    np.save(f"rewards_g{G}.npy", reward_dict)
    # read: reward_dict = np.load("rewards_g10.npy", allow_pickle=True).item()

    # Print steps report
    print("\n=== Success Steps Report ===")
    for name, steps in step_dict.items():
        print(f"{name}: mean = {np.mean(steps):.2f}, std = {np.std(steps):.2f}")
