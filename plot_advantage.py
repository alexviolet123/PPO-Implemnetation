import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from PPO import PPO
from GRPO import GRPO
from HAGRPO import HAGRPO
import seaborn as sns
from Environment import CartPoleEnvironment
import gym

sns.set(style="darkgrid", font_scale=1.0, rc={
    "figure.figsize": (8, 5),
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "legend.fontsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
repeat = 5
num_iterations = 1000

def run_algorithm(algo_name, G=None):
    advantages_all = []

    for _ in tqdm(range(repeat), desc=f"{algo_name} G={G if G else '--'}"):
        if algo_name == "PPO":
            algo = PPO(env=gym.make("CartPole-v1"), num_features=4, num_actions=2, gamma=0.98, lam=1)
            _, _, advantages = algo.run_model()
        elif algo_name == "GRPO":
            algo = GRPO(device=device, env=CartPoleEnvironment(), num_features=4, num_actions=2, group_size=G)
            _, advantages = algo.train()
        elif algo_name == "HAGRPO":
            algo = HAGRPO(device=device, env=CartPoleEnvironment(), num_features=4, num_actions=2, group_size=G)
            _, _, advantages = algo.train()
        else:
            raise ValueError("Unknown algorithm")

        advantages_all.append(np.array(advantages[:num_iterations]))

    return np.array(advantages_all)

def pad_and_plot_advantage(adv_dict, title, filename, pad_value=0.0):
    all_adv = sum(adv_dict.values(), [])
    max_len = max(len(a) for a in all_adv)

    def pad(arr):
        return np.concatenate([arr, np.full(max_len - len(arr), pad_value)]) if len(arr) < max_len else arr[:max_len]

    plt.figure(figsize=(10, 6))
    for label, adv_list in adv_dict.items():
        adv_padded = np.array([pad(a) for a in adv_list])
        mean = np.mean(adv_padded, axis=0)
        stderr = np.std(adv_padded, axis=0) / np.sqrt(len(adv_list))
        ci95 = 1.96 * stderr
        plt.plot(mean, label=label)
        plt.fill_between(np.arange(max_len), mean - ci95, mean + ci95, alpha=0.1)

    plt.xlabel("Iterations")
    plt.ylabel("Advantage")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)

def main():
    G = 16
    adv_dict = {}

    # Run PPO
    ppo_advantages = run_algorithm("PPO")
    adv_dict["PPO"] = list(ppo_advantages)

    # Run GRPO
    grpo_advantages = run_algorithm("GRPO", G=G)
    adv_dict["GRPO"] = list(grpo_advantages)

    # Run HAGRPO
    hagrpo_advantages = run_algorithm("HAGRPO", G=G)
    adv_dict["HAGRPO"] = list(hagrpo_advantages)

    # Plot advantages
    pad_and_plot_advantage(adv_dict, f"Advantage Comparison (G={G})", f"advantage_comparison_g{G}.pdf")

if __name__ == "__main__":
    main()
