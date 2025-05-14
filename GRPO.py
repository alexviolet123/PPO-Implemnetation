import torch
import torch.optim as optim
from typing import Tuple, List
import numpy as np
from Environment import CartPoleEnvironment
from Policy import PolicyModel


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

class GRPO:
    def __init__(self, env, num_features, num_actions, gamma=0.994, kl_beta=0.01,
                 policy_lr=3e-4, entropy_coeff=0.01, clip_epsilon=0.2, epochs=10,
                 group_size=16, max_steps_per_episode=500, epsilon_std=1e-8,
                 num_iterations=1000, render_interval=-1, print_interval=10,
                 policy_hidden_sizes=(40, 35, 30), batch_size=128,
                 device=torch.device("cpu")):

        self.env = env
        self.gamma = gamma
        self.kl_beta = kl_beta
        self.entropy_coeff = entropy_coeff
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.group_size = group_size
        self.max_steps = max_steps_per_episode
        self.epsilon_std = epsilon_std
        self.num_iterations = num_iterations
        self.render_interval = render_interval
        self.print_interval = print_interval
        self.batch_size = batch_size
        self.device = device

        self.policy = PolicyModel(num_features, num_actions, *policy_hidden_sizes).to(self.device)
        self.policy_ref = PolicyModel(num_features, num_actions, *policy_hidden_sizes).to(self.device)
        self.policy_ref.load_state_dict(self.policy.state_dict())
        self.policy_ref.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.threshold = 500.0

        self.rewards_per_iteration = []
        self.policy_losses = []
        self.entropies = []
        self.kl_divergences = []

    def collect_rollouts(self):
        states_list, actions_list, log_probs_list, rewards_list = [], [], [], []
        ep_rewards = []
        self.policy.eval()
        while len(states_list) * len(states_list[0]) < self.batch_size:  # Keep collecting until batch size is met
            state = self.env.reset(render=(self.render_interval > 0))
            done, step, episode_reward = False, 0, 0
            rollout_states, rollout_actions, rollout_log_probs, rollout_rewards = [], [], [], []
            while not done and step < self.max_steps:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device) if not isinstance(state, torch.Tensor) else state.to(self.device)
                with torch.no_grad():
                    probs = self.policy(state_tensor)
                    dist = torch.distributions.Categorical(probs=probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                next_state, reward, done = self.env.step(action.item())
                rollout_states.append(state_tensor)
                rollout_actions.append(action)
                rollout_log_probs.append(log_prob)
                rollout_rewards.append(reward)
                state = next_state
                episode_reward += reward
                step += 1
            if rollout_states:
                states_list.append(torch.stack(rollout_states))
                actions_list.append(torch.stack(rollout_actions))
                log_probs_list.append(torch.stack(rollout_log_probs))
                rewards_list.append(rollout_rewards)
                ep_rewards.append(episode_reward)
            else:
                states_list.append(torch.empty((0, self.policy.model[0].in_features), device=self.device))
                actions_list.append(torch.empty((0,), dtype=torch.long, device=self.device))
                log_probs_list.append(torch.empty((0,), device=self.device))
                rewards_list.append([])
                ep_rewards.append(0.0)
        return states_list, actions_list, log_probs_list, rewards_list, ep_rewards


    def compute_advantages(self, rewards_list):
        temp_advantages, all_advantages = [], []
        for rewards in rewards_list:
            returns = torch.zeros(len(rewards), device=self.device)
            ret = 0.0
            for t in reversed(range(len(rewards))):
                ret = rewards[t] + self.gamma * ret
                returns[t] = ret
            temp_advantages.append(returns)
            all_advantages.extend(returns.cpu().numpy())
        if len(all_advantages) > 1:
            mean, std = np.mean(all_advantages), np.std(all_advantages)
        elif len(all_advantages) == 1:
            mean, std = all_advantages[0], 0.0
        else:
            mean, std = 0.0, 0.0
        normalized = [(ret - mean) / (std + self.epsilon_std) if ret.nelement() > 0 else ret for ret in temp_advantages]
        return normalized

    def update_policy(self, states_list, actions_list, log_probs_list, advantages_list):
        try:
            states = torch.cat(states_list, dim=0).to(self.device)
            actions = torch.cat(actions_list, dim=0).to(self.device)
            log_probs_old = torch.cat(log_probs_list, dim=0).to(self.device)
            advantages = torch.cat(advantages_list, dim=0).to(self.device)
        except RuntimeError as e:
            print("Concatenation error:", e)
            raise e
        advantages = advantages.detach()
        log_probs_old = log_probs_old.detach()
        self.policy_ref.eval()
        total_loss, total_kl, total_entropy = 0.0, 0.0, 0.0
        for _ in range(self.epochs):
            probs = self.policy(states)
            dist = torch.distributions.Categorical(probs=probs)
            log_probs_new = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            surrogate = torch.min(surr1, surr2).mean()
            with torch.no_grad():
                ref_probs = self.policy_ref(states)
                ref_dist = torch.distributions.Categorical(probs=ref_probs)
                ref_log_probs = ref_dist.log_prob(actions)
            log_ratio = ref_log_probs - log_probs_new.detach()
            kl = torch.exp(log_ratio) - log_ratio - 1
            kl_mean = torch.relu(kl.mean())
            loss = -surrogate + self.kl_beta * kl_mean - self.entropy_coeff * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += surrogate.item()
            total_kl += kl_mean.item()
            total_entropy += entropy.item()
        return total_loss / self.epochs, total_kl / self.epochs, total_entropy / self.epochs

    def train(self):
        advantages_list = []
        for iter in range(self.num_iterations):
            states, actions, log_probs, rewards, ep_rewards = self.collect_rollouts()
            self.policy.train()
            advantages = self.compute_advantages(rewards)
            
            # Flatten and accumulate advantages
            advantages_flat = torch.cat(advantages).cpu().numpy()  # flatten into one array
            advantages_list.append(np.mean(advantages_flat))
            
            self.policy_ref.load_state_dict(self.policy.state_dict())
            self.policy_ref.eval()
            
            # Perform policy update with the accumulated batch data
            loss, kl, entropy = self.update_policy(states, actions, log_probs, advantages)
            
            # Log losses and other metrics
            self.policy_losses.append(loss)
            self.kl_divergences.append(kl)
            self.entropies.append(entropy)
            
            # Compute the average reward for the current iteration
            avg_reward = np.mean(ep_rewards)
            self.rewards_per_iteration.append(avg_reward)
            
            if avg_reward >= self.threshold:  # change this to 500 for v1, 200 for v2
                print(f"Iter {iter + 1}/{self.num_iterations} | Avg Reward: {avg_reward:.2f}")
                print("Solved!")
                return iter, advantages_list
            print(f"Iter {iter + 1}/{self.num_iterations} | Avg Reward: {avg_reward:.2f}")
            
        self.env.close()
        print("GRPO Training Complete.")
        return iter, advantages_list

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Create GRPO trainer instance
# trainer = GRPO(device=device, env=CartPoleEnvironment(), num_features=4, num_actions=2, group_size=16)

# # Run training loop
# trainer.train()
