import torch
import torch.optim as optim
from typing import Tuple, List
import numpy as np
from Environment import CartPoleEnvironment
from Policy import PolicyModel, ValueNetwork

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

class HAGRPO:
    def __init__(self, 
                 env, 
                 num_features, 
                 num_actions, 
                 gamma=0.994, 
                 lam=1.0,
                 epsilon=0.2,
                 value_network_lr=1e-3,
                 value_network_hidden_size=100,
                 policy_network_lr=1e-2,
                 kl_beta=0.01,
                 entropy_coeff=0.01, 
                 clip_epsilon=0.2, 
                 epochs=10,
                 group_size=16, 
                 max_steps_per_episode=500, 
                 epsilon_std=1e-8,
                 num_iterations=1000, 
                 render_interval=-1, 
                 print_interval=10,
                 policy_hidden_sizes=(40, 35, 30), 
                 device=torch.device("cpu")):

        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
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
        self.device = device

        self.value_net = ValueNetwork(num_features, value_network_hidden_size, learning_rate=value_network_lr)

        self.policy = PolicyModel(num_features, num_actions, *policy_hidden_sizes).to(self.device)
        self.policy_ref = PolicyModel(num_features, num_actions, *policy_hidden_sizes).to(self.device)
        self.policy_ref.load_state_dict(self.policy.state_dict())
        self.policy_ref.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_network_lr)

        self.rewards_per_iteration = []
        self.policy_losses = []
        self.entropies = []
        self.kl_divergences = []

    def get_dist(self, states):
        states = torch.tensor(states, dtype=torch.float32)
        with torch.no_grad():
            return self.policy(states).numpy()

    def collect_rollouts(self):
        states_list, actions_list, log_probs_list, rewards_list = [], [], [], []
        ep_rewards = []
        self.policy.eval()
        for g in range(self.group_size):
            state = self.env.reset(render=(self.render_interval > 0 and g == 0))
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
        
    def discount_rewards(self, rewards):
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_total = 0
        for t in reversed(range(len(rewards))):
            running_total = rewards[t] + self.gamma * running_total
            discounted[t] = running_total
        return discounted

    def calculate_PPO_advantages(self, rewards, values): # rewards: List[float], values: List[float]
        advantages = np.zeros_like(rewards)
        for t in range(len(rewards)):
            ad = 0
            for l in range(0, len(rewards) - t - 1):
                delta = rewards[t + l] + self.gamma * values[t + l + 1] - values[t + l]
                ad += ((self.gamma * self.lam) ** l) * delta
            ad += ((self.gamma * self.lam) ** l) * (rewards[t + l] - values[t + l])
            advantages[t] = ad
        return (advantages - np.mean(advantages)) / np.std(advantages)

    def compute_GRPO_advantages(self, rewards_list): #rewards_list: List[List[float]])
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
    
    def compute_advantages(self, rewards_list, values_list):
        """
        Hybrid advantage: w_t * PPO_adv + (1 - w_t) * GRPO_adv
        Per-timestep weight:
        w_t = (1 - sigma_r_t) / (sigma_v_t^2 + (1 - sigma_r_t)^2)
        """
        G = len(rewards_list)
        T_max = max(len(r) for r in rewards_list)

        # Step 1: pad rewards/values for per-timestep variance
        rewards_pad = np.full((G, T_max), np.nan, dtype=np.float32)
        values_pad = np.full((G, T_max), np.nan, dtype=np.float32)

        for g in range(G):
            T = len(rewards_list[g])
            rewards_pad[g, :T] = rewards_list[g]
            values_pad[g, :T] = values_list[g]

        # Step 2: compute w_t for each timestep
        sigma_r_t = np.nanstd(rewards_pad, axis=0)
        sigma_v_t = np.nanstd(values_pad, axis=0)

        w_t = np.zeros(T_max, dtype=np.float32)
        for t in range(T_max):
            sr = sigma_r_t[t]
            sv = sigma_v_t[t]
            if np.isnan(sr) or np.isnan(sv) or sv == 0:
                w_t[t] = 1.0
            else:
                w_t[t] = (1 - sr) / (sv ** 2 + (1 - sr) ** 2)

        # Step 3: compute PPO advantages per rollout using your function
        ppo_advantages = []
        for rewards, values in zip(rewards_list, values_list):
            adv = self.calculate_PPO_advantages(rewards, values)
            ppo_advantages.append(torch.tensor(adv, device=self.device, dtype=torch.float32))

        # Step 4: compute GRPO advantages using your function
        grpo_advantages = self.compute_GRPO_advantages(rewards_list)

        # Step 5: compute hybrid advantage using per-timestep weights
        hybrid_advantages = []
        for g in range(G):
            T = len(rewards_list[g])
            w_slice = torch.tensor(w_t[:T], device=self.device)
            hybrid = w_slice * ppo_advantages[g] + (1 - w_slice) * grpo_advantages[g]
            hybrid_advantages.append(hybrid)

        return hybrid_advantages

        

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
        for iter in range(self.num_iterations):
            states, actions, log_probs, rewards, ep_rewards = self.collect_rollouts()
            self.policy.train()
            # ==== Add this part ====
            # TODO: calculate the advantages for each output sequence
            values = [self.value_net.get(np.array(states)) for states in states]
            advantages = self.compute_advantages(rewards, values)
            # ==== End this part ====
            self.policy_ref.load_state_dict(self.policy.state_dict())
            self.policy_ref.eval()
            loss, kl, entropy = self.update_policy(states, actions, log_probs, advantages)
            self.policy_losses.append(loss)
            self.kl_divergences.append(kl)
            self.entropies.append(entropy)
            avg_reward = np.mean(ep_rewards)
            self.rewards_per_iteration.append(avg_reward)
            if True:
                print(f"Iter {iter + 1}/{self.num_iterations} | Avg Reward: {avg_reward:.2f} | KL: {kl:.4f} | Entropy: {entropy:.4f}")
                if avg_reward >= 500.0:
                    print("Solved!")
                    return iter
        
        self.env.close()
        print("HAGRPO Training Complete.")
        return iter


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Create GRPO trainer instance
# trainer = HAGRPO(device=device, env=CartPoleEnvironment(), num_features=4, num_actions=2, group_size=16)

# # Run training loop
# trainer.train()
