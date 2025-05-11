import torch
import torch.optim as optim
from typing import Tuple, List
import numpy as np
from Environment import CartPoleEnvironment
from Policy import PolicyModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class GRPO:
    def __init__(self, device: torch.device = torch.device("cpu")):
        # --- GRPO Hyperparameters ---
        self.gamma = 0.994
        self.kl_beta = 0.01
        self.actor_lr = 3e-4
        self.epochs = 10
        self.clip_epsilon = 0.2
        self.entropy_coeff = 0.01
        self.num_iterations = 1000
        self.group_size = 16
        self.max_steps_per_episode = 500
        self.epsilon_std = 1e-8
        self.print_interval = 10
        self.render_interval = -1
        self.device = device

        # --- Environment and Actor Initialization ---
        self.env = CartPoleEnvironment()
        self.n_actions = self.env.get_action_space_size()
        self.n_observations = self.env.get_state_dimension()

        self.actor = PolicyModel(self.n_observations, self.n_actions).to(self.device)
        self.actor_ref = PolicyModel(self.n_observations, self.n_actions).to(self.device)
        self.actor_ref.load_state_dict(self.actor.state_dict())  # Make it identical initially
        self.actor_ref.eval()

        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # --- Logging Containers ---
        self.rewards_per_iteration: List[float] = []
        self.avg_episode_lengths: List[float] = []
        self.policy_losses: List[float] = []
        self.entropies: List[float] = []
        self.kl_divergences: List[float] = []

    def update_grpo(
        self,
        group_states_list: List[torch.Tensor],
        group_actions_list: List[torch.Tensor],
        group_log_probs_old_list: List[torch.Tensor],
        group_advantages_list: List[torch.Tensor]
    ) -> Tuple[float, float, float]:
        """
        Performs the GRPO update for multiple epochs over the collected group data.
        Returns average policy objective, KL divergence, and entropy over update epochs.
        """

        total_policy_objective = 0.0
        total_kl_div = 0.0
        total_entropy = 0.0

        if not group_states_list or not group_actions_list or not group_log_probs_old_list or not group_advantages_list:
            print("Warning: Empty lists passed to update_grpo. Skipping update.")
            return 0.0, 0.0, 0.0

        try:
            states = torch.cat(group_states_list, dim=0).to(self.device)
            actions = torch.cat(group_actions_list, dim=0).to(self.device)
            log_probs_old = torch.cat(group_log_probs_old_list, dim=0).to(self.device)
            advantages = torch.cat(group_advantages_list, dim=0).to(self.device)
        except RuntimeError as e:
            print(f"Error during concatenation in update_grpo: {e}")
            for i in range(len(group_states_list)):
                print(f"  Rollout {i}: States={group_states_list[i].shape}, Actions={group_actions_list[i].shape}, LogProbs={group_log_probs_old_list[i].shape}, Advs={group_advantages_list[i].shape}")
            raise e

        advantages = advantages.detach()
        log_probs_old = log_probs_old.detach()
        self.actor_ref.eval()

        for epoch in range(self.epochs):
            action_probs = self.actor(states)
            policy_dist = torch.distributions.Categorical(probs=action_probs)
            log_probs_new = policy_dist.log_prob(actions)
            entropy = policy_dist.entropy().mean()

            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            clipped_surrogate_objective = torch.min(surr1, surr2).mean()

            with torch.no_grad():
                probs_ref = self.actor_ref(states)
                policy_dist_ref = torch.distributions.Categorical(probs=probs_ref)
                log_probs_ref = policy_dist_ref.log_prob(actions)

            log_ratio_ref_curr = log_probs_ref - log_probs_new.detach()
            kl_div_estimate = torch.exp(log_ratio_ref_curr) - log_ratio_ref_curr - 1
            kl_div_estimate_mean = torch.relu(kl_div_estimate.mean())

            policy_loss = -clipped_surrogate_objective + self.kl_beta * kl_div_estimate_mean - self.entropy_coeff * entropy

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            total_policy_objective += clipped_surrogate_objective.item()
            total_kl_div += kl_div_estimate_mean.item()
            total_entropy += entropy.item()

        avg_policy_objective = total_policy_objective / self.epochs
        avg_kl_div = total_kl_div / self.epochs
        avg_entropy = total_entropy / self.epochs

        return avg_policy_objective, avg_kl_div, avg_entropy

    def train(self):
        for iteration in range(self.num_iterations):
            group_states_list = []
            group_actions_list = []
            group_log_probs_old_list = []
            group_rewards_list = []

            episode_rewards_in_iter = []
            episode_lengths_in_iter = []

            self.actor.eval()
            for rollout_idx in range(self.group_size):
                rollout_states = []
                rollout_actions = []
                rollout_log_probs = []
                rollout_rewards = []

                render_flag = (self.render_interval != -1 and rollout_idx == 0 and iteration % self.render_interval == 0)
                state = self.env.reset(render=render_flag)

                episode_reward = 0.0
                episode_steps = 0
                done = False

                for t in range(self.max_steps_per_episode):
                    with torch.no_grad():
                        if not isinstance(state, torch.Tensor):
                            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                        else:
                            state_tensor = state.to(self.device)

                        policy_probs = self.actor(state_tensor)
                        policy_dist = torch.distributions.Categorical(probs=policy_probs)
                        action_tensor = policy_dist.sample()
                        log_prob = policy_dist.log_prob(action_tensor)

                    next_state, reward, done = self.env.step(action_tensor.item())

                    rollout_states.append(state_tensor)
                    rollout_actions.append(action_tensor)
                    rollout_log_probs.append(log_prob)
                    rollout_rewards.append(reward)

                    state = next_state
                    episode_reward += reward
                    episode_steps += 1

                    if done:
                        break

                if rollout_states:
                    group_states_list.append(torch.stack(rollout_states))
                    action_dim = rollout_actions[0].dim()
                    log_prob_dim = rollout_log_probs[0].dim()
                    group_actions_list.append(
                        torch.stack(rollout_actions).squeeze() if action_dim > 0 else torch.tensor([a.item() for a in rollout_actions], device=self.device)
                    )
                    group_log_probs_old_list.append(
                        torch.stack(rollout_log_probs).squeeze() if log_prob_dim > 0 else torch.tensor([lp.item() for lp in rollout_log_probs], device=self.device)
                    )
                    group_rewards_list.append(rollout_rewards)
                else:
                    group_states_list.append(torch.empty((0, self.n_observations), device=self.device))
                    group_actions_list.append(torch.empty((0,), dtype=torch.long, device=self.device))
                    group_log_probs_old_list.append(torch.empty((0,), device=self.device))
                    group_rewards_list.append([])

                episode_rewards_in_iter.append(episode_reward)
                episode_lengths_in_iter.append(episode_steps)

            self.actor.train()

            # --- Advantage Computation --- Just change this part in HAGRPO.py
            group_advantages_list = []
            temp_raw_advantages_tensors = []
            all_raw_advantages_in_group = []

            for rollout_rewards in group_rewards_list:
                rollout_len = len(rollout_rewards)
                raw_advantages = torch.zeros(rollout_len, dtype=torch.float32, device=self.device)

                discounted_return = 0.0
                for t in reversed(range(rollout_len)):
                    discounted_return = rollout_rewards[t] + self.gamma * discounted_return
                    raw_advantages[t] = discounted_return

                temp_raw_advantages_tensors.append(raw_advantages)
                all_raw_advantages_in_group.extend(raw_advantages.cpu().numpy())

            if len(all_raw_advantages_in_group) > 1:
                group_mean = np.mean(all_raw_advantages_in_group)
                group_std = np.std(all_raw_advantages_in_group)
            elif len(all_raw_advantages_in_group) == 1:
                group_mean = all_raw_advantages_in_group[0]
                group_std = 0.0
            else:
                group_mean = 0.0
                group_std = 0.0
                if iteration == 0:
                    print("Warning: No advantages calculated in group (all rollouts empty?).")

            for raw_advantages in temp_raw_advantages_tensors:
                if raw_advantages.nelement() > 0:
                    normalized_adv = (raw_advantages - group_mean) / (group_std + self.epsilon_std)
                else:
                    normalized_adv = raw_advantages
                group_advantages_list.append(normalized_adv)

            self.actor_ref.load_state_dict(self.actor.state_dict())
            self.actor_ref.eval()

            avg_policy_obj, avg_kl, avg_entropy = self.update_grpo(
                group_states_list,
                group_actions_list,
                group_log_probs_old_list,
                group_advantages_list
            )

            self.policy_losses.append(avg_policy_obj)
            self.entropies.append(avg_entropy)
            self.kl_divergences.append(avg_kl)

            avg_reward = np.mean(episode_rewards_in_iter) if episode_rewards_in_iter else np.nan
            avg_len = np.mean(episode_lengths_in_iter) if episode_lengths_in_iter else np.nan
            self.rewards_per_iteration.append(avg_reward)
            self.avg_episode_lengths.append(avg_len)

            if (iteration + 1) % self.print_interval == 0 or iteration == self.num_iterations - 1:
                print(f"Iter {iteration + 1}/{self.num_iterations} | Avg Reward (Group): {avg_reward:.2f}")

            if avg_reward >= 500.0:
                print("Solved!")
                break

        print("\nGRPO Training Loop Finished.")
        self.env.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create GRPO trainer instance
trainer = GRPO(device=device)

# Run training loop
trainer.train()