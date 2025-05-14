import torch
import torch.optim as optim
import numpy as np
import gym
from Policy import ValueNetwork, PolicyModel

class PPO:
    def __init__(self, env, num_features=1, num_actions=1, gamma=0.98, lam=1.0, epsilon=0.2,
                 value_network_lr=1e-3, policy_network_lr=1e-2, value_network_hidden_size=100,
                 policy_network_hidden_size_1=40, policy_network_hidden_size_2=35, policy_network_hidden_size_3=30,
                 batch_size=128):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.num_features = num_features
        self.num_actions = num_actions
        self.gamma = gamma
        self.lam = lam
        self.batch_size = batch_size

        self.policy = PolicyModel(num_features, num_actions,
                                  policy_network_hidden_size_1,
                                  policy_network_hidden_size_2,
                                  policy_network_hidden_size_3)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_network_lr)
        self.epsilon = epsilon

        self.value_net = ValueNetwork(num_features, value_network_hidden_size, learning_rate=value_network_lr)
        self.threshold = 500.0


    def get_dist(self, states):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)  # Move to GPU
        with torch.no_grad():
            return self.policy(states).cpu().numpy()  # Return to CPU after inference for NumPy

    def compute_loss(self, states, actions, old_probs, advantages):
        states = states.to(self.device)  # Move to GPU
        actions = actions.to(self.device)  # Move to GPU
        old_probs = old_probs.to(self.device)  # Move to GPU
        advantages = advantages.to(self.device)  # Move to GPU

        probs = self.policy(states)
        new_probs = torch.sum(probs * actions, dim=1)
        old_probs = torch.sum(old_probs * actions, dim=1)

        ratio = new_probs / (old_probs + 1e-8)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))
        return loss

    def update_policy(self, states, chosen_actions, old_probs, advantages):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)  # Move to GPU
        chosen_actions = torch.tensor(chosen_actions, dtype=torch.float32).to(self.device)  # Move to GPU
        old_probs = torch.tensor(old_probs, dtype=torch.float32).to(self.device)  # Move to GPU
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)  # Move to GPU

        loss = self.compute_loss(states, chosen_actions, old_probs, advantages)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()

    def discount_rewards(self, rewards):
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_total = 0
        for t in reversed(range(len(rewards))):
            running_total = rewards[t] + self.gamma * running_total
            discounted[t] = running_total
        return discounted

    def calculate_advantages(self, rewards, values):
        advantages = np.zeros_like(rewards)
        for t in range(len(rewards)):
            ad = 0
            for l in range(0, len(rewards) - t - 1):
                delta = rewards[t + l] + self.gamma * values[t + l + 1] - values[t + l]
                ad += ((self.gamma * self.lam) ** l) * delta
            ad += ((self.gamma * self.lam) ** l) * (rewards[t + l] - values[t + l])
            advantages[t] = ad
        return (advantages - np.mean(advantages)) / np.std(advantages)

    def run_model(self):
        episode = 1
        running_reward = []
        advantage_list = []

        render = False
        states_batch = []
        actions_batch = []
        rewards_batch = []
        old_probs_batch = []
        advantages_batch = []

        while episode < 1000:
            s0, _ = self.env.reset()
            done = False
            ep_rewards = []
            ep_actions = []
            ep_states = []
            score = 0

            while not done:
                if render:
                    self.env.render()

                s_array = np.asarray(s0, dtype=np.float32)
                s_tensor = torch.from_numpy(s_array).unsqueeze(0).to(self.device)  # Convert to tensor and move to GPU

                probs = self.get_dist(s_tensor)
                action = np.random.choice(self.num_actions, p=probs[0])

                a_binarized = np.zeros(self.num_actions)
                a_binarized[action] = 1

                s1, r, is_terminal, truncated, _ = self.env.step(action)
                done = is_terminal or truncated

                score += r
                ep_states.append(s0)
                ep_actions.append(a_binarized)
                ep_rewards.append(r)
                s0 = s1

            ep_states = np.vstack(ep_states)
            ep_actions = np.vstack(ep_actions)
            ep_rewards = np.array(ep_rewards, dtype=np.float32)

            # Store episode data in batch lists
            states_batch.append(torch.tensor(ep_states, dtype=torch.float32).to(self.device))  # Convert to tensor and move to GPU
            actions_batch.append(torch.tensor(ep_actions, dtype=torch.float32).to(self.device))  # Convert to tensor and move to GPU
            rewards_batch.append(torch.tensor(ep_rewards, dtype=torch.float32).to(self.device))  # Convert to tensor and move to GPU

            # If the batch size is reached, update the model
            if len(states_batch) * len(states_batch[0]) >= self.batch_size:
                # Flatten the batches and move them to GPU
                states_batch = torch.cat(states_batch, dim=0).to(self.device)  # Move batch to GPU
                actions_batch = torch.cat(actions_batch, dim=0).to(self.device)  # Move batch to GPU
                rewards_batch = torch.cat(rewards_batch, dim=0).to(self.device)  # Move batch to GPU

                targets = self.discount_rewards(rewards_batch)
                for i in range(len(states_batch)):
                    state_tensor = states_batch[i].unsqueeze(0)  # shape: [1, state_dim]
                    target_tensor = torch.tensor(targets[i], dtype=torch.float32).unsqueeze(0).to(self.device)  # shape: [1]
                    self.value_net.update(state_tensor, target_tensor)


                values = self.value_net.get(states_batch)
                advantages = self.calculate_advantages(rewards_batch, values)
                advantages_batch.append(advantages.mean(axis=0))

                old_probs = self.get_dist(states_batch)
                old_probs_batch.append(old_probs)

                # Perform the policy update with the batch data
                self.update_policy(states_batch, actions_batch, old_probs, advantages_batch)

                # Reset the batch data
                states_batch = []
                actions_batch = []
                rewards_batch = []
                advantages_batch = []
                old_probs_batch = []

            running_reward.append(score)

            if True:
                print(f"Episode: {episode}, Rewards: {score:.2f}")
                if score >= self.threshold:
                    print("Solved!")
                    render = True
                    return running_reward, episode, advantage_list
            episode += 1
        return running_reward, episode, advantage_list

# Example usage
# agent = PPO(
#     env=gym.make('CartPole-v1'),
#     num_features=4,
#     num_actions=2,
#     gamma=0.98,
#     lam=1
# )
# agent.run_model()
