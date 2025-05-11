import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    def __init__(self, num_features, hidden_size=100, learning_rate=0.001):
        super(ValueNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x).squeeze(-1)  # shape: [batch_size]

    def get(self, states):
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32)
            return self(states).numpy()

    def update(self, states, discounted_rewards):
        states = torch.tensor(states, dtype=torch.float32)
        targets = torch.tensor(discounted_rewards, dtype=torch.float32)

        predictions = self(states)
        loss = self.loss_fn(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class PolicyModel(nn.Module):
    def __init__(self, num_features, num_actions, layer_1_size=40, layer_2_size=35, layer_3_size=30):
        super(PolicyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, layer_1_size),
            nn.ReLU(),
            nn.Linear(layer_1_size, layer_2_size),
            nn.ReLU(),
            nn.Linear(layer_2_size, layer_3_size),
            nn.ReLU(),
            nn.Linear(layer_3_size, num_actions)
        )

    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=-1)
