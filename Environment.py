import gym
import torch
from typing import Tuple

class CartPoleEnvironment:
    def __init__(self) -> None:
        self.env = gym.make("CartPole-v1")
        self.state_dim: int = self.env.observation_space.shape[0]
        self.action_dim: int = self.env.action_space.n

    def reset(self,render: bool) -> torch.Tensor:
        if(render):
            self.env = gym.make("CartPole-v1", render_mode="human")
        else:
            self.env = gym.make("CartPole-v1")
        state = self.env.reset()
        return self._get_state_tensor(state[0])

    def _get_state_tensor(self, state) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        next_state, reward, done, _, _ = self.env.step(action)
        return self._get_state_tensor(next_state), reward, done
    
    def get_action_space_size(self) -> int:
        return self.action_dim

    def get_state_dimension(self) -> int:
        return self.state_dim

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()

class LunarLanderEnvironment:
    def __init__(self) -> None:
        self.env = gym.make("LunarLander-v2")
        self.state_dim: int = self.env.observation_space.shape[0]
        self.action_dim: int = self.env.action_space.n

    def reset(self,render: bool) -> torch.Tensor:
        if(render):
            self.env = gym.make("LunarLander-v2", render_mode="human")
        else:
            self.env = gym.make("LunarLander-v2")
        state = self.env.reset()
        return self._get_state_tensor(state[0])

    def _get_state_tensor(self, state) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        next_state, reward, done, _, _ = self.env.step(action)
        return self._get_state_tensor(next_state), reward, done
    
    def get_action_space_size(self) -> int:
        return self.action_dim

    def get_state_dimension(self) -> int:
        return self.state_dim

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()