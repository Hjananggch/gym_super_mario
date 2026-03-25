import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mario_rl.config import AgentConfig
from mario_rl.model import DoubleDQNNetwork
from mario_rl.utils import ensure_dir


class MarioAgent:
    def __init__(
        self,
        action_dim: int,
        config: AgentConfig,
        device: str,
        save_dir: str | Path,
        input_channels: int = 4,
    ) -> None:
        self.action_dim = action_dim
        self.config = config
        self.device = device
        self.save_dir = ensure_dir(save_dir)

        self.net = DoubleDQNNetwork(input_channels=input_channels, output_dim=action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.online.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = deque(maxlen=self.config.replay_buffer_size)
        self.exploration_rate = self.config.exploration_rate
        self.curr_step = 0

    def act(self, state) -> int:
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = self._state_tensor(state).unsqueeze(0).to(self.device)
                action_values = self.net(state_tensor, branch="online")
                action = torch.argmax(action_values, dim=1).item()

        self.exploration_rate *= self.config.exploration_rate_decay
        self.exploration_rate = max(self.config.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action

    def remember(self, state, next_state, action: int, reward: float, done: bool) -> None:
        self.memory.append(
            (
                self._state_tensor(state),
                self._state_tensor(next_state),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(done, dtype=torch.bool),
            )
        )

    def learn(self) -> float | None:
        if self.curr_step % self.config.sync_steps == 0:
            self.sync_target()

        if len(self.memory) < self.config.batch_size:
            return None

        state, next_state, action, reward, done = self.recall()
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        q_values = self.net(state, branch="online").gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_action = self.net(next_state, branch="online").argmax(dim=1, keepdim=True)
            next_q_values = self.net(next_state, branch="target").gather(1, best_action).squeeze(1)
            q_target = reward + (1 - done.float()) * self.config.gamma * next_q_values

        loss = self.loss_fn(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def recall(self):
        batch = random.sample(self.memory, self.config.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action, reward, done

    def sync_target(self) -> None:
        self.net.target.load_state_dict(self.net.online.state_dict())

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.net.load_state_dict(checkpoint["model"])
        self.exploration_rate = checkpoint.get("exploration_rate", self.exploration_rate)

    def save_checkpoint(self, episode: int) -> Path:
        path = self.save_dir / f"checkpoint_{episode}.pth"
        torch.save(
            {
                "model": self.net.state_dict(),
                "exploration_rate": self.exploration_rate,
            },
            path,
        )
        return path

    @staticmethod
    def _state_tensor(state) -> torch.Tensor:
        return torch.as_tensor(np.asarray(state), dtype=torch.uint8)
