from collections import deque

import numpy as np

from mario_rl.agent import MarioAgent
from mario_rl.config import TrainingConfig
from mario_rl.utils import reset_env, step_env


class Trainer:
    def __init__(self, env, agent: MarioAgent, config: TrainingConfig) -> None:
        self.env = env
        self.agent = agent
        self.config = config
        self.best_moving_average = float("-inf")
        self.reward_window = deque(maxlen=self.config.log_window)

    def train(self) -> None:
        print(f"Training on device: {self.config.device}")
        try:
            for episode in range(1, self.config.episodes + 1):
                reward = self.run_episode()
                self.reward_window.append(reward)
                moving_average = float(np.mean(self.reward_window))

                print(
                    "Episode={episode} Step={step} Epsilon={epsilon:.4f} Reward={reward:.2f} "
                    "MovingAvg({window})={moving_average:.2f}".format(
                        episode=episode,
                        step=self.agent.curr_step,
                        epsilon=self.agent.exploration_rate,
                        reward=reward,
                        window=len(self.reward_window),
                        moving_average=moving_average,
                    )
                )

                if moving_average > self.best_moving_average:
                    self.best_moving_average = moving_average
                    path = self.agent.save_checkpoint(episode)
                    print(f"Saved new best checkpoint: {path}")

                if episode % self.config.checkpoint_period == 0:
                    path = self.agent.save_checkpoint(episode)
                    print(f"Saved periodic checkpoint: {path}")
        finally:
            self.env.close()

    def run_episode(self) -> float:
        state = reset_env(self.env)
        done = False
        episode_reward = 0.0
        step = 0

        while not done:
            if self.config.render:
                self.env.render()

            action = self.agent.act(state)
            next_state, reward, done, _ = step_env(self.env, action)
            self.agent.remember(state, next_state, action, reward, done)
            self.agent.learn()

            state = next_state
            episode_reward += reward
            step += 1

            if self.config.max_steps_per_episode and step >= self.config.max_steps_per_episode:
                break

        return episode_reward
