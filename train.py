import argparse
from pathlib import Path

from mario_rl.agent import MarioAgent
from mario_rl.config import AgentConfig, EnvConfig, TrainingConfig
from mario_rl.env import build_env
from mario_rl.trainer import Trainer
from mario_rl.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDQN agent for Super Mario Bros.")
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0", help="Gym environment id.")
    parser.add_argument(
        "--action-set",
        default="simple",
        choices=["right_only", "simple", "complex"],
        help="Discrete Mario action set.",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--checkpoint", help="Optional checkpoint path to resume from.")
    parser.add_argument("--save-dir", default="weights", help="Directory used to save checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--render", action="store_true", help="Render the game during training.")
    parser.add_argument("--device", help="Torch device, for example cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_config = EnvConfig(
        env_id=args.env_id,
        action_set=args.action_set,
        seed=args.seed,
        render=args.render,
    )
    agent_config = AgentConfig()
    training_config = TrainingConfig(
        episodes=args.episodes,
        checkpoint_path=args.checkpoint,
        save_dir=Path(args.save_dir),
        device=resolve_device(args.device),
        render=args.render,
    )

    env = build_env(env_config)
    agent = MarioAgent(
        action_dim=env.action_space.n,
        config=agent_config,
        device=training_config.device,
        save_dir=training_config.save_dir,
        input_channels=env.observation_space.shape[0],
    )
    if training_config.checkpoint_path:
        agent.load(training_config.checkpoint_path)

    trainer = Trainer(env=env, agent=agent, config=training_config)
    trainer.train()


if __name__ == "__main__":
    main()
