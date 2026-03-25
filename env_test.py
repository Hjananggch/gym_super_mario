import argparse

from mario_rl.config import EnvConfig
from mario_rl.env import build_env
from mario_rl.utils import reset_env, step_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for the wrapped Super Mario environment.")
    parser.add_argument("--steps", type=int, default=200, help="Number of random steps to execute.")
    parser.add_argument(
        "--action-set",
        default="complex",
        choices=["right_only", "simple", "complex"],
        help="Discrete Mario action set.",
    )
    parser.add_argument("--render", action="store_true", help="Render the environment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = build_env(EnvConfig(action_set=args.action_set, render=args.render))
    state = reset_env(env)
    print(f"Initial observation shape: {state.shape}")

    for step in range(1, args.steps + 1):
        if args.render:
            env.render()
        action = env.action_space.sample()
        state, reward, done, info = step_env(env, action)
        if done:
            state = reset_env(env)
        if step % 50 == 0:
            print(f"Step {step}: reward={reward:.2f}, x_pos={info.get('x_pos')}")

    env.close()


if __name__ == "__main__":
    main()
