from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class EnvConfig:
    env_id: str = "SuperMarioBros-1-1-v0"
    action_set: str = "simple"
    skip_frames: int = 4
    resize_shape: int = 84
    stack_frames: int = 4
    seed: int = 42
    render: bool = False


@dataclass(slots=True)
class AgentConfig:
    batch_size: int = 32
    gamma: float = 0.78
    learning_rate: float = 1e-4
    replay_buffer_size: int = 10_000
    exploration_rate: float = 0.75
    exploration_rate_decay: float = 0.999998
    exploration_rate_min: float = 0.01
    sync_steps: int = 10


@dataclass(slots=True)
class TrainingConfig:
    episodes: int = 1000
    checkpoint_period: int = 10
    checkpoint_path: str | None = None
    save_dir: Path = Path("weights")
    device: str = "cpu"
    render: bool = False
    max_steps_per_episode: int | None = None
    log_window: int = 10
