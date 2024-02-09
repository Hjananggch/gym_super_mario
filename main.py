import torch
from utils.log import MetricLogger
from utils.pro_img import * 
from pathlib import Path
from nes_py.wrappers import JoypadSpace
import datetime
import gym_super_mario_bros
from gym.wrappers import FrameStack
from model.agent import Mario



if __name__ == "__main__":
    # 初始化 Super Mario 环境
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    # 限制动作空间为：向右走和向右跳
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env.reset()
    next_state, reward, done, info = env.step(action=0)

    # 应用环境包装器
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    # 实例化 Mario
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
    logger = MetricLogger()

    #train
    e=0
    while True:
        state = env.reset()

        # 开始游戏循环
        while True:

            # 选择动作
            action = mario.act(state)
            env.render()
            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 记忆当前经历
            mario.cache(state, next_state, action, reward, done)

            # 学习更新网络
            q, loss = mario.learn()

            # 记录日志
            logger.log_step(reward, loss, q)

            # 更新状态
            state = next_state

            # 检查游戏是否结束
            if done:
                e += 1
                logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
                if e % 20 == 0:
                    mario.save()
                break