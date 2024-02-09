#马里奥游戏环境test

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# 创建环境
env = gym_super_mario_bros.make('SuperMarioBros-v0')

# 将动作空间转换为简单动作
env = JoypadSpace(env, COMPLEX_MOVEMENT)

# 重置环境
state = env.reset()
for step in range(1000):
    # 渲染环境
    env.render()
    #获取游戏状态
    state, reward, done, info = env.step(env.action_space.sample())
    #判断游戏是否结束
    if done:
        state = env.reset()
# 关闭环境
env.close()