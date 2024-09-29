import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


# Create and wrap the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')

# Apply an action space wrapper to the environment
env = JoypadSpace(env, COMPLEX_MOVEMENT)

# Reset the environment
env.reset()
# Take a step in the environmentd
for step in range(1000):
    env.render()
    state,reward,done,info = env.step(env.action_space.sample())
    if done:
        env.reset()
env.close()