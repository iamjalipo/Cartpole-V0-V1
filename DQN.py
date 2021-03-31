import gym
import time

env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(1000):
    env.render()
    time.sleep(0.1)
    action = env.action_space.sample()
    observation, reward , done , info = env.step(action)
    
    if done:
        observation= env.reset
env.close