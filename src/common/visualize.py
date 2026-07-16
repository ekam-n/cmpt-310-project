from stable_baselines3 import DQN
from common import env_factory as factory

model = DQN.load("./logs/double_dqn/best_model.zip")
env = factory.make_vec(n_envs = 1, render_mode="human")

# Run the agent
obs = env.reset()
total_reward = 0


while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]
    env.render()
    if done[0]:
        break

print(f"Episode finished with total reward: {total_reward:.2f}")
env.close()