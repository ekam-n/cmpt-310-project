import argparse
import os

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder

from common import env_factory as factory

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="./logs/dueling_dqn_100/best_model.zip",
        help="Path to trained model"
    )

    parser.add_argument(
        "--video",
        action="store_true",
        help="Save video instead of rendering"
    )

    parser.add_argument(
        "--video_path",
        default="./videos",
        help="Directory to save video"
    )

    args = parser.parse_args()


    model = DQN.load(args.model)

    env = factory.make_vec(
        n_envs=1,
        render_mode="rgb_array" if args.video else "human"
    )


    if args.video:
        os.makedirs(args.video_path, exist_ok=True)

        env = VecVideoRecorder(
            env,
            args.video_path,
            record_video_trigger=lambda x: x == 0,
            video_length=1000,
            name_prefix="race"
        )


    obs = env.reset()

    total_reward = 0
    step = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]

        if not args.video:
            env.render()

        if done[0]:
            break
    print(f"Episode finished with total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()