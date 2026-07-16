"""
Shared reward wrapper for CarRacing-v3.

Starts from the default CarRacing reward (-0.1/frame, +1000/N per tile) and
adds optional shaping terms. ALL coefficients default to values that make
this wrapper a near-passthrough; turn terms on deliberately when you're
studying reward shaping (Hargun's thread) so the effect is measurable.

Keep the *baseline* runs using default coefficients (or no wrapper at all)
so "improved reward function" can be cleanly compared against it.
"""

import numpy as np
from gymnasium import RewardWrapper


class CarRacingRewardWrapper(RewardWrapper):

    def __init__(self, env,
                 speed_coef=0.01,
                 smoothness_coef=-0.02,
                 on_track_reward=0.05,
                 movement_penalty=-0.05,
                 position_threshold=0.05,
                 progress_coef=0.1):
        print("using custom wrapper")
        super().__init__(env)
        self.speed_coef = speed_coef
        self.smoothness_coef = smoothness_coef
        self.on_track_reward = on_track_reward
        self.movement_penalty = movement_penalty
        self.position_threshold = position_threshold
        self.progress_coef = progress_coef
        self.last_position = None
        self.last_tile_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_position = self._car_position()
        self.last_tile_count = self._tile_count()
        return obs, info

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        new_pos = self._car_position()
        new_tiles = self._tile_count()
        speed = self._car_speed()
        steering = action[0] if isinstance(
            action, (list, tuple, np.ndarray)) else 0.0

        # discourage standing still
        if self.movement_penalty and self.last_position is not None:
            dx = new_pos[0] - self.last_position[0]
            dy = new_pos[1] - self.last_position[1]
            if (dx * dx + dy * dy) ** 0.5 < self.position_threshold:
                reward += self.movement_penalty

        # reward speed
        reward += speed * self.speed_coef

        # penalize jerky steering
        reward += abs(steering) * self.smoothness_coef

        # small per-step bonus for staying alive
        if not terminated and not truncated:
            reward += self.on_track_reward
        # bonus for new tiles (progress)
        progress = new_tiles - self.last_tile_count

        if progress > 0:
            reward += progress * self.progress_coef

        self.last_position = new_pos
        self.last_tile_count = new_tiles
        return obs, reward, terminated, truncated, info

    # --- helpers reach into the unwrapped Box2D car ---
    def _car_position(self):
        env = self.env.unwrapped
        if hasattr(env, "car") and env.car is not None and hasattr(env.car, "hull"):
            p = env.car.hull.position
            return (p.x, p.y)
        return (0.0, 0.0)

    def _car_speed(self):
        env = self.env.unwrapped
        if hasattr(env, "car") and env.car is not None and hasattr(env.car, "hull"):
            v = env.car.hull.linearVelocity
            return (v.x * v.x + v.y * v.y) ** 0.5
        return 0.0

    def _tile_count(self):
        env = self.env.unwrapped
        return getattr(env, "tile_visited_count", 0)
