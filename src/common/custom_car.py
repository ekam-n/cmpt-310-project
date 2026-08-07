from gymnasium.envs.box2d.car_racing import CarRacing

class CustomCarRacing(CarRacing):
    def __init__(self, car_color=(0.0, 0.0, 1.0), **kwargs):
        super().__init__(**kwargs)
        self.car_color = car_color

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        
        self.car.hull.color = self.car_color

        return obs, info