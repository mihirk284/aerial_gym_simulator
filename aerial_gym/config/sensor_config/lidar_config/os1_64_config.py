from aerial_gym.config.sensor_config.lidar_config.os0_128_config import OS_0_128_Config


class OS_1_64_Config(OS_0_128_Config):
    # keep everything pretty much the same and change the number of vertical rays
    height = 64

    horizontal_fov_deg_min = -180
    horizontal_fov_deg_max = 180
    vertical_fov_deg_min = -22.5
    vertical_fov_deg_max = +22.5

    max_range = 90.0
    min_range = 0.7

    class sensor_noise:
        enable_sensor_noise = False
        pixel_dropout_prob = 0.01
        pixel_std_dev_multiplier = 0.01
