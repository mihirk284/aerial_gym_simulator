import time
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.registry.task_registry import task_registry
import torch

from aerial_gym.examples.rl_games_example.rl_games_inference import MLP

import time
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    logger.print_example_message()
    start = time.time()
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    plt.style.use("seaborn-v0_8-colorblind")
    rl_task_env = task_registry.make_task(
        "position_setpoint_task_sim2real_end_to_end",
        # "position_setpoint_task_acceleration_sim2real",
        # other params are not set here and default values from the task config file are used
        seed=seed,
        headless=False,
        num_envs=24,
        use_warp=True,
    )
    rl_task_env.reset()
    actions = torch.zeros(
        (
            rl_task_env.sim_env.num_envs,
            rl_task_env.task_config.action_space_dim,
        )
    ).to("cuda:0")
    model = (
        MLP(
            rl_task_env.task_config.observation_space_dim,
            rl_task_env.task_config.action_space_dim,
            # "networks/morphy_policy_for_rigid_airframe.pth"
            # "networks/attitude_policy.pth"
            "/home/mihir/workspaces/aerial_gym_simulator_ws/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runs/gen_ppo_28-13-42-36/nn/gen_ppo.pth"
            # "networks/morphy_policy_for_flexible_airframe_joint_aware.pth",
        )
        .to("cuda:0")
        .eval()
    )
    actions[:] = 0.0
    counter = 0
    action_list = []
    obs_list = []
    euler_angle_list = []
    
    obs_dict = rl_task_env.obs_dict
    with torch.no_grad():
        for i in range(1000):
            if i == 100:
                start = time.time()
            # actions = torch.tanh(actions)
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)
            start_time = time.time()
            actions[:] = model.forward(obs["observations"])
            action_list.append(actions[0].cpu().numpy())
            obs_list.append(obs["observations"][0].cpu().numpy())
            euler_angle_list.append(obs_dict["robot_euler_angles"][0].cpu().numpy())
            end_time = time.time()
        
        state_array = np.array(obs_list)
        action_array = np.array(action_list)
        euler_array = np.array(euler_angle_list)

        # plot various states
        # position
        # 4 subplots
        fig, axs = plt.subplots(4)
        # subplot 1
        axs[0].plot(state_array[:, 0], label="x")
        axs[0].plot(state_array[:, 1], label="y")
        axs[0].plot(state_array[:, 2], label="z")
        axs[0].set_title("Position")
        axs[0].legend()
        # subplot 2
        axs[1].plot(state_array[:, 9], label="vx")
        axs[1].plot(state_array[:, 10], label="vy")
        axs[1].plot(state_array[:, 11], label="vz")
        axs[1].set_title("Velocity")
        axs[1].legend()
        # subplot 3
        axs[2].plot(euler_array[:, 0], label="roll")
        axs[2].plot(euler_array[:, 1], label="pitch")
        axs[2].plot(euler_array[:, 2], label="yaw")
        axs[2].set_title("Euler Angles")
        axs[2].legend()
        # subplot 4
        axs[3].plot(action_array[:, 0], label="u1")
        axs[3].plot(action_array[:, 1], label="u2")
        axs[3].plot(action_array[:, 2], label="u3")
        axs[3].plot(action_array[:, 3], label="u4")
        axs[3].set_title("Action")
        axs[3].legend()
        plt.show()

    # # TO change setpoints:
    # rl_task_env.target_position[:] = torch.tensor([0.0, 0.0, 1.0]).to("cuda:0")

    end = time.time()
