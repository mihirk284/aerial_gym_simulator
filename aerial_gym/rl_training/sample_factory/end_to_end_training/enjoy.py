import os
import yaml
import matplotlib.pyplot as plt
import gym
import numpy as np

from aerial_gym.rl_training.sample_factory.end_to_end_training.helper import parse_aerialgym_cfg

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym import AERIAL_GYM_DIRECTORY

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.helpers import get_args
from aerial_gym.registry.task_registry import task_registry

from aerial_gym.rl_training.sample_factory.end_to_end_training.enjoy_ros import NN_Inference_ROS
from aerial_gym.rl_training.sample_factory.end_to_end_training.deployment.convert_model import convert_model_to_script_model

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_euler_angles, quaternion_to_matrix, matrix_to_rotation_6d, euler_angles_to_matrix

from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.registry.env_registry import env_config_registry

logger = CustomLogger(__name__)

import torch
import torch.nn as nn

from tqdm import tqdm as tqdm

import scipy.fftpack as fft_pack
from scipy.signal import butter, lfilter, freqz
    
def test_policy_script_export():
    
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    
    cfg = parse_aerialgym_cfg(evaluation=True)
    # cfg.train_dir = "./train_dir"
    
    env_cfg = task_registry.get_task_config("position_setpoint_task_rpm")
    env = task_registry.make_task("position_setpoint_task_rpm", num_envs=1)
    obs = env.reset()[0]
    
    pos_list = []
    pos_err_list = []
    vel_list = []
    ori_list = []
    ang_vel_list = []
    
    action_motor_command_list = []
    
    goals = torch.tensor([[-.7,-.7,0.],[.7,-.7,0.],[.7,.7,0.],[-.7,.7,0.]] ).to(torch.float).to("cuda:0")

    nn_model = NN_Inference_ROS(cfg, env_cfg)
        
    n_crashes = 0
    total_runs = 0
    
    dt = 0.01
    reset_time = 10
    n_steps = int(reset_time*4/dt)
    reset_counter = 1
    
    for i in range(n_steps):
        
        obs["observations"][:,:3] = goals[reset_counter-1] + obs["observations"][:,:3]
        
        actions_motor_commands = nn_model.get_action(obs)
        actions_motor_commands = env.task_config.action_transformation_function(actions_motor_commands)
            
        obs, _, crashes, successes, _ = env.step(actions=actions_motor_commands)
        
        n_crashes += torch.sum(crashes)
        total_runs += torch.sum(crashes) + torch.sum(successes)
    
        pos_list.append((-obs["observations"].squeeze().cpu()[0:3].detach().numpy()).tolist())
        pos_err_list.append((goals[reset_counter-1].cpu() + obs["observations"].squeeze().cpu()[0:3]).detach().numpy().tolist())
        
        ori_6d = obs["observations"].squeeze().cpu()[3:9].detach()
        ori_matrix = rotation_6d_to_matrix(ori_6d)
        ori_euler = matrix_to_euler_angles(ori_matrix, "XYZ")
        ori_list.append(ori_euler.detach().numpy().tolist())

        vel_list.append(obs["observations"].squeeze().cpu()[9:12].detach().numpy().tolist())
        
        ang_vel_list.append(obs["observations"].squeeze().cpu()[12:15].detach().numpy().tolist())

        action_motor_command_list.append(actions_motor_commands.detach().squeeze().cpu().numpy().tolist())
        
        if i*dt > reset_time*reset_counter:
            print("switching goal")
            reset_counter += 1
            if reset_counter == goals.shape[0]+1:
                break        
    
    print("crash rate: ", n_crashes/total_runs)
    plot_results(pos_list, pos_err_list, vel_list, ori_list, 
                 ang_vel_list, action_motor_command_list, goals.cpu().numpy().tolist(), dt)
    
def plot_results(pos_list, pos_err_list, vel_list, ori_list, 
                 ang_vel_list, action_motor_command_list, goals, dt):
    
    n_steps = len(pos_list)
    # Plot the data
    fig, axs = plt.subplots(3, 2)
    axs[0,0].plot(np.linspace(0,n_steps,n_steps)*dt, np.array(pos_err_list))
    axs[0,0].legend(["x", "y", "z"])
    axs[0,0].set_title("Position")
    axs[0,0].set_xlabel("Time")
    axs[0,0].set_ylabel("Position")
    
    axs[0,1].plot(np.linspace(0,n_steps,n_steps)*dt,np.array(vel_list))
    axs[0,1].legend(["x", "y", "z"])
    axs[0,1].set_title("Velocity")
    axs[0,1].set_xlabel("Time")
    axs[0,1].set_ylabel("Velocity")
    
    axs[1,0].plot(np.linspace(0,n_steps,n_steps)*dt,np.array(ori_list))
    axs[1,0].legend(["x", "y", "z"])
    axs[1,0].set_title("Angles")
    axs[1,0].set_xlabel("Time")
    axs[1,0].set_ylabel("Angles")
    
    axs[1,1].plot(np.linspace(0,n_steps,n_steps)*dt,np.array(ang_vel_list))
    axs[1,1].set_title("Angular Velocity")
    axs[1,1].set_xlabel("Time")
    axs[1,1].set_ylabel("Angular Velocity")
    axs[1,1].legend(["x", "y", "z"])
    
    axs[2,0].plot(np.linspace(0,n_steps,n_steps)*dt,np.array(action_motor_command_list))
    axs[2,0].set_title("Actions motor commands")
    axs[2,0].set_xlabel("Time")
    axs[2,0].set_ylabel("Action")
    axs[2,0].legend(["u1", "u2", "u3", "u4"])
    
    
    yf_u1 = fft_pack.fft(np.array(action_motor_command_list)[:,0])
    yf_u2 = fft_pack.fft(np.array(action_motor_command_list)[:,1])
    yf_u3 = fft_pack.fft(np.array(action_motor_command_list)[:,2])
    yf_u4 = fft_pack.fft(np.array(action_motor_command_list)[:,3])
    
    yf_ang_vel_x = fft_pack.fft(np.array(ang_vel_list)[:,0])
    yf_ang_vel_y = fft_pack.fft(np.array(ang_vel_list)[:,1])
    yf_ang_vel_z = fft_pack.fft(np.array(ang_vel_list)[:,2])

    xf = np.linspace(0.0, 1.0/(2.0*dt), n_steps//2)
    
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(xf, 2.0/n_steps * np.abs(yf_ang_vel_x[0:n_steps//2]))
    ax[0].set_title("Angular Velocity x")
    ax[0].set_xlabel("Frequency")
    ax[0].set_ylabel("Amplitude")
    ax[1].plot(xf, 2.0/n_steps * np.abs(yf_ang_vel_y[0:n_steps//2]))
    ax[1].set_title("Angular Velocity y")
    ax[1].set_xlabel("Frequency")
    ax[2].plot(xf, 2.0/n_steps * np.abs(yf_ang_vel_z[0:n_steps//2]))
    ax[2].set_title("Angular Velocity z")
    ax[2].set_xlabel("Frequency")
    
    fig, ax = plt.subplots(2, 2)
    ax[0,0].plot(xf[1:], 2.0/n_steps * np.abs(yf_u1[1:n_steps//2]))
    ax[0,0].set_title("u1")
    ax[0,0].set_xlabel("Frequency")
    ax[0,0].set_ylabel("Amplitude")
    ax[0,1].plot(xf[1:], 2.0/n_steps * np.abs(yf_u2[1:n_steps//2]))
    ax[0,1].set_title("u2")
    ax[0,1].set_xlabel("Frequency")
    ax[0,1].set_ylabel("Amplitude")
    ax[1,0].plot(xf[1:], 2.0/n_steps * np.abs(yf_u3[1:n_steps//2]))
    ax[1,0].set_title("u3")
    ax[1,0].set_xlabel("Frequency")
    ax[1,0].set_ylabel("Amplitude")
    ax[1,1].plot(xf[1:], 2.0/n_steps * np.abs(yf_u4[1:n_steps//2]))
    ax[1,1].set_title("u4")
    ax[1,1].set_xlabel("Frequency")
    ax[1,1].set_ylabel("Amplitude")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(np.array(pos_list)[:,0], np.array(pos_list)[:,1], np.array(pos_list)[:,2])
    ax.scatter(np.array(goals)[:,0], np.array(goals)[:,1], np.array(goals)[:,2], c='r', marker='o')
    plt.show()

    
if __name__ == "__main__":
    
    test_policy_script_export()