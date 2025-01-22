from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.bem import BEMParams, bem_algorithm
import torch
import numpy as np
from aerial_gym.utils.math import *


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

import time


if __name__ == '__main__':
    with torch.no_grad():
        env = SimBuilder().build_env(
            sim_name="base_sim",
            env_name="empty_env",
            robot_name="bem_lmf2",
            controller_name="lee_position_control",
            args=None,
            num_envs=1,
            device="cuda:0",
            headless=False,
            use_warp=True,
        )

        env.reset()

        state_dict = env.get_obs()

        robot_linvel = state_dict['robot_linvel'][0]
        robot_body_linvel = state_dict['robot_body_linvel'][0]
        robot_body_angvel = state_dict['robot_body_angvel'][0]
        
        roll_rate = robot_body_angvel[0]
        pitch_rate = robot_body_angvel[1]

        params = BEMParams(
            rho= 1.204,      # kg/m^3
            R= 5.1*2.54/2 * 0.01,        # m 2.5 inch prop
            b= 3,            # blades
            c= 0.015,         # m
            cd0= 13.53063,
            cl0= 15.20569,
            theta0= 21.77*np.pi/180,     # rad
            theta1= -11.00*np.pi/180,    # rad
            k_beta= 7.571,     # N*m/rad
            e= 0.01,         # m
            I_b= 0.00122*(0.0635)**2,       # kg*m^2 (blade moment of inertia)
            m_b= 0.00122,       # kg (blade mass)
            # robot_mass = 1.51, # kg
            # gravity = 9.81     # m/s^2
        )

        # get propeller angular rate from the thrust force applied using the robot's force constant
        print(env.robot_manager.robot.control_allocator.motor_model.current_motor_thrust[0])
        print(env.robot_manager.robot.control_allocator.motor_model.motor_thrust_constant[0])
        thrusts = env.robot_manager.robot.control_allocator.motor_model.current_motor_thrust[0]
        consts = env.robot_manager.robot.control_allocator.motor_model.motor_thrust_constant[0]
        rpm = torch.sqrt(thrusts / consts)


        # wind velocity in world frame
        v_wind = torch.zeros((1,3), device='cuda:0')
        v_wind[0, 0] = 0.0

        v_wind_body = quat_rotate(state_dict['robot_orientation'], v_wind)[0]
        v_wind = v_wind_body
        v_hor = torch.norm(v_wind[0:2])



        # calculate the robot horiontal velocity in the rotor frame
        v_hor_vector = robot_linvel[0] - v_wind[0]
        v_hor_scalar = torch.norm(v_hor_vector)
        v_hor = 0*v_hor_scalar.item()

        # calculate the robot vertical velocity in the rotor frame
        v_ver = robot_linvel[2] - v_wind[2]
        v_ver_scalar = torch.norm(v_ver)
        v_ver = 0*v_ver_scalar.item()

        # calculate the roll rate in the rotor frame
        p = roll_rate

        # calculate the pitch rate in the rotor frame
        q = pitch_rate

        st = time.time()

        actual_thrusts = torch.zeros(thrusts.shape, device='cuda:0')

        robot = env.robot_manager.robot

        action = torch.zeros(4, device='cuda:0')

        for timestep in range(5000):
            thrusts = env.robot_manager.robot.control_allocator.motor_model.current_motor_thrust[0]
            consts = env.robot_manager.robot.control_allocator.motor_model.motor_thrust_constant[0]
            rpm = torch.sqrt(thrusts / consts)
            # print("rpm", rpm)
            for i in range(rpm.shape[0]):
                # print("rpm:", rpm[i].item())
                omega = rpm[i].item()
                # calculate the propeller forces and torques
                thrust, torque = bem_algorithm(params, omega, v_hor, v_ver, p, q)
                motor_thrust_z = thrust[2]
                
                actual_thrusts[i] = -motor_thrust_z
                # print(robot.robot_force_tensors.shape)
                robot.robot_force_tensors[0, 5+i, 2] = - thrust[2]
                # robot.robot_torque_tensors[0][i] = - torch.tensor(torque)
                env.step(action)
            # calculate the propeller forces and torques
            # thrust, torque = bem_algorithm(params, omega, v_hor, v_ver, p, q)
            # print("Motor thrust:", thrusts)
            # print("Actual thrust:", actual_thrusts)
            print(thrusts - actual_thrusts)

        print("Time taken:", time.time() - st)






            