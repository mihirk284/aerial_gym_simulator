from aerial_gym.robots.base_multirotor import BaseMultirotor



class BEMQuad(BaseMultirotor):
    def __init__(self, robot_config, controller_name, env_config, device):
        super().__init__(
            robot_config=robot_config, controller_name=controller_name, env_config=env_config, device=device
        )
        self.counter = 0


    def call_controller(self):
        """
        Convert the action tensor to the controller inputs. The action tensor is the input and can be parametrized as desired by the user.
        This function serves the purpose of converting the action tensor to the controller inputs.
        """

        self.clip_actions()

        controller_output = self.controller(self.action_tensor)
        self.control_allocation(controller_output, self.output_mode)

        # the assignment of force is not done here. Alternatively, these are written externally using the BEM computed forces
        
        if self.counter < 100:
            self.robot_force_tensors[:] = self.output_forces
            self.robot_torque_tensors[:] = self.output_torques
        else:
            print("Forces are not assigned to the robot")
        self.counter += 1