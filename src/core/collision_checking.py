import torch
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

# cuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml


tensor_args = TensorDeviceType()

config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))["robot_cfg"]
robot_cfg = RobotConfig.from_dict(config_file, tensor_args)

kin_model = CudaRobotModel(robot_cfg.kinematics)

# compute forward kinematics:
# torch random sampling might give values out of joint limits
q = torch.rand((10, kin_model.get_dof()), **vars(tensor_args))
out = kin_model.get_state(q)

robot_file = "franka.yml"
world_file = "collision_test.yml"

config = RobotWorldConfig.load_from_config(robot_file, world_file, collision_activation_distance=0.01)
empty_world = RobotWorld(config)

def check_for_self_collision(joint_states: torch.tensor) -> bool:
    """..."""


    distance_world, distance_self = empty_world.get_world_self_collision_distance_from_joints(joint_states)

    #if torch.any(distance_self):
    #    print("Self collision.")
    #if torch.any(distance_world):
    #    print("World collision.")

    return distance_self


def check_for_self_collision_test(joint_states: torch.tensor) -> bool:
    """..."""

    world_config = {
        "cuboid": {
            "table": {"dims": [2, 2, 0.2], "pose": [0.4, 0.0, 0.3, 1, 0, 0, 0]},
            "cube_1": {"dims": [0.1, 0.1, 0.2], "pose": [0.4, 0.0, 0.5, 1, 0, 0, 0]},
        },
        "mesh": {
            "scene": {
                "pose": [1.5, 0.080, 1.6, 0.043, -0.471, 0.284, 0.834],
                "file_path": "scene/nvblox/srl_ur10_bins.obj",
            }
        },
    }
    tensor_args = TensorDeviceType()
    config = RobotWorldConfig.load_from_config(robot_file, world_file,
                                               collision_activation_distance=0.0)
    curobo_fn = RobotWorld(config)

    d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(joint_states)
    if torch.any(d_self):
        print("Self collision.")

