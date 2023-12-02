# --- external imports ---
import torch
# --- curobo imports ---
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.types.math import Pose
from curobo.wrap.reacher.ik_solver import IKSolverConfig, IKSolver


def define_franka_robot() -> RobotConfig:
    """Defines the Franka robot.

    Returns:
        RobotConfig:
            The Franka robot.
    """
    tensor_args = TensorDeviceType()
    config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))

    urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

    return RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)


def forward_kinematics(robot: RobotConfig, joint_angles: torch.Tensor) -> torch.Tensor:
    """Computes the forward kinematics for the provided robot.

    Notes:
        Forward kinematics refers to mapping the robot's joint angles to the end effector's pose.

    Args:
        robot: RobotConfig
            Defines the robot's geometry.
        joint_angles: torch.Tensor
            N x (robot's degrees of freedom) tensor. N set of joint angles.

    Returns:
        torch.Tensor:
            N x (robot's degrees of freedom) tensor. N sets of poses. [[x, y, z, w, i, j, k], ...]
    """
    kinematic_model = CudaRobotModel(robot.kinematics)
    states = kinematic_model.get_state(joint_angles)
    poses = torch.cat((states.ee_position, states.ee_quaternion), dim=1)
    return poses


def random_joint_angles(robot: RobotConfig, num_samples: int) -> torch.Tensor:
    """Computes a set of random joint angles for the provided robot.

    Args:
        robot: RobotConfig
            Defines the robot's links and joints.
        num_samples: int
            The number of random joint angle sets to create.

    Returns:
        torch.Tensor
            N x (robot's degrees of freedom) tensor. N sets of joint angles.
    """
    kinematic_model = CudaRobotModel(robot.kinematics)
    return torch.rand((num_samples, kinematic_model.get_dof()), **vars(robot.tensor_args))


def test(robot):

    ik_config = IKSolverConfig.load_from_robot_config(
        robot,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=robot.tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)

    q_sample = ik_solver.sample_configs(5000) # A set of joint angles
    kin_state = ik_solver.fk(q_sample)

    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)  # Poses
    result = ik_solver.solve_batch(goal)
    q_solution = result.solution[result.success] # A set of joint angles


def inverse_kinematics(robot: RobotConfig, poses: torch.Tensor) -> torch.Tensor:
    """Computes the inverse kinematics for the provided robot.

    Notes:
        Inverse kinematics refers to mapping the robot's end effector pose to a set of corresponding joint angles.

    Args:
        robot: RobotConfig
            Defines the robot's geometry.
        poses: torch.Tensor
            N x 7 tensor. [[x, y, z, w, i, j, k], ...]
    Returns:
        torch.Tensor:
            N x (robot's degrees of freedom) tensor. N sets of joint angles.
    """
    ik_config = IKSolverConfig.load_from_robot_config(
        robot,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=robot.tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)

    #goal = Pose.from_batch_list(poses, robot.tensor_args)  #Pose(kin_state.ee_position, kin_state.ee_quaternion)  # Poses

    #position, quaternion = torch.split(poses, 2, dim=1)
    split_poses = torch.split(poses, 1 * [3, 4], dim=1)
    positions = split_poses[0]
    quaternions = split_poses[1]
    goal = Pose(position=positions, quaternion=quaternions)

    #goal = Pose(poses.ee_position.view(len(world_file), 10, 3), poses.ee_quaternion.view(len(world_file), 10, 4))
    #result = ik_solver.solve_batch(goal)


    result = ik_solver.solve_batch_env_goalset(goal)
    q_solution = result.solution[result.success]  # A set of joint angles
    return q_solution


if __name__ == "__main__":
    num_samples = 10
    robot = define_franka_robot()

    for i in range(100):
        original_joint_angles = random_joint_angles(robot, num_samples)
        poses_1 = forward_kinematics(robot, original_joint_angles)

        computed_joint_angles = inverse_kinematics(robot, poses_1)
        poses_2 = forward_kinematics(robot, computed_joint_angles)

        try:
            print(torch.isclose(poses_1, poses_2, atol=0.01))
            print(torch.all(torch.isclose(poses_1, poses_2, atol=0.01)))
        except RuntimeError as e:
            print("Error.")
            continue
        assert torch.all(torch.isclose(poses_1, poses_2, atol=0.01)).item()
