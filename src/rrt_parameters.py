import torch
from utilities import random_joint_angles


class RRTParameters:
    """
    ...
    """

    def __init__(self, starting_state: torch.tensor, goal_state: torch.tensor):
        """..."""

        self.starting_state = starting_state
        self.goal_state = goal_state

        self.batch_size = 100
        self.step_size = 0.25
        self.goal_distance_threshold = 1.0
        self.max_num_iterations = 1000
