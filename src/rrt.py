import torch
from curobo.types.robot import RobotConfig
from graph import Graph
from rrt_parameters import RRTParameters
from utilities import random_joint_angles

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RRT:
    """
    ...
    """

    def __init__(self, robot: RobotConfig, parameters: RRTParameters):
        """..."""

        self.robot = robot
        self.parameters = parameters
        self.graph = Graph()
        self.graph.vertices = parameters.starting_state

    def plan(self):
        """..."""


        for i in range(self.parameters.max_num_iterations):

            vertices = self.graph.vertices

            q_randoms = random_joint_angles(self.robot, self.parameters.batch_size)
            q_randoms.to(device)

            distances = torch.cdist(q_randoms, vertices)
            nearest_neighbor_indices = torch.argmin(distances, dim=1)
            nearest_neighbors = vertices[nearest_neighbor_indices]
            directions = torch.sub(q_randoms, nearest_neighbors)
            normalized_directions = torch.nn.functional.normalize(directions)

            q_news = torch.add(nearest_neighbors, normalized_directions * self.parameters.step_size)

            distances_to_goal = torch.cdist(q_news, self.parameters.goal_state)
            if torch.any(distances_to_goal < self.parameters.goal_distance_threshold):

                print("Found a solution!")

                q_final_indices = (distances_to_goal < self.parameters.goal_distance_threshold).flatten().nonzero().flatten().tolist()
                q_final_index = q_final_indices[0]
                q_final = q_news[q_final_index]

                self.graph.update(nearest_neighbors, nearest_neighbor_indices, q_news)
                return self.graph.extract_path(self.parameters.starting_state[0], q_final)

            self.graph.update(nearest_neighbors, nearest_neighbor_indices, q_news)




from utilities import random_joint_angles, define_franka_robot

robot = define_franka_robot()
starting_state = random_joint_angles(robot, 1)
goal_state = random_joint_angles(robot, 1)
parameters = RRTParameters(starting_state, goal_state)

rrt = RRT(robot, parameters)
path = rrt.plan()
print(path)


