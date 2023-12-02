import torch

device = torch.device("cpu")


class Graph:
    """
    ...
    """

    def __init__(self):
        """..."""

        self.vertices = None
        self.edges = {}

    def update(self, nearest_neighbors: torch.tensor, nearest_neighbor_indices: torch.tensor, q_news: torch.tensor):
        """..."""
        # 1. Add vertices.
        prior_size = len(self.vertices)
        self.add_vertices(q_news)
        new_size = len(self.vertices)

        # 2. Add edges.
        new_indices = torch.tensor([i for i in range(prior_size, new_size)])
        self.add_edges(nearest_neighbor_indices, new_indices)

    def add_vertices(self, new_vertices: torch.tensor):
        """..."""
        self.vertices = torch.cat((self.vertices, new_vertices), 0)

    def add_edges(self, from_indices: torch.tensor, to_indices: torch.tensor):
        """..."""

        from_indices = from_indices.to(device).tolist()
        to_indices = to_indices.to(device).tolist()
        for from_index, to_index in zip(from_indices, to_indices):
            if to_index in self.edges:
                raise RuntimeError("Parent already exists.")
            else:
                self.edges[to_index] = from_index

    def get_index(self, target: torch.tensor):
        """..."""

        target = target.reshape([1, len(target)])
        distances = torch.cdist(torch.tensor(target), self.vertices)
        return torch.argmin(distances, dim=1).tolist()[0]

    def extract_path_indices(self, q_start: torch.tensor, q_final: torch.tensor):
        """..."""
        q_start_index = self.get_index(q_start)
        q_final_index = self.get_index(q_final)

        current_index = q_final_index
        path_indices = [q_final_index]

        while q_start_index not in path_indices:
            prior_index = self.edges[current_index]
            path_indices.append(prior_index)
            current_index = prior_index

        return list(reversed(path_indices))

    def extract_path(self, q_start: torch.tensor, q_final: torch.tensor):
        """..."""

        path_indices = self.extract_path_indices(q_start, q_final)
        return self.vertices[path_indices]

