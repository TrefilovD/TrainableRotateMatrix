import torch
import torch.nn as nn

class RotateLayer(nn.Module):
    def __init__(self, device):
        super(RotateLayer, self).__init__()

        self.theta = nn.Parameter(torch.tensor(0.))
        self.sigmoida = nn.Sigmoid()
        self.device = device
        self.to(device)

    def get_rotation_matrix(self) -> torch.Tensor:
        theta_rad = torch.deg2rad(-180 * self.sigmoida(self.theta))

        rotation_matrix = torch.stack([
            torch.stack([torch.cos(theta_rad), -torch.sin(theta_rad)]),
            torch.stack([torch.sin(theta_rad), torch.cos(theta_rad)])
        ])

        return rotation_matrix

    def get_grid(self, width: int, height: int, batch_size: int) -> torch.Tensor:
        """
        Создаем сетку координат
        """
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1, 1, width, device=self.device),
            torch.linspace(-1, 1, height, device=self.device),
            indexing='xy'
        )
        grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0)
        grid = grid.repeat(batch_size, 1, 1, 1).reshape(batch_size, -1, 2)

        return grid

    def forward(self, x):
        batch_size, _, height, width = x.shape

        rotation_matrix = self.get_rotation_matrix()
        rotation_matrix = rotation_matrix.expand(batch_size, 2, 2)

        grid = self.get_grid(width, height, batch_size)

        # Поворот сетки
        grid = torch.bmm(grid, rotation_matrix)
        grid = grid.reshape(batch_size, height, width, 2)

        # Поворот изображения
        return torch.nn.functional.grid_sample(x, grid, align_corners=True)