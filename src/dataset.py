from typing import List, Optional, Tuple

import torch
import torchvision.transforms as transforms

from PIL import Image


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_samples: int = 10,
        image_size: Tuple[int, int] = (128, 128),
        transform: Optional[transforms.Compose] = None
    ) -> None:
        super().__init__()

        self.num_samples = num_samples
        self.image_size = image_size
        if transform:
            self.transform = transform
        else:
            self.transform = self.default_transform(image_size)

        self.samples = self.generate_samples()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_image = self.samples[index]
        input_image = self.transform(input_image)
        target_image = self.create_target_image(input_image)
        return input_image, target_image

    def generate_samples(self) -> List[torch.Tensor]:
        samples = []
        for _ in range(self.num_samples):
            samples.append(self.load_image_from_file())
        return samples

    def generate_random_input(self) -> torch.Tensor:
        return torch.rand((self.image_size))[None, ...].repeat(3, 1, 1)

    def load_image_from_file(
            self,
            image_path: str = "./src/data/img.jpg"
        ):
        image = Image.open(image_path).convert('RGB')
        return image

    @staticmethod
    def create_target_image(input_image) -> torch.Tensor:
        with torch.no_grad():
            return torch.rot90(input_image.clone(), k=1, dims=[1, 2])

    @staticmethod
    def default_transform(image_size) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomRotation((-90, 90)),
            transforms.ToTensor(),
        ])


def get_dataloader(
        batch_size: int = 4,
        shuffle: bool = False,

        num_samples: int = 10,
        image_size: Tuple[int, int] = (128, 128),
        transform: Optional[transforms.Compose] = None
    ) -> torch.utils.data.DataLoader:
    dataset = SimpleDataset(
        num_samples=num_samples,
        image_size=image_size,
        transform=transform
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )