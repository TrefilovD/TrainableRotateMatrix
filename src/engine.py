import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from dataset import get_dataloader
from model import RotateLayer
from utils import visualize_results


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_epochs = args.max_epoch
    frequency_vis = args.frequency_vis
    frequency_log = args.frequency_log

    dataloader = get_dataloader(
        batch_size=4,
        shuffle=True,
        num_samples=100,
        image_size=(81, 81),
        transform=None
    )

    model = RotateLayer(device)
    optimizer = optim.Adam(model.parameters(), lr=0.3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs // 1, eta_min=0.001)
    criterion = nn.L1Loss()
    best_loss = 1e9

    for epoch in range(num_epochs):
        for input_image, target_image in dataloader:
            input_image = input_image.to(device)
            target_image = target_image.to(device)

            optimizer.zero_grad()

            output_image = model(input_image)
            loss = criterion(output_image, target_image)
            loss.backward()

            optimizer.step()

        stop = loss <= 0.0001

        if (epoch + 1) % frequency_log == 0 or epoch + 1 == num_epochs or stop:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, '
                  f'Angle: {model.sigmoida(model.theta).item() * -180:.2f} degrees, lr: {scheduler.get_last_lr()}')

        if (epoch + 1) % frequency_vis == 0 or epoch + 1 == num_epochs or stop:
            visualize_results(input_image, output_image, target_image, epoch + 1, args.save_dir)

        loss = loss.cpu().item()
        if loss < best_loss:
            best_loss == loss
            path_checkpoint = os.path.join(args.checkpoint_path, "best.pth")
            torch.save(model.state_dict(), path_checkpoint)

        if stop:
            break

        # scheduler.step()

    # Визуализируем тестовую
    input_image = dataloader.dataset.load_image_from_file()
    transform = transforms.Compose([
        transforms.Resize(dataloader.dataset.image_size),
        transforms.ToTensor(),
    ])
    input_image = transform(input_image).to(device)
    target_image = dataloader.dataset.create_target_image(input_image)
    input_image = input_image.unsqueeze(0)
    target_image = target_image.unsqueeze(0)
    with torch.no_grad():
        final_output = model(input_image)
    visualize_results(input_image, final_output, target_image, -1, args.save_dir)
    print(f'Final rotation angle: {model.theta.item():.2f} degrees')