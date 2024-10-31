import os

import matplotlib.pyplot as plt
import torch


def visualize_results(
        input_image: torch.Tensor,
        output_image: torch.Tensor,
        target_image: torch.Tensor,
        epoch: int,
        path: str
    ) -> None:
    bs = input_image.shape[0]
    fig, axes = plt.subplots(bs, 3, figsize=(15, 5))
    print(bs)
    for i in range(bs):
        if bs > 1:
            ax1, ax2, ax3 = axes[i]
        else:
            ax1, ax2, ax3 = axes
        input_img = input_image[i, ...].detach().cpu()
        output_img = output_image[i, ...].detach().cpu()
        target_img = target_image[i, ...].detach().cpu()

        ax1.imshow(input_img.permute(1, 2, 0).clamp(0, 1))
        ax1.set_title('Input Image')
        ax1.axis('off')

        ax2.imshow(output_img.permute(1, 2, 0).clamp(0, 1))
        ax2.set_title('Output Image')
        ax2.axis('off')

        ax3.imshow(target_img.permute(1, 2, 0).clamp(0, 1))
        ax3.set_title('Target Image')
        ax3.axis('off')

    if epoch == -1:
        plt.suptitle(f'Final')
        save_path = os.path.join(path, "Final.jpg")
    else:
        plt.suptitle(f'Epoch {epoch}')
        save_path = os.path.join(path, f'Epoch_{epoch}.jpg')

    plt.savefig(save_path)
    plt.close()