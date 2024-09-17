import pydicom
import tifffile
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np


def load_image(image_path):
    if image_path.endswith(".dcm"):
        image = (
            torch.tensor(pydicom.dcmread(image_path).pixel_array)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
    elif image_path.endswith(".tif"):
        image = torch.tensor(tifffile.imread(image_path)).permute(2, 0, 1).unsqueeze(0)
    elif image_path.endswith(".jpg") or image_path.endswith(".png"):
        image = Image.open(image_path).convert("RGB")
        image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError("Unsupported image format")

    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    elif image.dtype == torch.uint16:
        image = image.float() / 65535.0
    else:
        image = image.float()

    return image


def pad_image(image, patch_size):
    # Center pad image to fit patch size
    h_remainder = image.shape[2] % patch_size[0]
    w_remainder = image.shape[3] % patch_size[1]
    if h_remainder != 0:
        pad_top = (patch_size[0] - h_remainder) // 2
        pad_bottom = patch_size[0] - h_remainder - pad_top
    else:
        pad_top = 0
        pad_bottom = 0
    if w_remainder != 0:
        pad_left = (patch_size[1] - w_remainder) // 2
        pad_right = patch_size[1] - w_remainder - pad_left
    else:
        pad_left = 0
        pad_right = 0
    image = torch.nn.functional.pad(image, (pad_left, pad_right, pad_top, pad_bottom))
    return image, (pad_top, pad_bottom, pad_left, pad_right)


def crop_image(image, patch_size):
    # Center crop image to be divisible by patch size
    h_remainder = image.shape[2] % patch_size[0]
    w_remainder = image.shape[3] % patch_size[1]
    crop_top = h_remainder // 2
    crop_bottom = h_remainder - crop_top
    crop_left = w_remainder // 2
    crop_right = w_remainder - crop_left
    image = image[
        :,
        :,
        crop_top : (image.shape[2] - crop_bottom),
        crop_left : (image.shape[3] - crop_right),
    ]
    return image, (crop_top, crop_bottom, crop_left, crop_right)


def image_to_patches(image, patch_size, mode="crop"):
    # Center pad image to fit patch size
    if mode == "crop":
        image, borders = crop_image(image, patch_size)
    elif mode == "pad":
        image, borders = pad_image(image, patch_size)
    else:
        raise ValueError('mode must be either "crop" or "pad"')

    # get n_patches_h and n_patches_w
    n_patches_h = image.shape[2] // patch_size[0]
    n_patches_w = image.shape[3] // patch_size[1]

    # Image unfolding
    patches = (
        image.unfold(2, *patch_size)
        .unfold(3, *patch_size)
        .permute(0, 2, 3, 1, 4, 5)
        .flatten(0, 2)
        .unsqueeze(0)
    )

    return (
        patches,
        (n_patches_h, n_patches_w),
        borders,
        mode,
    )


def get_heatmaps(model, image, patch_size):
    image = crop_image(image, patch_size)[0]
    patches, shape, _, _ = image_to_patches(image, patch_size)
    with torch.no_grad():
        model.eval().cuda()
        logits = model(patches.squeeze().cuda())  # [0]
        mmlogits = (logits - logits.min()) / (logits.max() - logits.min())
    maps = (
        torch.nn.functional.interpolate(
            input=mmlogits.reshape(*shape, -1).permute(2, 0, 1).unsqueeze(0),
            size=(224 * shape[0], 224 * shape[1]),
            mode="bicubic",
        )
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    heatmaps = np.stack(
        [
            np.mean(
                [
                    plt.cm.viridis(map)[:, :, :3],
                    image.squeeze().permute(1, 2, 0).detach().numpy(),
                ],
                axis=0,
            )
            for map in maps
        ],
        axis=0,
    )
    return heatmaps


def jigsaw_to_image(x, grid_size, borders, mode="crop"):
    # x shape is batch_size x num_patches x c x jigsaw_h x jigsaw_w
    batch_size, num_patches, c, jigsaw_h, jigsaw_w = x.size()
    assert num_patches == grid_size[0] * grid_size[1]
    x_image = x.view(batch_size, grid_size[0], grid_size[1], c, jigsaw_h, jigsaw_w)
    output_h = grid_size[0] * jigsaw_h
    output_w = grid_size[1] * jigsaw_w
    x_image = x_image.permute(0, 3, 1, 4, 2, 5).contiguous()
    x_image = x_image.view(batch_size, c, output_h, output_w)

    if mode == "pad":
        # Crop image to remove pad
        x_image = x_image[
            :,
            :,
            borders[0] : (output_h - borders[1]),
            borders[2] : (output_w - borders[3]),
        ]
    elif mode == "crop":
        # Pad image to restore original size08=
        x_image = torch.nn.functional.pad(
            x_image, (borders[2], borders[3], borders[0], borders[1])
        )
    else:
        raise ValueError('mode must be either "crop" or "pad"')

    return x_image
