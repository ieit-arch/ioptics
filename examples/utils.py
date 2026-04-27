from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.nn.functional import interpolate
from torchvision import datasets


def _balanced_class_counts(n_samples: int, n_classes: int) -> torch.Tensor:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")

    base = n_samples // n_classes
    remainder = n_samples % n_classes
    counts = torch.full((n_classes,), base, dtype=torch.long)
    counts[:remainder] += 1
    return counts


def _sample_mnist_subset(
    dataset: datasets.MNIST,
    n_samples: int,
    digits: Sequence[int],
    target_size: Tuple[int, int],
    normalize: bool,
    remap_labels: bool,
    flatten: bool,
    shuffle: bool,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    digits = tuple(digits)
    if len(digits) == 0:
        raise ValueError("digits must not be empty")

    class_counts = _balanced_class_counts(n_samples=n_samples, n_classes=len(digits))
    sampled_indices = []

    for i, digit in enumerate(digits):
        digit_indices = torch.where(dataset.targets == digit)[0]
        need = int(class_counts[i].item())
        if need > len(digit_indices):
            raise ValueError(
                f"Not enough samples for digit {digit}: requested {need}, available {len(digit_indices)}"
            )

        if shuffle:
            perm = torch.randperm(len(digit_indices), generator=generator)
            digit_indices = digit_indices[perm]

        sampled_indices.append(digit_indices[:need])

    selected_indices = torch.cat(sampled_indices, dim=0)

    if shuffle:
        perm = torch.randperm(len(selected_indices), generator=generator)
        selected_indices = selected_indices[perm]

    selected_data = dataset.data[selected_indices].float().unsqueeze(1)
    images = interpolate(selected_data, size=target_size, mode="bilinear", align_corners=False)

    if normalize:
        images = images / 255.0

    labels = dataset.targets[selected_indices].clone()

    if remap_labels:
        remapped = torch.empty_like(labels)
        for new_label, digit in enumerate(digits):
            remapped[labels == digit] = new_label
        labels = remapped

    if flatten:
        images = images.view(images.shape[0], -1)

    return images, labels


def create_mnist_dataset(
    n_samples: int = 100,
    digits: Sequence[int] = (3, 6),
    target_size: Tuple[int, int] = (14, 14),
    normalize: bool = True,
    seed: int = 42,
    shuffle: bool = True,
    remap_labels: bool = True,
    flatten: bool = False,
    train: bool = True,
    data_root: str = "./data",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """创建MNIST子数据集，支持可复现采样与标签重映射。"""
    generator = torch.Generator().manual_seed(seed)
    mnist = datasets.MNIST(root=data_root, train=train, download=True)
    return _sample_mnist_subset(
        dataset=mnist,
        n_samples=n_samples,
        digits=digits,
        target_size=target_size,
        normalize=normalize,
        remap_labels=remap_labels,
        flatten=flatten,
        shuffle=shuffle,
        generator=generator,
    )


def create_mnist_splits(
    train_samples: int,
    val_samples: int,
    test_samples: int,
    digits: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    target_size: Tuple[int, int] = (14, 14),
    normalize: bool = True,
    seed: int = 42,
    shuffle: bool = True,
    remap_labels: bool = True,
    flatten: bool = False,
    data_root: str = "./data",
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """创建MNIST的train/val/test三路数据。"""
    if train_samples <= 0 or val_samples <= 0 or test_samples <= 0:
        raise ValueError("train_samples, val_samples and test_samples must be positive")

    generator = torch.Generator().manual_seed(seed)

    mnist_train = datasets.MNIST(root=data_root, train=True, download=True)
    mnist_test = datasets.MNIST(root=data_root, train=False, download=True)

    total_train_val = train_samples + val_samples
    train_val_x, train_val_y = _sample_mnist_subset(
        dataset=mnist_train,
        n_samples=total_train_val,
        digits=digits,
        target_size=target_size,
        normalize=normalize,
        remap_labels=remap_labels,
        flatten=flatten,
        shuffle=shuffle,
        generator=generator,
    )

    if shuffle:
        perm = torch.randperm(total_train_val, generator=generator)
        train_val_x = train_val_x[perm]
        train_val_y = train_val_y[perm]

    train_x = train_val_x[:train_samples]
    train_y = train_val_y[:train_samples]
    val_x = train_val_x[train_samples:]
    val_y = train_val_y[train_samples:]

    test_x, test_y = _sample_mnist_subset(
        dataset=mnist_test,
        n_samples=test_samples,
        digits=digits,
        target_size=target_size,
        normalize=normalize,
        remap_labels=remap_labels,
        flatten=flatten,
        shuffle=shuffle,
        generator=generator,
    )

    return {
        "train": (train_x, train_y),
        "val": (val_x, val_y),
        "test": (test_x, test_y),
    }


def create_mog7_dataset(
    n_samples: int = 200,
    std: float = 0.08,
    radius: float = 2.0,
    seed: int = 42,
    shuffle: bool = True,
    return_labels: bool = False,
    min_value: float = 1e-6,
    max_value: float = 2.0,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """创建7点环形MoG训练集（无中心点，坐标缩放到[min_value, max_value]）。"""
    if std <= 0:
        raise ValueError("std must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if min_value <= 0:
        raise ValueError("min_value must be positive")
    if max_value <= min_value:
        raise ValueError("max_value must be greater than min_value")

    generator = torch.Generator().manual_seed(seed)

    angles = torch.linspace(0, 2 * torch.pi, steps=8)[:-1]
    ring_centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    center_shift = radius + 3.0 * std + min_value
    centers = ring_centers + center_shift

    class_counts = _balanced_class_counts(n_samples=n_samples, n_classes=7)

    data_list = []
    label_list = []
    for class_idx, center in enumerate(centers):
        count = int(class_counts[class_idx].item())
        noise = torch.randn((count, 2), generator=generator) * std
        samples = noise + center
        samples = torch.clamp(samples, min=min_value)
        data_list.append(samples)
        label_list.append(torch.full((count,), class_idx, dtype=torch.long))

    data = torch.cat(data_list, dim=0)
    labels = torch.cat(label_list, dim=0)

    current_min = data.min()
    current_max = data.max()
    scale = (max_value - min_value) / (current_max - current_min + 1e-12)
    data = (data - current_min) * scale + min_value

    if shuffle:
        perm = torch.randperm(data.shape[0], generator=generator)
        data = data[perm]
        labels = labels[perm]

    if return_labels:
        return data, labels
    return data


def normalize_inputs(data, num_inputs, P0 = 10):
    _, input_size = data.shape
    injection_port = input_size
    data_normalized = np.array(
        np.pad(data, ((0,0), (0, num_inputs - input_size)), mode='constant'))
    for i, x in enumerate(data_normalized):
        power_remaining = P0 - np.sum(x**2)
        if power_remaining >= 0:
            data_normalized[i, injection_port] = np.sqrt(power_remaining)
        else:
            raise ValueError(f"输入数据能量 {np.sum(x**2)} 超过限制 {P0}")

    return data_normalized
