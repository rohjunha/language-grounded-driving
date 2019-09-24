import random
from functools import partial
from pathlib import Path
from typing import List, Tuple, Callable

import cv2

import imageio
import numpy as np
import torch
from PIL import ImageEnhance, Image
from torchvision.transforms import Normalize, ToTensor, Compose, ToPILImage


def put_text(image: np.ndarray, text: str, pos: Tuple[int, int], color: Tuple[int, int, int], scale: float):
    face = cv2.FONT_HERSHEY_DUPLEX
    thickness = 1
    ltype = cv2.LINE_AA
    margin_left = 10
    margin_bottom = 15
    margin_space = round(20 * scale)

    text_list = ['{}'.format(text)]
    sz_list = [cv2.getTextSize(text, face, scale, thickness) for text in text_list]
    for text, sz in zip(text_list, sz_list):
        # # pos = (pos[0], image.shape[1] - sz[0][1])
        # pos = (image.shape[0] - sz[0][0], pos[1])
        # corners = [(pos[0], pos[1]),
        #            (pos[0] + sz[0][0], pos[1]),
        #            (pos[0] + sz[0][0], pos[1] + sz[0][1]),
        #            (pos[0], pos[1] + sz[0][1])]
        # print(corners)
        # cv2.fillPoly(image, np.array([corners], dtype=np.int32), (0, 0, 0))
        cv2.putText(image, text, (pos[0], pos[1]),
                    face, scale, color, thickness, ltype)
        margin_bottom += sz[1] + thickness + margin_space


def to_float32(image: np.ndarray) -> np.ndarray:
    assert image.dtype == np.uint8
    data = image.astype(np.float32) / 255
    return data


def to_uint8(image: np.ndarray) -> np.ndarray:
    assert image.dtype == np.float32
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def normalize_array(image: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return np.divide(image - mean[None, None, :], std[None, None, :])


def normalize_image(image: np.ndarray) -> np.ndarray:
    assert image.dtype == np.float32
    if image.ndim == 3:
        if image.shape[-1] == 3:
            return normalize_array(image, mean=[0.655, 0.596, 0.578], std=[0.270, 0.299, 0.309])
        elif image.shape[-1] == 4:
            return normalize_array(image, mean=[0.655, 0.596, 0.578, 0], std=[0.270, 0.299, 0.309, 1])
    return image


def video_from_memory(
        images: List[np.ndarray],
        output_path: Path,
        texts: List[str] = [],
        framerate: int = 30,
        revert: bool = True):
    kargs = {
        'quality': 10,
        'macro_block_size': None,
        'fps': framerate,
        'ffmpeg_log_level': 'panic'}
    if texts and len(texts) == len(images):
        for image, text in zip(images, texts):
            lines = text.splitlines()
            for i, line in enumerate(lines):
                put_text(image, line, (12, 12 + 10 * i), (255, 255, 255), 0.3)
    if revert:
        images = [image[:, :, ::-1] for image in images]
    imageio.mimwrite(str(output_path), images, 'mp4', **kargs)


def video_from_files(
        filepaths: List[Path],
        output_path: Path,
        texts: List[str] = [],
        framerate: int = 30,
        revert: bool = True):
    images = [cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED) for filepath in filepaths]
    video_from_memory(images, output_path, texts, framerate, revert)


def transform_with_probability(
        image: np.ndarray,
        prob: float,
        transform):
    if random.uniform(0, 1) < prob:
        return transform(image)
    else:
        return image


def _gaussian_blur(image: np.ndarray):
    assert image.dtype == np.float32
    std = random.uniform(0, 1.3)
    cv2.GaussianBlur(image, (3, 3), std, image)


def _additive_gaussian_noise(image: np.ndarray):
    assert image.dtype == np.float32
    std = random.uniform(0, 0.05)
    image += np.random.normal(0, std, image.shape)


def _spatial_dropout(image: np.ndarray):
    d = random.uniform(0, 0.1)
    h, w = image.shape[0], image.shape[1]
    d = round(d * h * w)
    ys = np.random.randint(0, h, d)
    xs = np.random.randint(0, w, d)
    for i in range(d):
        image[ys[i], xs[i], :] = 0


def _brightness_additive(image: np.ndarray):
    assert image.dtype == np.float32
    if image.ndim == 3:
        nc = image.shape[2]
        for i in range(nc):
            image[:, :, i] += random.uniform(-0.08, 0.08)
    elif image.ndim == 2:
        image += random.uniform(-0.08, 0.08)
    else:
        raise ValueError('ndim should be 2 or 3: {}'.format(image.ndim))


def _brightness_multiplicative(image: np.ndarray):
    assert image.dtype == np.float32
    if image.ndim == 3:
        nc = image.shape[2]
        for i in range(nc):
            image[:, :, i] *= random.uniform(0.25, 2.5)
    elif image.ndim == 2:
        image *= random.uniform(0.25, 2.5)
    else:
        raise ValueError('ndim should be 2 or 3: {}'.format(image.ndim))


def _contrast_multiplicative(image: np.ndarray):
    assert image.dtype == np.float32
    cf = random.uniform(0.5, 1.5)
    pil_image = Image.fromarray(to_uint8(image))
    enhancer = ImageEnhance.Contrast(pil_image)
    img = enhancer.enhance(cf)
    return to_float32(np.array(img))


def _saturation_multiplicative(image: np.ndarray):
    assert image.dtype == np.float32
    sf = random.uniform(0, 1)
    pil_image = Image.fromarray(to_uint8(image))
    enhancer = ImageEnhance.Color(pil_image)
    img = enhancer.enhance(sf)
    return to_float32(np.array(img))


def test():
    prob1 = 1.0  # 0.05
    prob2 = 1.0  # 0.1
    image = np.random.uniform(0, 1, (88, 200, 3))

    gaussian_blur = partial(transform_with_probability, prob=prob1, transform=_gaussian_blur)
    additive_gaussian_noise = partial(transform_with_probability, prob=prob1, transform=_additive_gaussian_noise)
    spatial_dropout = partial(transform_with_probability, prob=prob1, transform=_spatial_dropout)
    brightness_additive = partial(transform_with_probability, prob=prob2, transform=_brightness_additive)

    images = []
    images.append(np.array(image, copy=True))
    transforms = [gaussian_blur, additive_gaussian_noise, spatial_dropout, brightness_additive]
    for transform in transforms:
        transform(image)
        images.append(np.array(image, copy=True))
    print(len(images))

    for i, (j, k) in enumerate(zip(images[:-1], images[1:])):
        print(i, np.sum(np.abs(k - j)[:]) / k.size)


prob1 = 0.05
prob2 = 0.1
prob3 = 0.2
gaussian_blur = partial(transform_with_probability, prob=prob1, transform=_gaussian_blur)
additive_gaussian_noise = partial(transform_with_probability, prob=prob1, transform=_additive_gaussian_noise)
spatial_dropout = partial(transform_with_probability, prob=prob1, transform=_spatial_dropout)
brightness_additive = partial(transform_with_probability, prob=prob2, transform=_brightness_additive)
brightness_multiplicative = partial(transform_with_probability, prob=prob3, transform=_brightness_multiplicative)
contrast_multiplicative = partial(transform_with_probability, prob=prob1, transform=_contrast_multiplicative)
saturation_multiplicative = partial(transform_with_probability, prob=prob1, transform=_saturation_multiplicative)
__transforms__ = [gaussian_blur, additive_gaussian_noise, spatial_dropout, brightness_additive,
                  brightness_multiplicative, contrast_multiplicative, saturation_multiplicative]


def apply_transforms(in_image: np.ndarray):
    assert in_image.dtype == np.float32
    random.shuffle(__transforms__)
    for transform in __transforms__:
        transform(in_image)
    return in_image


def tensor_from_numpy_image(image: np.ndarray, apply_transform: bool) -> torch.Tensor:
    image = to_float32(image)
    if apply_transform:
        image = apply_transforms(image)
    image = normalize_image(image)
    # image = to_uint8(image)
    return ToTensor()(image)


if __name__ == '__main__':
    in_image = cv2.imread(str(Path.home() / 'projects/carla_python/00010000c.png'))
    in_image = to_float32(in_image)
    for i in range(10):
        apply_transforms(in_image, i)
