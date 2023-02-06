from __future__ import annotations

import glob
import os
from collections.abc import Callable
from typing import Any
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from PIL import Image
from PIL import ImageEnhance
from sklearn.model_selection import train_test_split


class ImgMask(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int


IMG_MASK = ImgMask(left=50, top=40, right=210, bottom=200)
IMG_SHAPE = (125, 125)
BASE_DIR_RGS = 'data/rgs'
SWIMCAT_CLASSES = {
    'A-sky': 0,
    'B-pattern': 1,
    'C-thick-dark': 2,
    'D-thick-white': 3,
    'E-veil': 4,
}
SWIMCAT_EXT_CLASSES = {
    'A-Clear Sky': 0,
    'B-Patterned Clouds': 1,
    'E-Thick Dark Clouds': 2,
    'D-Thick White Clouds': 3,
    'F-Veil Clouds': 4,
}


class Jitter:
    def __init__(
            self,
            color: list[float] = [1],
            contrast: list[float] = [1],
            brightness: list[float] = [1],
            sharpness: list[float] = [1],
    ) -> None:
        self.color = color
        self.contrast = contrast
        self.brightness = brightness
        self.sharpness = sharpness

    def __call__(self, img: Image.Image) -> list[npt.NDArray[np.int_]]:
        jitted = []
        for col in self.color:
            col_converter = ImageEnhance.Color(img)
            img_jitted = col_converter.enhance(col)
            for con in self.contrast:
                con_converter = ImageEnhance.Contrast(img_jitted)
                img_jitted = con_converter.enhance(con)
                for b in self.brightness:
                    bright_converter = ImageEnhance.Brightness(img_jitted)
                    img_jitted = bright_converter.enhance(b)
                    for s in self.sharpness:
                        sharp_converter = ImageEnhance.Sharpness(img_jitted)
                        img_jitted = sharp_converter.enhance(s)
                        jitted.append(np.array(img_jitted))

        return jitted


def read_img(
        path: str,
        *,
        enhancements: Callable[[Image.Image], Image.Image] | None = None,
        color_jitter: Jitter | None = None,
        size: tuple[int, int] = IMG_SHAPE,
) -> npt.NDArray[np.int_]:
    files = sorted(glob.glob(path))
    if not files:
        raise ValueError(f'did not find any images in {path}...')

    images: list[Image.Image] = []
    for file in files:
        image = Image.open(file)
        if enhancements is not None:
            image = enhancements(image)

        # resize the file so we have all uniform
        image = image.resize(size)
        if color_jitter is not None:
            images.extend(color_jitter(image))
        else:
            image = np.array(image)
            images.append(image)

    return np.asarray(images)


class ModelInput(NamedTuple):
    x: npt.NDArray[np.number[Any]]
    y: npt.NDArray[np.number[Any]]

    def normalize(
            self,
            divisor: float,
            *,
            dtype: npt.DTypeLike = np.float64,
            only_x: bool = True,
    ) -> ModelInput:
        if only_x is True:
            return ModelInput(
                x=np.divide(self.x, divisor, dtype=dtype),
                y=self.y,
            )
        else:
            return ModelInput(
                x=np.divide(self.x, divisor, dtype=dtype),
                y=np.divide(self.y, divisor, dtype=dtype),
            )


class TestTrainInput(NamedTuple):
    x_train: npt.NDArray[np.number[Any]]
    x_test: npt.NDArray[np.number[Any]]
    y_train: npt.NDArray[np.number[Any]]
    y_test: npt.NDArray[np.number[Any]]


def prep_img_data(
        basedir: str,
        classes_dict: dict[str, int],
        file_pattern: str,
        jitter: Jitter | None = None,
) -> ModelInput:
    x_list = []
    y_list = []

    for img_class in classes_dict:
        imgs = read_img(
            path=os.path.join(basedir, img_class, file_pattern),
            color_jitter=jitter,
            size=IMG_SHAPE,
        )
        x_list.append(imgs)
        y_list.append(np.full(len(imgs), classes_dict[img_class]))

    x_swimcat = np.concatenate(x_list)
    y_swimcat = np.concatenate(y_list)
    return ModelInput(x=x_swimcat, y=y_swimcat)


def prepare_test_train(
        swimcat_dir: str,
        swimcat_ext_dir: str,
        rgs_dir: str,
        *,
        test_size: float = 0.2,
        max_nr: int | None = None,
        jitter: Jitter | None = None,
) -> TestTrainInput:
    swimcat = prep_img_data(
        basedir=swimcat_dir,
        classes_dict=SWIMCAT_CLASSES,
        file_pattern='*.png',
        jitter=jitter,
    )
    swimcat_ext = prep_img_data(
        basedir=swimcat_ext_dir,
        classes_dict=SWIMCAT_EXT_CLASSES,
        file_pattern='*.png',
        jitter=jitter,
    )
    rgs = prep_img_data(
        basedir=rgs_dir,
        classes_dict=SWIMCAT_CLASSES,
        file_pattern='*.jpg',
        jitter=jitter,
    )

    x_list = []
    y_list = []

    for dataset in (swimcat, swimcat_ext, rgs):
        # TODO (JK): maybe set the dtype to np.float16
        dataset_norm = dataset.normalize(divisor=255.0)
        x_list.append(dataset_norm.x)
        y_list.append(dataset_norm.y)

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)

    assert len(x) == len(y)

    if max_nr is not None:
        selection_idx = np.random.randint(0, len(x), size=max_nr)
        x = x[selection_idx]
        y = y[selection_idx]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=test_size,
        random_state=42,
        shuffle=True,
    )
    return TestTrainInput(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )
