from __future__ import annotations

import os

import numpy as np
from sklearn.model_selection import train_test_split

from cloud_classification.utils import Jitter
from cloud_classification.utils import ModelInput
from cloud_classification.utils import read_img
from cloud_classification.utils import TestTrainInput


def prep_img_data(
        basedir: str,
        file_pattern: str,
        jitter: Jitter | None = None,
) -> ModelInput:
    x = read_img(
        path=os.path.join(basedir, 'Images', file_pattern),
        color_jitter=jitter,
        size=(128, 128),
    )
    y = read_img(
        path=os.path.join(basedir, 'GTmaps', file_pattern),
        color_jitter=jitter,
        size=(128, 128),
    )
    assert len(x) == len(y)
    return ModelInput(x=x, y=y)


def prepare_test_train(
        daytime_dir: str,
        nighttime_dir: str,
        *,
        test_size: float = 0.2,
        max_nr: int | None = None,
        jitter: Jitter | None = None,
) -> TestTrainInput:
    day = prep_img_data(
        basedir=daytime_dir,
        # basedir='data/swimseg',
        file_pattern='*.png',
        jitter=jitter,
    ).normalize(divisor=255.0, only_x=False)
    night = prep_img_data(
        basedir=nighttime_dir,
        # basedir='data/swinseg',
        file_pattern='*.jpg',
        jitter=jitter,
    ).normalize(divisor=255.0, only_x=False)

    x = np.concatenate([day.x, night.x])
    y = np.concatenate([day.y, night.y])

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
