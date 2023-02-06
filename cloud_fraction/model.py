from __future__ import annotations

from typing import Any

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from cloud_fraction.utils import prepare_test_train


def build_model(input_shape: tuple[int, ...]) -> keras.models.Sequential:
    print('constructing model...')

    model = keras.models.Sequential()
    # level 1
    model.add(
        keras.layers.Conv2D(
            64,  kernel_size=(3, 3),
            padding='same', activation='relu', input_shape=input_shape,
        ),
    )
    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        ),
    )
    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        ),
    )

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # level 2
    model.add(
        keras.layers.Conv2D(
            64, kernel_size=(3, 3),
            padding='same',
            activation='relu',
        ),
    )
    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        ),
    )
    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        ),
    )

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # level 3
    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        ),
    )
    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        ),
    )
    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        ),
    )

    model.add(
        keras.layers.Conv2DTranspose(
            1,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='same',
        ),
    )
    # level 2
    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        ),
    )
    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        ),
    )
    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        ),
    )

    model.add(
        keras.layers.Conv2DTranspose(
            1,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='same',
        ),
    )
    # level 1
    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        ),
    )
    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        ),
    )
    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        ),
    )
    # finally
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'],
    )
    return model


def evaluate_model(
        history: keras.callbacks.History,
        model: keras.models.Sequential,
        x_test: npt.NDArray[np.number[Any]],
        y_test: npt.NDArray[np.number[Any]],
) -> None:
    print('evaluating model...')
    # create a plot showing the development of the accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('learning.png')
    plt.close()

    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print(f'{test_loss=}, {test_acc=}')


def main() -> int:
    print('preparing model data...')
    model_data = prepare_test_train(
        daytime_dir='data/swimseg',
        nighttime_dir='data/swinseg',
    )
    print('compiling model...')
    model = build_model(input_shape=(128, 128, 3))
    print('fitting model...')
    history = model.fit(
        x=model_data.x_train,
        y=model_data.y_train,
        epochs=30,
        validation_data=(model_data.x_test, model_data.y_test),
        verbose=1,
    )
    print('evaluating model')
    evaluate_model(
        history=history,
        model=model,
        x_test=model_data.x_test,
        y_test=model_data.y_test,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
