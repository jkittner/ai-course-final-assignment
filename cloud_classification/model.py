from typing import Any

import keras
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from cloud_classification.utils import Jitter
from cloud_classification.utils import prepare_test_train


def build_model(input_shape: tuple[int, ...]) -> keras.models.Sequential:
    print('constructing model...')
    model = keras.models.Sequential()
    model.add(keras.Input(shape=input_shape))
    # conv 1
    model.add(
        keras.layers.Conv2D(
            filters=16,
            kernel_size=(4, 4),
            activation='relu',
            padding='same',
        ),
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # conv 2
    model.add(
        keras.layers.Conv2D(
            filters=512,
            kernel_size=(4, 4),
            activation='relu',
            padding='same',
        ),
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # conv 3
    model.add(
        keras.layers.Conv2D(
            filters=64,
            kernel_size=(4, 4),
            activation='relu',
            padding='same',
        ),
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # conv 2
    model.add(
        keras.layers.Conv2D(
            filters=32,
            kernel_size=(4, 4),
            activation='relu',
            padding='same',
        ),
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # conv 3
    model.add(
        keras.layers.Conv2D(
            filters=64,
            kernel_size=(4, 4),
            activation='relu',
            padding='same',
        ),
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # conv 4
    model.add(
        keras.layers.Conv2D(
            filters=256,
            kernel_size=(4, 4),
            activation='relu',
            padding='same',
        ),
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=0.1))
    # there are 5 possible classes
    model.add(keras.layers.Dense(units=5, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
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
    plt.savefig('learning_cloud_class.png')
    plt.close()

    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print(f'{test_loss=}, {test_acc=}')

    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred.argmax(axis=1))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig('cm.png')
    plt.close()


def main() -> int:
    print('preparing model data...')
    jitter = Jitter(color=[0.66, 1, 1.33])
    model_data = prepare_test_train(
        swimcat_dir='data/swimcat',
        swimcat_ext_dir='data/swimcat-ext',
        rgs_dir='data/rgs/classified',
        jitter=jitter,
    )
    print('compiling model...')
    model = build_model(input_shape=model_data.x_train.shape[1:])
    print('fitting model...')
    history = model.fit(
        x=model_data.x_train,
        y=model_data.y_train,
        epochs=100,
        batch_size=32,
        validation_data=(model_data.x_test, model_data.y_test),
        verbose=1,
        callbacks=[
            keras.callbacks.TensorBoard('tensorboard_logs_cloud_class_final'),
        ],
    )
    print('saving model...')
    model.save('cloud_class_model.h5')

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
