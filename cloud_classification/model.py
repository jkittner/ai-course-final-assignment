from typing import Any

import keras
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from cloud_classification.utils import prepare_test_train


def build_model(input_shape: tuple[int, ...]) -> keras.models.Sequential:
    print('constructing model...')
    model = keras.models.Sequential()
    # conv 1
    model.add(
        keras.layers.Conv2D(
            filters=32,
            kernel_size=(11, 11),
            padding='same',
            activation='relu',
            input_shape=input_shape,
        ),
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
    # conv 2
    model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3)))
    # conv 3
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    # conv4
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=0.5))
    # there are 5 possible classes
    model.add(keras.layers.Dense(units=5))

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model


def evaluate_model(
        history: keras.callbacks.History,
        model: keras.models.Sequential,
        # x: keras.engine.sequential.Sequential
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

    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred.argmax(axis=1))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig('cm.png')
    plt.close()


def main() -> int:
    print('preparing model data...')
    model_data = prepare_test_train(
        swimcat_dir='data/swimcat',
        swimcat_ext_dir='data/swimcat-ext',
        rgs_dir='data/rgs/classified',
    )
    print('compiling model...')
    model = build_model(input_shape=model_data.x_train.shape[1:])
    print('fitting model...')
    history = model.fit(
        x=model_data.x_train,
        y=model_data.y_train,
        epochs=100,
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
