import functools
import json

import keras.callbacks
import keras.layers
import keras.models
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.tuners.randomsearch import RandomSearch

from cloud_classification.utils import Jitter
from cloud_classification.utils import prepare_test_train


def build_model(
        hp: HyperParameters,
        input_shape: tuple[int, ...],
) -> keras.models.Sequential:
    model = keras.models.Sequential()

    # optimize 1st convolution
    kernel_size_conv1 = hp.Int(
        name='kernel_size_conv1',
        min_value=6,
        max_value=12,
        step=1,
    )
    filter_conv1 = hp.Int(
        name='filters_conv1',
        min_value=32,
        max_value=512,
        step=2,
        sampling='log',
    )
    model.add(
        keras.layers.Conv2D(
            filters=filter_conv1,
            kernel_size=(kernel_size_conv1, kernel_size_conv1),
            padding='same',
            activation='relu',
            input_shape=input_shape,
        ),
    )

    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))

    # optimize 2nd convolution
    filter_conv2 = hp.Int(
        name='filters_conv2',
        min_value=16,
        max_value=256,
        step=2,
        sampling='log',
    )
    model.add(keras.layers.Conv2D(filter_conv2, (5, 5), activation='relu'))

    model.add(keras.layers.MaxPooling2D((3, 3)))

    # optimize filters of 3rd convolution
    filter_conv3 = hp.Int(
        name='filters_conv3',
        min_value=16,
        max_value=256,
        step=2,
        sampling='log',
    )
    model.add(keras.layers.Conv2D(filter_conv3, (3, 3), activation='relu'))

    # optimize filter of 4th convolution
    filter_conv4 = hp.Int(
        name='filters_conv4',
        min_value=16,
        max_value=256,
        step=2,
        sampling='log',
    )
    model.add(keras.layers.Conv2D(filter_conv4, (3, 3), activation='relu'))

    model.add(keras.layers.MaxPooling2D((3, 3)))
    model.add(keras.layers.Flatten())

    # optimize dropout rate
    dropout_hp = hp.Float(
        name='dropout_rate',
        min_value=0.05,
        max_value=0.8,
        step=0.05,
    )
    model.add(keras.layers.Dropout(rate=dropout_hp))
    # there are 5 possible classes
    model.add(keras.layers.Dense(units=5))

    # tune the learning rate of the optimizer
    learning_rate = hp.Float(
        name='learning_rate',
        min_value=1e-4,
        max_value=1e-2,
        sampling='log',
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model


def main() -> int:
    jitter = Jitter(color=[0.66, 1, 1.33])
    print('preparing model data...')
    model_data = prepare_test_train(
        swimcat_dir='data/swimcat',
        swimcat_ext_dir='data/swimcat-ext',
        rgs_dir='data/rgs/classified',
        jitter=jitter,
    )
    print('compiling model for hyperparameter tuning...')

    build_model_partial = functools.partial(
        build_model,
        input_shape=model_data.x_train.shape[1:],
    )
    tuner = RandomSearch(
        hypermodel=build_model_partial,
        objective='val_accuracy',
        max_trials=100,
        executions_per_trial=1,
        seed=42069,
        overwrite=True,
        directory='hyper_param_tuning',
        project_name='test_hyper',
    )
    print(' search_space_summary '.center(79, '='))
    tuner.search_space_summary()
    tuner.search(
        x=model_data.x_train,
        y=model_data.y_train,
        epochs=15,
        validation_data=(model_data.x_test, model_data.y_test),
        callbacks=[keras.callbacks.TensorBoard('tensorboard_logs')],
    )
    print(' results_summary '.center(79, '='))
    print(tuner.results_summary())
    print(' best models '.center(79, '='))
    models = tuner.get_best_models(num_models=3)
    for m in models:
        m.summary()

    print(' best hyperparameters '.center(79, '='))
    best_hyper_params = tuner.get_best_hyperparameters(5)
    for hp in best_hyper_params:
        print(json.dumps(hp.values, indent=4))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
