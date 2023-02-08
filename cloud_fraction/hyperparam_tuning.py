import functools
import json

import keras.callbacks
import keras.layers
import keras.models
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.tuners.randomsearch import RandomSearch

from cloud_classification.utils import Jitter
from cloud_fraction.utils import prepare_test_train


def build_model(
        hp: HyperParameters,
        input_shape: tuple[int, ...],
) -> keras.models.Sequential:
    filters = hp.Int(
        name='filters_conv',
        min_value=16,
        max_value=512,
        step=2,
        sampling='log',
    )
    activation_conv = hp.Choice(
        name='activations_conv',
        values=('relu', 'tanh', 'sigmoid'),
    )
    model = keras.models.Sequential()
    # level 1
    model.add(
        keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
            input_shape=input_shape,
        ),
    )
    model.add(
        keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
        ),
    )
    model.add(
        keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
        ),
    )

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # level 2
    model.add(
        keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
        ),
    )
    model.add(
        keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
        ),
    )
    model.add(
        keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
        ),
    )

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # level 3
    model.add(
        keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
        ),
    )
    model.add(
        keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
        ),
    )
    model.add(
        keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
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
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
        ),
    )
    model.add(
        keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
        ),
    )
    model.add(
        keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
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
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
        ),
    )
    model.add(
        keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
        ),
    )
    model.add(
        keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation=activation_conv,
        ),
    )
    # finally
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # optimize the learning rate
    learning_rate = hp.Float(
        name='learning_rate',
        min_value=1e-4,
        max_value=1e-2,
        sampling='log',
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'],
    )
    return model


def main() -> int:
    jitter = Jitter(color=[0.66, 1, 1.33])
    print('preparing model data...')
    model_data = prepare_test_train(
        daytime_dir='data/swimseg',
        nighttime_dir='data/swinseg',
        jitter=jitter,
    )
    print('compiling model for hyperparameter tuning...')

    build_model_partial = functools.partial(
        build_model,
        input_shape=(128, 128, 3),
    )
    tuner = RandomSearch(
        hypermodel=build_model_partial,
        objective='val_accuracy',
        max_trials=50,
        executions_per_trial=1,
        seed=42069,
        overwrite=True,
        directory='hyper_param_tuning_cloud_frac',
        project_name='test_hyper',
    )
    print(' search_space_summary '.center(79, '='))
    tuner.search_space_summary()
    tuner.search(
        x=model_data.x_train,
        y=model_data.y_train,
        epochs=100,
        validation_data=(model_data.x_test, model_data.y_test),
        callbacks=[keras.callbacks.TensorBoard('tensorboard_logs_cloud_frac')],
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
