import json
from typing import Any

import keras.callbacks
import keras.layers
import keras.models
from keras_tuner import HyperModel
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.tuners.randomsearch import RandomSearch

from cloud_classification.utils import Jitter
from cloud_fraction.utils import prepare_test_train


class HyperParamTuningCloudFrac(HyperModel):
    def build(
            self,
            hp: HyperParameters,
    ) -> keras.models.Sequential:
        filters_lvl_1 = hp.Int(
            name='filters_conv_lvl_1',
            min_value=16,
            max_value=128,
            step=2,
            sampling='log',
        )
        model = keras.models.Sequential()
        model.add(keras.Input(shape=(128, 128, 3)))
        # level 1
        model.add(
            keras.layers.Conv2D(
                filters_lvl_1,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            ),
        )
        model.add(
            keras.layers.Conv2D(
                filters_lvl_1,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            ),
        )
        model.add(
            keras.layers.Conv2D(
                filters_lvl_1,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            ),
        )

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        filters_lvl_2 = hp.Int(
            name='filters_conv_lvl_2',
            min_value=64,
            max_value=512,
            step=2,
            sampling='log',
        )
        # level 2
        model.add(
            keras.layers.Conv2D(
                filters_lvl_2,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            ),
        )
        model.add(
            keras.layers.Conv2D(
                filters_lvl_2,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            ),
        )
        model.add(
            keras.layers.Conv2D(
                filters_lvl_2,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            ),
        )

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # level 3
        filters_lvl_3 = hp.Int(
            name='filters_conv_lvl_3',
            min_value=256,
            max_value=1024,
            step=2,
            sampling='log',
        )
        model.add(
            keras.layers.Conv2D(
                filters_lvl_3,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            ),
        )
        model.add(
            keras.layers.Conv2D(
                filters_lvl_3,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            ),
        )
        model.add(
            keras.layers.Conv2D(
                filters_lvl_3,
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
                filters_lvl_2,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            ),
        )
        model.add(
            keras.layers.Conv2D(
                filters_lvl_2,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            ),
        )
        model.add(
            keras.layers.Conv2D(
                filters_lvl_2,
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
                filters_lvl_1,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            ),
        )
        model.add(
            keras.layers.Conv2D(
                filters_lvl_1,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            ),
        )
        model.add(
            keras.layers.Conv2D(
                filters_lvl_1,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
            ),
        )
        # finally
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        # optimize the learning rate
        learning_rate = hp.Choice(
            name='learning_rate',
            values=[1e-2, 1e-3, 1e-4],
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=['accuracy'],
        )
        return model

    def fit(
            self,
            hp: HyperParameters,
            model: keras.models.Sequential,
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', [32, 64, 128, 256]),
            **kwargs,
        )


def main() -> int:
    jitter = Jitter(color=[0.66, 1, 1.33])
    print('preparing model data...')
    model_data = prepare_test_train(
        daytime_dir='data/swimseg',
        nighttime_dir='data/swinseg',
        jitter=jitter,
    )
    print('compiling model for hyperparameter tuning...')

    tuner = RandomSearch(
        hypermodel=HyperParamTuningCloudFrac(),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        seed=42069,
        overwrite=False,
        directory='hyper_param_tuning_cloud_frac',
        project_name='hyper_cloud_frac',
        max_consecutive_failed_trials=10,
    )
    print(' search_space_summary '.center(79, '='))
    tuner.search_space_summary()
    tuner.search(
        x=model_data.x_train,
        y=model_data.y_train,
        epochs=150,
        validation_data=(model_data.x_test, model_data.y_test),
        callbacks=[
            keras.callbacks.TensorBoard('tensorboard_logs_cloud_frac'),
            # don't continue training models if there is no improvement for
            # 10 epochs
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                start_from_epoch=30,
            ),
        ],
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
