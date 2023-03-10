import json
from typing import Any

import keras.callbacks
import keras.layers
import keras.models
from keras_tuner import HyperModel
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.tuners.randomsearch import RandomSearch

from cloud_classification.utils import Jitter
from cloud_classification.utils import prepare_test_train


class HyperParamTuningCloudClass(HyperModel):
    # https://github.com/keras-team/keras-tuner/issues/122#issuecomment-1152541536
    def build(
            self,
            hp: HyperParameters,
    ) -> keras.models.Sequential:
        model = keras.models.Sequential()
        model.add(keras.Input(shape=(128, 128, 3)))
        # check how many convolutions make sense
        for i in range(hp.Int('cnn_layers', 1, 6)):
            model.add(
                keras.layers.Conv2D(
                    filters=hp.Int(
                        f'filters_{i}', 16, 512,
                        step=2, sampling='log',
                    ),
                    kernel_size=(4, 4),
                    activation='relu',
                    padding='same',
                ),
            )
            model.add(keras.layers.MaxPooling2D((2, 2)))

        model.add(keras.layers.Flatten())

        # optimize dropout rate
        dropout_hp = hp.Choice(
            name='dropout_rate',
            values=[0.05, 0.1, 0.2, 0.4, 0.8],
        )
        model.add(keras.layers.Dropout(rate=dropout_hp))
        # there are 5 possible classes
        model.add(keras.layers.Dense(units=5, activation='softmax'))

        # tune the learning rate of the optimizer
        learning_rate = hp.Choice(
            name='learning_rate',
            values=[1e-2, 1e-3, 1e-4],
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
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
        swimcat_dir='data/swimcat',
        swimcat_ext_dir='data/swimcat-ext',
        rgs_dir='data/rgs/classified',
        jitter=jitter,
    )
    print('compiling model for hyperparameter tuning...')

    tuner = RandomSearch(
        hypermodel=HyperParamTuningCloudClass(),
        objective='val_accuracy',
        max_trials=75,
        executions_per_trial=1,
        seed=42069,
        overwrite=True,
        directory='hyper_param_tuning',
        project_name='test_hyper',
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
            keras.callbacks.TensorBoard('tensorboard_logs_cloud_class_tuning'),
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
