import os
import time
from datetime import datetime

import keras.models
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
from PIL import UnidentifiedImageError

from cloud_classification.utils import IMG_MASK
from cloud_classification.utils import IMG_SHAPE

ImageFile.LOAD_TRUNCATED_IMAGES = True

INPUT_PATH = '/data/obs/backups/stations-rub-gis6/rgs2_div/allskycam/'
BATCH_SIZE = 10000


def main() -> int:
    # 1. iterate over all images
    # 2. prepare the image (crop and resize)
    # 3. classify the image using the models
    # 4. save the classification result to a dataframe and to a CSV file
    KERAS_CLOUD_CLASS_MODEL = keras.models.load_model(
        'resources/cloud_class_model.h5',
    )
    KERAS_CLOUD_FRAC_MODEL = keras.models.load_model(
        'resources/cloud_frac_model.h5',
    )
    df_stack = []
    for root, _, files in os.walk(INPUT_PATH):
        print(f' processing folder {root} '.center(79, '='))
        start = time.monotonic()
        n = 0
        images = []
        dates = []
        file_names = []
        for file in files:
            file_name = os.path.join(root, file)
            try:
                img = Image.open(file_name)
            except UnidentifiedImageError:
                print(f'ERROR: could not read {file_name}')
                n += 1
                continue

            img = img.crop(IMG_MASK)
            img = img.resize(IMG_SHAPE)
            # normalize data to 0 - 1
            img_array = np.array(img) / 255.0
            images.append(img_array)
            # extract the date from the filename
            fname = file.partition('ASC_')[-1]
            fname = os.path.splitext(fname)[0]
            date = datetime.strptime(fname, '%Y%m%d_%H%M%S')
            dates.append(date)

            file_names.append(file_name)
            img.close()

            n += 1
            print(
                f'[{n:>5}/{len(files)}] '
                f'{(time.monotonic() - start) / n:.5f}s/img, '
                f'ellapsed time: {time.monotonic() - start:.2f}s',
            )
            if n % BATCH_SIZE == 0 or n == len(files):
                print('classifying current batch')
                current_batch = np.asarray(images)
                pred_class = KERAS_CLOUD_CLASS_MODEL.predict(current_batch)
                cloud_class = np.argmax(pred_class, axis=1)

                pred_frac = KERAS_CLOUD_FRAC_MODEL.predict(current_batch)
                # calculate the percentage of cloud cover
                cloud_frac = ((pred_frac > 0.8).sum(axis=(1, 2))
                              * 100 / (128 * 128)).reshape(-1)

                # make sure nothing goes wrong and the rows do not match
                assert len(dates) == len(cloud_class) == len(
                    cloud_frac,
                ) == len(file_names)
                df_stack.append(
                    pd.DataFrame(
                        {
                            'date': dates,
                            'cloud_class': cloud_class,
                            'cloud_frac': cloud_frac,
                            'fname': file_names,
                        },
                    ).set_index('date'),
                )
                # reset the variables
                start = time.monotonic()
                images = []
                dates = []
                file_names = []

    df_all = pd.concat(df_stack).sort_index()
    df_all.to_csv('data/classified_rgs.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
