import os
import shutil

import numpy as np


def main() -> int:
    rand_state = np.random.RandomState(6942069)

    # this is on the server
    INPUT_ROOT = '/data/obs/backups/stations-rub-gis6/rgs2_div/allskycam'
    NR_IMG_TO_SELECT = 5 * 75

    all_images = []

    for root, _, files in os.walk(INPUT_ROOT):
        # filter for files that are either jpeg or png
        for name in files:
            if name.endswith(('jpg', 'png')):
                all_images.append(os.path.join(root, name))
            else:
                print(f'skipping {name}')

    # randomly select a few images
    rand_idx = rand_state.randint(0, len(all_images), NR_IMG_TO_SELECT)
    for idx in rand_idx:
        shutil.copy(all_images[idx], 'selected_img')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
