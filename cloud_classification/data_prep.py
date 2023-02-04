import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime

from PIL import Image

from cloud_classification.utils import BASE_DIR_RGS
from cloud_classification.utils import IMG_MASK
from cloud_classification.utils import IMG_SHAPE


@contextmanager
def open_image(path: str) -> Generator[Image.Image, None, None]:
    img = Image.open(path)
    yield img
    img.close()


def main() -> int:
    img_files = os.listdir(os.path.join(BASE_DIR_RGS, 'raw'))
    for f in img_files:
        with open_image(os.path.join(BASE_DIR_RGS, 'raw', f)) as img:
            img = img.crop(IMG_MASK)
            img = img.resize(IMG_SHAPE)
            fname = f.partition('ASC_')[-1]
            fname = os.path.splitext(fname)[0]
            fname_date = datetime.strptime(fname, '%Y%m%d_%H%M%S')
            new_fname = f"{fname_date.strftime('%Y-%m-%d_%H%M%S')}.jpg"
            img.save(os.path.join(BASE_DIR_RGS, 'processed', new_fname))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
