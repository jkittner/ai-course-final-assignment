import os

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw

from cloud_classification.utils import BASE_DIR_RGS
from cloud_classification.utils import IMG_MASK


def main() -> int:
    img = Image.open(
        os.path.join(BASE_DIR_RGS, 'raw', 'ASC_20141022_113742.jpg'),
    )
    drawer = ImageDraw.Draw(img)

    img_size_cropped = img.crop(IMG_MASK).size
    drawer.rectangle(xy=IMG_MASK, outline='red', width=3)

    plt.imshow(img)
    plt.text(
        IMG_MASK.left + 5, IMG_MASK.top + 13,
        s=f'({IMG_MASK.left}/{IMG_MASK.top})',
        color='red',
        fontweight='bold',
        fontsize=14,
    )
    plt.text(
        IMG_MASK.right - 70, IMG_MASK.bottom - 7,
        s=f'({IMG_MASK.right}/{IMG_MASK.bottom})',
        color='red',
        fontweight='bold',
        fontsize=14,
    )
    plt.text(
        95, 125,
        s=f'{img_size_cropped}',
        color='red',
        fontweight='bold',
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig('figs/crop_window.png', dpi=600)
    plt.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
