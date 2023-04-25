import pandas as pd
from PIL import Image
from PIL import ImageDraw
from PIL import UnidentifiedImageError

from cloud_classification.utils import IMG_MASK
from cloud_classification.utils import IMG_SHAPE


def main() -> int:
    # see: https://stackoverflow.com/a/16377244/17798119
    # https://note.nkmk.me/en/python-pillow-gif/

    fnames = pd.read_csv(
        'data/classified_rgs.csv',
        index_col='date',
        parse_dates=['date'],
    )
    fnames = fnames[['fname']].loc[
        '2015-03-31 18:00':'2015-04-01 18:30'  # type: ignore[misc]
    ].sort_index()

    images = []
    for date, file_name in fnames.itertuples():
        print(file_name)
        try:
            img = Image.open(file_name)
            img = img.crop(IMG_MASK)
            img = img.resize(IMG_SHAPE)
            draw = ImageDraw.Draw(img)
            draw.text((8, 0), f'{date:%Y-%m-%d %H:%M:%S}', (255, 255, 255))
            images.append(img)
        except UnidentifiedImageError:
            print(f'ERROR: could not read {file_name}')
            continue

    images[0].save(
        'figs/timelapse.gif',
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=150,
        loop=0,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
