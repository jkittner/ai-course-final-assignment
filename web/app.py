from __future__ import annotations

import hashlib
import io
import os
from datetime import datetime
from datetime import timezone
from typing import TypeVar

import keras
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import PIL.Image
from flask import flash
from flask import Flask
from flask import redirect
from flask import render_template
from flask import url_for
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from flask_wtf.file import FileRequired
from wtforms import FileField
from wtforms import SubmitField

from cloud_classification.utils import read_img
from cloud_classification.utils import SWIMCAT_CLASSES


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'super_secret_key'
    SQLALCHEMY_DATABASE_URI = ''
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = 'sqlite:///project.db'
    KERAS_CLOUD_CLASS_MODEL = keras.models.load_model(
        'resources/cloud_class_model.h5',
    )
    KERAS_CLOUD_FRAC_MODEL = keras.models.load_model(
        'resources/cloud_frac_model.h5',
    )
    DATA_DIR = 'web/static/img'
    SWIMCAT_CLASS_MAP = {v: k for k, v in SWIMCAT_CLASSES.items()}
    # TODO: max req len!


db = SQLAlchemy()


class Image(db.Model):  # type: ignore[name-defined]
    id = db.Column(db.String, primary_key=True)
    date = db.Column(db.DateTime, nullable=False)
    cloud_fraction = db.Column(db.Numeric, nullable=False)
    cloud_type = db.Column(db.String, nullable=False)

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}('
            f'id={self.id}, '
            f'date={self.date!r}, '
            f'cloud_fraction={self.cloud_fraction!r}, '
            f'cloud_type={self.cloud_type!r}'
            ')'
        )

    @property
    def date_formatted(self) -> str:
        return self.date.strftime('%Y-%m-%d %H:%M:%S')

    @property
    def id_css_escaped(self) -> str:
        if self.id[0].isnumeric():
            return f'\\3{self.id[0]} {self.id[1:]}'
        else:
            return self.id


ALLOWED_FILE_TYPES = ('jpg', 'jpeg', 'png')


class SubmitImage(FlaskForm):
    imagefile = FileField(
        'Upload an Image',
        validators=(
            FileAllowed(
                ALLOWED_FILE_TYPES,
                message=f'Only {", ".join(ALLOWED_FILE_TYPES)} Files Allowed',
            ),
            FileRequired(),
        ),
    )
    submit = SubmitField('upload image')


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    with app.app_context():
        db.create_all()

    return app


app = create_app()

T = TypeVar('T', bound=npt.NBitBase)


def plot_probability(
        data: npt.NDArray[np.floating[T]],
        img_hash: str,
        data_dir: str = app.config['DATA_DIR'],
) -> None:
    try:
        prob = plt.imshow(data[0], cmap='nipy_spectral')
        plt.tick_params(
            axis='both', left=False, bottom=False,
            labelleft=False, labelbottom=False,
        )
        plt.colorbar(
            prob,
            ticks=np.linspace(0, 1, 11),
            location='bottom',
            fraction=.0465,
            pad=0.02,
        )
        plt.savefig(
            os.path.join(data_dir, f'{img_hash}_prob.png'),
            bbox_inches='tight',
            dpi=150,
        )
    finally:
        plt.close()


def run_models(img_path: str, img_hash: str) -> tuple[float, str]:
    img = read_img(path=img_path)
    class_model = app.config['KERAS_CLOUD_CLASS_MODEL']
    pred_class = class_model.predict(img, verbose=0)
    cloud_class = int(np.argmax(pred_class))
    frac_model = app.config['KERAS_CLOUD_FRAC_MODEL']
    pred_frac = frac_model.predict(img, verbose=0)
    cloud_frac = (pred_frac > 0.8).sum() * 100 / (128 * 128)
    plot_probability(data=pred_frac, img_hash=img_hash)
    # TODO: error if probability is below a certain threshold
    return cloud_frac, app.config['SWIMCAT_CLASS_MAP'][cloud_class]


def resize_img(img_bytes: bytes) -> PIL.Image.Image:
    img_resized = PIL.Image.open(io.BytesIO(img_bytes))

    if img_resized.mode != 'RGB':
        img_resized = img_resized.convert('RGB')

    if img_resized.size == (128, 128):
        return img_resized

    is_square = False
    if img_resized.width < img_resized.height:
        is_portrait = True
        ratio_diff = (img_resized.height - img_resized.width) // 2
    elif img_resized.width > img_resized.height:
        is_portrait = False
        ratio_diff = (img_resized.width - img_resized.height) // 2
    else:
        is_square = True

    if is_square:
        pass
    elif is_portrait and not is_square:
        img_resized = img_resized.crop((
            0, ratio_diff, img_resized.width,
            img_resized.height - ratio_diff,
        ))
    else:
        img_resized = img_resized.crop((
            ratio_diff, 0,
            img_resized.width - ratio_diff, img_resized.height,
        ))
    return img_resized.resize(size=(128, 128))


@app.route('/', methods=('GET', 'POST'))
def index() -> str:
    form = SubmitImage()
    if form.validate_on_submit():
        image = form.imagefile.data
        img_bytes = image.stream.read()
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        queried_img = db.session.execute(
            db.select(Image).filter_by(id=img_hash),
        ).one_or_none()
        print(queried_img)
        if queried_img:
            flash('The Image was Submitted Before!', category='danger')
            return redirect(url_for('index'))

        fname = os.path.join(app.config['DATA_DIR'], f'{img_hash}.png')
        resize_img(img_bytes=img_bytes).save(fname)
        cloud_frac, cloud_type = run_models(img_path=fname, img_hash=img_hash)

        submitted_img = Image(
            id=img_hash,
            date=datetime.now(timezone.utc),
            cloud_fraction=cloud_frac,
            cloud_type=cloud_type,
        )
        db.session.add(submitted_img)
        db.session.commit()
        return redirect(url_for('index'))

    img_page = db.paginate(
        db.select(Image).order_by(Image.date.desc()),
        per_page=6,
    )
    return render_template(
        'index.html',
        title='Cloud Classifier',
        form=form,
        page=img_page,
    )


@app.route('/terms')
def terms() -> str:
    return render_template('terms.html')


@app.route('/privacy')
def privacy() -> str:
    return render_template('privacy.html')


if __name__ == '__main__':
    app.run(debug=True)
