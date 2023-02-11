from __future__ import annotations

import hashlib
import os
from datetime import datetime
from datetime import timezone

import keras
import numpy as np
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
    KERAS_CLOUD_CLASS_MODEL = keras.models.load_model('cloud_class_model')
    KERAS_CLOUD_FRAC_MODEL = None
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
        return self.date.strftime('%Y-%m-%d %H:%M-%S')


ALLOWED_FILE_TYPES = ('jpg',)


class SubmitImage(FlaskForm):
    imagefile = FileField(
        'Upload an Image',
        validators=(
            FileAllowed(
                ALLOWED_FILE_TYPES,
                message=f'Only {",".join(ALLOWED_FILE_TYPES)} Files Allowed',
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


def run_model(img_path: str) -> tuple[float, str]:
    img = read_img(path=img_path)
    pred = app.config['KERAS_CLOUD_CLASS_MODEL'].predict(img)
    cloud_class = int(np.argmax(pred))
    return 69.420, app.config['SWIMCAT_CLASS_MAP'][cloud_class]


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

        fname = os.path.join(app.config['DATA_DIR'], f'{img_hash}.jpg')
        with open(fname, 'wb') as f:
            f.write(img_bytes)

        cloud_frac, cloud_type = run_model(fname)

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
