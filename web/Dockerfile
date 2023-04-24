FROM python:3.10-slim-bullseye

RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        wait-for-it \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN : \
    && mkdir /usr/src/app \
    && mkdir /usr/src/app/instance

WORKDIR /usr/src/app

COPY ./requirements.txt .

RUN python -m venv venv
RUN venv/bin/pip install wheel
RUN venv/bin/pip install -r requirements.txt

COPY web web
COPY cloud_classification cloud_classification
COPY resources resources
