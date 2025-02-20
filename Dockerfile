FROM nvidia/cuda:12.4.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential libreadline-dev \ 
libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev libffi-dev libdb-dev

RUN wget --no-check-certificate https://www.python.org/ftp/python/3.9.21/Python-3.9.21.tgz \
    && tar -xf Python-3.9.21.tgz \
    && cd Python-3.9.21 \
    && ./configure --enable-optimizations\
    && make \
    && make install \
    && cd ../ && rm -rf Python-3.9.21 \
    && python3 -V \
    && ln -s /usr/local/bin/python3 /usr/local/bin/python

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN set -xe \
    && apt-get install -y python3-pip \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /usr/local/src/*