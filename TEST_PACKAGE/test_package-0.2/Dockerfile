FROM ubuntu:latest

RUN mkdir tamle

# COPY ././dist/.-0.2.tar.gz ./tamle/.-0.2.tar.gz

COPY ./deploy/mle-dev.yml ./tamle/mle-dev.yml

COPY ./build ./tamle/build
# COPY ./datasets ./tamle/datasets
COPY ./data ./tamle/data
COPY ./deploy ./tamle/deploy
COPY ./dist ./tamle/dist
COPY ./docs ./tamle/docs
# COPY ./images ./tamle/images
# COPY ./logs ./tamle/logs
# COPY ./notebooks ./tamle/notebooks
COPY ./src ./tamle/src
COPY ./tests ./tamle/tests
COPY ./pyproject.toml ./tamle/pyproject.toml
COPY ./README.md ./tamle/README.md
COPY ./setup.py ./tamle/setup.py

ENV PATH="/root/miniconda3/bin:$PATH"

RUN cd tamle

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

WORKDIR /tamle

RUN conda env create -f mle-dev.yml

ENV PATH="../root/miniconda3/envs/tamle/bin:$PATH"

# RUN tar xsf ._ml-0.2.tar.gz

# RUN python setup.py install