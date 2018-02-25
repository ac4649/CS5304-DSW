FROM continuumio/miniconda3
# Install a simple text editor
RUN apt-get update \
    && apt-get install -y vim
# Install jupyter notebook
RUN conda install jupyter
# Install pytoch (Requires CUDA 8.0)
RUN conda install pytorch torchvision -c pytorch
# Install pytorch (cpu)
# RUN conda install pytorch-cpu torchvision -c pytorch
# Install useful tools for interactive debugging
RUN pip install ipython ipdb
# Install additional datascience tools
RUN pip install matplotlib pandas scipy scikit-learn
RUN useradd -ms /bin/bash nyc
USER nyc
WORKDIR /code
