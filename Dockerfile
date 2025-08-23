FROM continuumio/anaconda3:latest
SHELL ["bash", "-lc"]

ARG DEFAULT_BUILD_TYPE=Release
ARG DEFAULT_PARALLEL=4
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=${DEFAULT_PARALLEL} \
    SKBUILD_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${DEFAULT_BUILD_TYPE}"

# Build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/*

RUN conda create -y -n urbanflow python=3.11 pip

ENV CONDA_DEFAULT_ENV=urbanflow
ENV PATH=/opt/conda/envs/urbanflow/bin:$PATH

RUN echo '. /opt/conda/etc/profile.d/conda.sh' >  /etc/profile.d/activate_urbanflow.sh && \
    echo 'conda activate urbanflow'           >> /etc/profile.d/activate_urbanflow.sh && \
    chmod +x /etc/profile.d/activate_urbanflow.sh

RUN python -m pip install --upgrade pip

COPY . .
RUN /opt/conda/envs/urbanflow/bin/pip install -e .[dev]

WORKDIR /workspace
