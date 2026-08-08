# All-in-one inference container for Linux + NVIDIA: builds the PrismML fork's
# llama-server with CUDA and serves the models mounted at /app/models.
# Build context is the repo root (see docker/compose.linux.yaml), which is why
# the COPY paths below are relative to it.

# --- build stage ------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

ARG PRISM_SHA=4dd165625bb6c020285eec8b342af25cf60233dd
# sm_86 = RTX 30xx (Ampere consumer); override for other GPUs.
ARG CUDA_ARCHS=86

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git cmake build-essential libcurl4-openssl-dev \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/PrismML-Eng/llama.cpp /app/src/llama.cpp-prism \
 && git -C /app/src/llama.cpp-prism checkout "$PRISM_SHA"

WORKDIR /app/src/llama.cpp-prism
RUN cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
      -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build --target llama-server -j"$(nproc)"

# --- runtime stage ----------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# libcurl for llama-server, libgomp for ggml's OpenMP; sed is already in base.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libcurl4 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Kept at this exact path so start-server.sh finds the binary unmodified
# ($ROOT/src/llama.cpp-prism/build/bin/llama-server).
COPY --from=build /app/src/llama.cpp-prism/build/bin /app/src/llama.cpp-prism/build/bin
COPY start-server.sh models.ini.in /app/

ENTRYPOINT ["/app/start-server.sh"]
