# Maintained Prism runtime. Fetch the official PQ2_0 ternary artifact before
# upgrading an existing installation; keep legacy Q2_0 and the old image.
# Build: docker build -f docker/Dockerfile --build-arg QCM_REVISION=<sha> .
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5 AS build
ARG PRISM_SHA=922be44aa6ac81b46f092716351cddff1c1733a7
ARG CUDA_ARCHS=86
ARG BUILD_JOBS=4
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates cmake build-essential libssl-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app/src/llama.cpp-prism
RUN test "$PRISM_SHA" = 922be44aa6ac81b46f092716351cddff1c1733a7 \
 && git init . \
 && git remote add origin https://github.com/PrismML-Eng/llama.cpp \
 && git fetch --depth 1 origin "$PRISM_SHA" \
 && git checkout --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "$PRISM_SHA"
# Driver stub needed only for linking; the NVIDIA toolkit injects the real driver.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
      -DGGML_NATIVE=OFF -DLLAMA_OPENSSL=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF \
      -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
 && cmake --build build --target llama-server -j"$BUILD_JOBS" \
 && mkdir /runtime-bin \
 && cp -a build/bin/llama-server build/bin/*.so* /runtime-bin/

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:517da2300c184c9999ec203c2665244bdebd3578d12fcc7065e83667932643d9
ARG PRISM_SHA=922be44aa6ac81b46f092716351cddff1c1733a7
ARG QCM_REVISION=unknown
LABEL org.opencontainers.image.revision="$QCM_REVISION" \
      io.quickclevermodel.prism.repository="https://github.com/PrismML-Eng/llama.cpp" \
      io.quickclevermodel.prism.sha="$PRISM_SHA" \
      io.quickclevermodel.prism.commit-date="2026-09-21T16:31:15Z" \
      io.quickclevermodel.llama-upstream="5ea87ddad22541a37053c7ba92b02ec1923617c6" \
      io.quickclevermodel.cuda="12.4.1" \
      io.quickclevermodel.validation="native-tested-cuda-pending"
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libssl3 libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /runtime-bin/ /app/src/llama.cpp-prism/build/bin/
COPY start-server.sh models.ini.in bonsai-chat-template.jinja qwen3.5-chat-template.jinja version.sh models.lock.tsv runtime-versions.env /app/
RUN printf '%s\n' "$QCM_REVISION" > /app/build-revision \
 && sed -i "s/^PRISM_SHA=.*/PRISM_SHA=$PRISM_SHA/;s/^PRISM_COMMIT_DATE=.*/PRISM_COMMIT_DATE=2026-09-21T16:31:15Z/;s/^LLAMA_UPSTREAM_SHA=.*/LLAMA_UPSTREAM_SHA=5ea87ddad22541a37053c7ba92b02ec1923617c6/" /app/runtime-versions.env
ENTRYPOINT ["/app/start-server.sh"]
