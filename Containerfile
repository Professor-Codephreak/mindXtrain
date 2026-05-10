# Default-discovered Containerfile for `podman build .` at the repo root.
# Canonical content also lives at ops/containerfiles/containerfile_train.
#
# Pin: rocm/primus:v26.2 ships ROCm 7.2.1, PyTorch 2.9.1, Primus-Turbo with FlashAttention,
# AITER, and hipBLASLt. SHA256 digest in ops/containerfiles/digest.lock.
FROM docker.io/rocm/primus:v26.2

ENV PYTORCH_ROCM_ARCH=gfx942 \
    HSA_NO_SCRATCH_RECLAIM=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    GPU_MAX_HW_QUEUES=1 \
    NVTE_CK_USES_BWD_V3=1 \
    NVTE_CK_IS_V3_ATOMIC_FP32=1 \
    PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1 \
    NCCL_MIN_NCHANNELS=112

WORKDIR /workspace/mindxtrain

RUN pip install --no-cache-dir uv

COPY . /workspace/mindxtrain
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "mindxtrain"]
CMD ["--help"]
