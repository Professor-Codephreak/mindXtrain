# Ops

Container, compose, Kubernetes, vmm, and Gensyn manifests for mindxtrain. Per
mindxtrain2.md §Part 4 (`ops/`).

## Layout

```
ops/
├── containerfiles/
│   ├── containerfile_train     # FROM rocm/primus:v26.2          (training image)
│   ├── containerfile_serve     # FROM rocm/vllm-dev:rocm7.2.1    (inference image)
│   └── digest.lock             # SHA256 pins, populated post-pull
├── compose/
│   └── compose_dev.yaml        # full stack: vLLM + operator FastAPI for the live demo
├── k8s/
│   └── train_job.yaml          # single-MI300X training Job
├── vmm/                        # OpenBSD vmm vm definitions (post-hackathon)
└── gensyn/                     # Gensyn distributed-training configs (post-hackathon)
```

## Build the train image (on the MI300X droplet)

```bash
podman build -f ops/containerfiles/containerfile_train -t mindxtrain/train:latest .
podman inspect --format '{{index .RepoDigests 0}}' mindxtrain/train:latest \
  | tee -a ops/containerfiles/digest.lock
```

## Build the serve image

```bash
podman build -f ops/containerfiles/containerfile_serve -t mindxtrain/serve:latest .
```

## Run the demo stack

```bash
podman-compose -f ops/compose/compose_dev.yaml up -d
# vLLM-ROCm at :8000, mindxtrain operator FastAPI at :8080
```

## Submit the K8s training job

```bash
kubectl create configmap mindxtrain-run-config --from-file=run.yaml=run.yaml
kubectl apply -f ops/k8s/train_job.yaml
kubectl logs -f job/mindxtrain-job
```

The Job spec assumes a node labeled `accelerator: mi300x` and the AMD ROCm
device plugin (`amd.com/gpu` resource).
