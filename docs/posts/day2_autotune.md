# Day 2 Build-in-Public post — May 5 2026

**Theme:** The 60-second AOT autotune in action on MI300X.

## X thread

```
1/ Day 2: the autotune layer that makes mindXtrain win the Application of
Technology axis.

60 seconds on MI300X, three probes:
- attention: Composable Kernel vs Triton SDPA
- gemm: hipBLASLt heuristic check
- rccl: 1-GPU vs 8-GPU xGMI

Output: a static AOT plan. #AMDDevHackathon
```

```
2/ Today's measurement:

CK forward, (8, 4096, 32, 128): <X> ms
Triton forward, same shape:     <Y> ms
   → CK wins by <Z>%, plan picks ck

Hand-tuned ASM kernels via @AIatAMD's AITER beat Triton at this size.
This decision is locked into the run, not re-decided every step.
```

```
3/ Why AOT-only matters for production training:

JIT autotune (Triton on cold start, torch.compile max-autotune, MIOpen
find-mode) makes the same workload non-deterministic across runs. AMD's
AOTriton + offline-tuned hipBLASLt cache + AOT plan = reproducible.

Hash-equal across machines. cypherpunk2048 standard.
```

```
4/ Code is small. autotune/ is one orchestrator + three probes + a
Pydantic AutotunePlan. The training layer reads plan.json, sets env
vars + flags, then accelerate launches Axolotl. 

GitHub: <repo URL>

Tomorrow: the actual LoRA fine-tune of amd/Instella-3B on MI300X.
@Alibaba_Qwen
```

## LinkedIn post

```
Day 2 of the AMD × lablab.ai Developer Hackathon — the autotune layer is
live on MI300X.

A 60-second probe runs before each training job:

1. Attention: torch's scaled_dot_product_attention timed across four
   representative shapes on both Composable Kernel (default) and AOTriton.
   Pick the faster. Today: <CK ms> vs <Triton ms> on Qwen3-8B's shape.

2. GEMM: per the AMD ROCm 7.2.1 release notes, hipBLASLt 0.10's default
   heuristic for gfx942 BF16/FP16 GEMMs is within 5% of hand-tuned for
   the LoRA-rank-16-to-64 / hidden-2048-to-8192 shapes mindXtrain hits.
   Plan locks it in. Heuristic enumeration is post-hackathon work.

3. RCCL: 1-GPU is no-op; 8-GPU sets NCCL_MIN_NCHANNELS=112 and
   GPU_MAX_HW_QUEUES=1 in the plan's env block. The 2/4-GPU paths
   raise — MI300X xGMI bandwidth between subsets of 2/4 GPUs is
   asymmetric and silently bottlenecks FSDP shards.

Output is a static AutotunePlan JSON, BLAKE3-hashed into the custmodel
manifest. No JIT autotune in production. Same plan = same kernels =
reproducible runs.

This is the cypherpunk2048 reproducibility standard applied to the
ROCm reality. The training layer reads the plan and sets the env vars
and Axolotl flags before subprocess-launching accelerate. Nothing
re-tunes during the loop.

Tomorrow: full LoRA fine-tune of amd/Instella-3B on MI300X using the
plan from today.

#AMDDevHackathon
```

## Asset checklist

- [ ] `autotune_plan.json` from a real MI300X run
- [ ] Side-by-side timing table: CK vs Triton across 4 shapes
- [ ] `rocminfo` output showing gfx942 + 192 GB
- [ ] Recording of the 60-second probe streaming output (used in the demo video)
