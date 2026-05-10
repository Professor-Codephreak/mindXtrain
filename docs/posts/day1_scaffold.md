# Day 1 Build-in-Public post — May 4 2026

**Theme:** Why MI300X for sovereign cognition, and the framework I'm shipping for the hackathon.

## X thread (≤280 chars per tweet)

```
1/ Day 1 of the AMD × @lablabai Developer Hackathon. Shipping mindXtrain — the
first one-command Qwen3 fine-tuner native to MI300X. 192 GB HBM3 means BF16 8B
fits with headroom and 32B fits at all. H100 80 GB just OOMs.

#AMDDevHackathon @AIatAMD
```

```
2/ The differentiator: a 60-second AOT autotune probe. Before each run, a
micro-benchmark picks Composable Kernel vs Triton attention, hipBLASLt
heuristics, and RCCL config. Static plan, no JIT autotune in production.

Nothing else in the @huggingface ecosystem ships this.
```

```
3/ Day-1 deliverables green:

✓ uv workspace, 64 Python files, 12 Qwen3 recipes
✓ 27 tests passing (CPU-only)
✓ Pydantic schema enforces MI300X invariants (xGMI gotcha, MoE gate frozen)
✓ Foundry contracts for write-once provenance anchoring

GitHub: <repo URL when public>
```

```
4/ Hero workload target on Qwen3-8B / 1× MI300X / BF16:
- >15 000 tok/s
- MFU >40%
- time-to-loss-1.5 <90 min
- total cost <$3 (vs ~$32 on 2× H100)

Cost slide writes itself. @Alibaba_Qwen
```

## LinkedIn post (long-form)

```
Day 1 of the AMD × lablab.ai Developer Hackathon — shipping mindXtrain, a
one-command Qwen3 fine-tuner native to AMD MI300X.

Why MI300X for this specific work: 192 GB HBM3 means a Qwen3-8B BF16 LoRA
job at bs=8 seq=4096 fits with massive headroom on a single GPU. The same
workload on H100 80 GB requires either quantization or splitting across
two cards. At AMD Developer Cloud's $1.99/hr versus H100 list of $4/hr,
the same 1B-token training run lands at $3 versus $32. 4× cheaper.

The differentiator isn't the model or the dataset — it's the 60-second
AOT autotune probe that runs before each training job. It picks
Composable Kernel vs Triton SDPA based on measured timings, picks the
hipBLASLt heuristic for the run's shape, and locks in the NCCL channel
count. Static plan, written to disk, consumed at training start. No JIT
autotune in production — full reproducibility.

Day 1 status:
✓ uv workspace with 3 packages (automindXtrain → mindXtrain → custmodel)
✓ 64 Python files, 12 Qwen3 recipes, 27 tests passing on CPU
✓ Pydantic schema enforces MI300X invariants (1- or 8-GPU FSDP, MoE
  gate frozen, AOT-only autotune policy)
✓ Foundry contracts for write-once provenance anchoring (no proxy,
  no admin keys)
✓ Full doc hub under docs/

Heading to the AMD Developer Cloud now to provision the MI300X droplet
for Day 2's autotune probes. The hard part — making Composable Kernel
and Triton race head-to-head and capturing the wow-moment for the demo
video — starts tomorrow.

#AMDDevHackathon
```

## Asset checklist

- [ ] Screenshot of `uv run mindxtrain init --list` output
- [ ] Screenshot of `uv run pytest -q` showing 27 passed
- [ ] Screenshot of the `mindxtrain.tuned.yaml` from a dry-run bench
- [ ] Repo URL once public
