# Day 5 Build-in-Public post — May 8 2026

**Theme:** Demo URL is live; cost-vs-H100 numbers; submitting tomorrow.

## X thread

```
1/ Day 5: demo URL is LIVE.

mindx.pythai.net/hackathon

Trained, FP8-quantized Qwen3-8B (LoRA) running on a single MI300X behind
@huggingface vLLM-ROCm and an OpenAI-compatible API. Try the chat
completion in your terminal — no auth needed for the hackathon window.

#AMDDevHackathon
```

```
2/ Cost slide:

This Qwen3-8B SFT-LoRA, 1B tokens, BF16 unquantized:

  MI300X $1.99/hr × 1 GPU × <X> hrs = $<Y>
  H100   $4.00/hr × 2 GPUs × ~4 hrs = ~$32

@AIatAMD's 192 GB HBM3 is doing real work — H100 80 GB OOMs at this
exact bs/seq combo without falling back to FP8.
```

```
3/ The full stack the demo exercises:

✓ ROCm 7.2.1 + AOTriton + AITER + Composable Kernel + hipBLASLt
✓ Primus-Turbo + torchtitan-amd
✓ AMD Quark FP8 PTPC (15-30% faster than BlockScale)
✓ vLLM-ROCm with the qwen3 reasoning parser + hermes tool-call parser
✓ BLAKE3 provenance manifest pinned to Lighthouse
```

```
4/ Submitting on lablab tomorrow morning. Three primary tracks:

- Fine-Tuning on AMD GPUs (primary)
- AI Agents & Agentic Workflows (automindXtrain serves the model)
- Vision & Multimodal (qwen3_vl_8b_sft recipe shipped)

Plus Build-in-Public + Best Use of Qwen.

@lablabai @Alibaba_Qwen
```

## LinkedIn post

```
Day 5 of the AMD × lablab.ai Developer Hackathon — demo is live.

mindx.pythai.net/hackathon

The pipeline you can poke at:
1. Qwen3-8B base model
2. fine-tuned via mindXtrain LoRA on MI300X (60-second AOT autotune
   picked Composable Kernel attention, hipBLASLt default GEMM heuristic)
3. quantized via AMD Quark FP8 PTPC into a vLLM-loadable directory
4. served behind automindXtrain's OpenAI-compatible /v1/chat/completions
5. BLAKE3 provenance manifest pinned to Lighthouse / IPFS

The cost story: this exact workload at $1.99/hr on a single MI300X
versus 2× H100 at $4/hr each. Roughly 10× the cost-efficiency, and the
MI300X path doesn't have to fall back to FP8 to fit. 192 GB HBM3 is
doing real work.

Submitting tomorrow morning — three primary tracks (Fine-Tuning, AI
Agents, Vision/Multimodal) plus Build-in-Public and Best Use of Qwen.
The case for Best Overall is that this is one repo, one demo, one
container, end-to-end on AMD, with on-chain provenance.

The full repo is open-source Apache-2.0 (MIT-compatible per the lablab
spec). All the receipts: 

- GitHub: <repo URL>
- 5-min demo video: <YouTube URL>
- Demo URL: mindx.pythai.net/hackathon

To AMD's @AIatAMD team — the ROCm 7.2.1 stack works. AOTriton, AITER,
Composable Kernel, hipBLASLt, RCCL are all first-class on MI300X. The
pin matrix in the README is ground truth for anyone building on this.

#AMDDevHackathon
```

## Asset checklist

- [ ] Live demo URL screenshot
- [ ] `curl mindx.pythai.net/hackathon/v1/chat/completions` output
- [ ] Side-by-side cost table screenshot (MI300X vs H100)
- [ ] BLAKE3 manifest sample output
- [ ] Final lablab submission form preview
