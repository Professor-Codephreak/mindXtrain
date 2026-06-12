# The Open-Source LLM Training Stack in Mid-2026: A Landscape Survey Anchored on mindXtrain

**Point of departure:** [github.com/professor-codephreak/mindXtrain](https://github.com/professor-codephreak/mindXtrain)

---

## TL;DR

- **mindXtrain** is an AMD × lablab.ai Developer Hackathon training project by Gregory Magnusson ("Professor Codephreak"), built around a **"60-second AOT autotune probe"** that runs on AMD Instinct MI300X silicon under an **"AOT-only" reproducibility discipline** — it sits at the intersection of three fast-maturing open-source ecosystems: training frameworks (Axolotl/Unsloth/torchtune/TRL), automated training+evaluation loops, and decentralized/training-as-a-service compute.
- The end-to-end open-source pipeline now composes cleanly and is overwhelmingly Apache-2.0/MIT licensed: **data curation (datatrove/Dolma/NeMo Curator) → training framework (Axolotl/Unsloth/torchtune/Megatron/DeepSpeed) → automated eval+HPO loop (lm-eval-harness/lighteval + Optuna/Ray Tune) → LoRA/adapter artifact (PEFT safetensors) → quantized export for GPU (AWQ/GPTQ/FP8) and CPU (GGUF k-quants) → optional TaaS exposure (Together/Modal/RunPod or decentralized Prime Intellect/Gensyn).**
- For an AOT-disciplined, model- and hardware-agnostic build like mindXtrain, the pragmatic 2026 stack is: **Hugging Face PEFT/TRL or Axolotl on ROCm for training, rsLoRA/DoRA rank-16 adapters, lm-evaluation-harness for the eval gate, torch.export+AOTInductor for compiled artifacts, and dual GGUF (CPU) + AWQ/FP8 (GPU) exports** — every piece has a permissive license and runs on NVIDIA CUDA, AMD ROCm, Intel, or Apple Silicon.

---

## 1. The mindXtrain Point of Departure

mindXtrain is the training/fine-tuning component of the broader **mindX ("augmentic intelligence orchestration")** ecosystem authored by Gregory L. Magnusson under the "Professor Codephreak" persona (part of the pythAI / automindx / aGLM / MASTERMIND / RAGE family of repos — see [rage.pythai.net](https://rage.pythai.net) and [mindx.pythai.net](https://mindx.pythai.net)). It was built for the **AMD Developer Hackathon hosted by lablab.ai**, which provides participants ~$100 AMD Developer Cloud credits and access to **AMD Instinct MI300X (192 GB HBM3) GPUs via ROCm**, with Qwen models as a featured partner family.

The defining architectural idea, confirmed verbatim from the author's own blog (rage.pythai.net), is a **"60-second AOT autotune probe — the layer that mindXtrain is built around"** that "runs on real MI300X silicon." The blog frames **"AOT-only" as a discipline**: a short ahead-of-time autotune/compile step runs first, its compiled/tuned artifacts are persisted, and those artifacts then flow into the rest of the pipeline so that training is reproducible across machines and across runs. This maps directly onto PyTorch's AOT machinery (AOTAutograd/Inductor caches, `torch.compiler.save_cache_artifacts`) and ROCm's offline GEMM tuning (TunableOp/hipBLASLt) — tune once ahead of time, then reuse deterministically rather than re-tuning kernels at runtime.

**Caveat:** The repository contents themselves (exact file structure, dependency pins, license file, and whether it targets Qwen3.5 vs Qwen3.6 specifically) could not be retrieved during research. The hackathon premise (Qwen3.5/3.6, AOT-only policy, AMD/ROCm) is consistent with everything found, and peer hackathon projects (e.g., a MedQA project that fine-tuned Qwen3-1.7B with LoRA) confirm the standard ROCm stack — **HuggingFace Transformers + PEFT + TRL + Accelerate** — runs on MI300X with no CUDA dependency and only three environment variables (`ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`).

Context on targets: **Qwen3.5** (released Feb 16, 2026) and **Qwen3.6-35B-A3B** (released ~April 2026, a 35B-total/3B-active MoE) are both **Apache 2.0** ([github.com/QwenLM/Qwen3.6](https://github.com/QwenLM/Qwen3.6)) and have Day-0 AMD MI300X/ROCm support via vLLM and SGLang.

---

## 2. Open-Source Training Frameworks

The single-/multi-GPU fine-tuning layer consolidated dramatically by 2026. Per a 2026 community comparison, GitHub stars and releases stood at roughly: **LLaMA-Factory 68.4K stars (v0.9.4, Dec '25), Unsloth 53.9K (Feb 2026 release), TRL 17.6K (v0.15.0, Mar '26), Axolotl 11.4K (v0.29.0, Feb '26)**. All four now support LoRA, QLoRA, full fine-tuning, DPO, GRPO, and vision models — the differentiation is workflow, not capability.

| Framework | License | Repo | Niche |
|---|---|---|---|
| Unsloth | Apache 2.0 | [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) | Single-GPU speed/VRAM leader (up to 2× faster, up to 70% less VRAM) |
| Axolotl | Apache 2.0 | [github.com/axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) | YAML config-driven production workhorse; FSDP/DeepSpeed; RLHF |
| torchtune | BSD-3 | [github.com/pytorch/torchtune](https://github.com/pytorch/torchtune) | PyTorch-native recipes; compile speedups; QAT; distillation |
| LLaMA-Factory | Apache 2.0 | [github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | GUI-first (LlamaBoard), 100+ model templates, Megatron backend |
| HF TRL/PEFT/Accelerate | Apache 2.0 | [github.com/huggingface/trl](https://github.com/huggingface/trl), [github.com/huggingface/peft](https://github.com/huggingface/peft), [github.com/huggingface/accelerate](https://github.com/huggingface/accelerate) | The institutional substrate: SFT/DPO/GRPO/PPO + all LoRA variants |

**Pretraining/large-scale:** [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (tensor/sequence/pipeline/expert parallelism), [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) (ZeRO 1/2/3, MoE, 3D parallelism), [NVIDIA NeMo](https://github.com/NVIDIA/NeMo), [Colossal-AI](https://github.com/hpcaitech/ColossalAI), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) (EleutherAI), [LLM Foundry](https://github.com/mosaicml/llm-foundry) / [Composer](https://github.com/mosaicml/composer) (MosaicML), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [Lightning](https://github.com/Lightning-AI/pytorch-lightning), and HF [Nanotron](https://github.com/huggingface/nanotron) (used for the FineWeb ablations).

**Hardware support is genuinely multi-vendor:** NVIDIA CUDA everywhere; AMD ROCm mature (the whole HF stack runs on MI300X unchanged); Intel via PyTorch XPU/IPEX; Apple Silicon via [MLX](https://github.com/ml-explore/mlx); CPU-only training feasible but slow (§6).

---

## 3. Automated / Autonomous Training Pipelines

- **HPO:** [Optuna](https://github.com/optuna/optuna), [Ray Tune](https://github.com/ray-project/ray), and Weights & Biases Sweeps are the dominant open-source hyperparameter optimizers.
- **Automated evaluation loops:** EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (MIT) is the de facto standard and was the backend for the (now-retired, March 2025) Open LLM Leaderboard; [HELM](https://github.com/stanford-crfm/helm) (Stanford CRFM, Apache 2.0) for multi-metric holistic eval; [OpenCompass](https://github.com/open-compass/opencompass) (Apache 2.0, 100+ datasets, strong CJK); [lighteval](https://github.com/huggingface/lighteval) (HF, MIT, integrates with Accelerate/Nanotron/vLLM); plus newer entrants [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) (UK AISI) and [DeepEval](https://github.com/confident-ai/deepeval). As of Dec 2025 lm-eval-harness refactored its CLI into subcommands and made transformers/torch optional installs.
- **Synthetic data / RLAIF:** [Distilabel](https://github.com/argilla-io/distilabel) (Argilla/HF, Apache 2.0) is the leading programmatic pipeline — typed steps, vLLM/HF/OpenAI backends, prepackaged Self-Instruct, Evol-Instruct, UltraFeedback, and [Magpie](https://github.com/magpie-align/magpie) tasks. Per the Magpie paper ([arXiv:2406.08464](https://arxiv.org/abs/2406.08464), ICLR 2025), models SFT'd with Magpie data performed comparably to official Llama-3-8B-Instruct despite the latter's 10M-datapoint pipeline — Magpie generated 4M instructions, filtered to 300K. Magpie-Ultra used Llama-3.1-405B. Cosmopedia-style synthetic textbooks and Nemotron-4 pipelines round out pretraining-scale synthesis.
- **MLOps orchestration:** [MLflow](https://github.com/mlflow/mlflow) (experiment tracking + registry), [ClearML](https://github.com/clearml/clearml), [ZenML](https://github.com/zenml-io/zenml), [Kubeflow](https://github.com/kubeflow/kubeflow), [Flyte](https://github.com/flyteorg/flyte), [Metaflow](https://github.com/Netflix/metaflow), and [SkyPilot](https://github.com/skypilot-org/skypilot) (multi-cloud/K8s job orchestration — commonly paired with MLflow for LLM fine-tuning). All Apache 2.0.

---

## 4. Training-as-a-Service: Centralized and Decentralized

**Centralized, OSS-friendly:**
- [Together AI](https://www.together.ai) — serverless + dedicated + fine-tuning + GPU clusters. H100 clusters quoted between $2.25–$5.49/hr depending on commitment and source/date; Batch API up to 50% off.
- [Modal](https://modal.com) — Python-native serverless, sub-5s cold starts; H100 ≈ $3.95/hr equivalent at per-second billing.
- [RunPod](https://www.runpod.io) — $0.39–$2.89/hr by GPU; per-second billing.
- [Replicate](https://replicate.com), [Hugging Face AutoTrain](https://github.com/huggingface/autotrain-advanced), [Predibase](https://predibase.com) / [LoRAX](https://github.com/predibase/lorax), [OpenPipe](https://openpipe.ai), [Lambda](https://lambdalabs.com), [Vast.ai](https://vast.ai) (bid marketplace, cheapest).
- A typical 4-hour Llama-2 fine-tune runs ~$8–10 on RunPod, $12–16 on Modal, $14–18 on Replicate.

**Decentralized:**
- [Prime Intellect](https://www.primeintellect.ai) is the clear frontrunner — **INTELLECT-1** (10B, trained on FineWeb-Edu via [OpenDiLoCo](https://github.com/PrimeIntellect-ai/OpenDiLoCo); int8 pseudo-gradient quantization for a ~400× bandwidth reduction at 83–98% compute utilization across up to 14 nodes on 3 continents over 1T tokens; [arXiv:2412.01152](https://arxiv.org/abs/2412.01152)), **INTELLECT-2** (32B, first globally-distributed RL run, built on [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) with TOPLOC verifiable inference and [shardcast](https://github.com/PrimeIntellect-ai/shardcast) weight broadcast; [arXiv:2505.07291](https://arxiv.org/abs/2505.07291); both Apache 2.0; model: [huggingface.co/PrimeIntellect/INTELLECT-2](https://huggingface.co/PrimeIntellect/INTELLECT-2)), and a teased **INTELLECT-3** (100B+ MoE).
- [Gensyn](https://www.gensyn.ai) — RL Swarm on testnet ([github.com/gensyn-ai/rl-swarm](https://github.com/gensyn-ai/rl-swarm)), uses Hivemind gossip; execution/communication/verification architecture; AXL coordination layer.
- [Nous Research DisTrO](https://github.com/NousResearch/DisTrO), [Petals](https://github.com/bigscience-workshop/petals) (collaborative 100B+ inference/fine-tuning), [Hivemind](https://github.com/learning-at-home/hivemind) (volunteer DiLoCo training), [Bittensor](https://github.com/opentensor/bittensor) training subnets, and compute markets [Akash](https://akash.network) and [io.net](https://io.net).

The economics: frontier centralized runs now cost billions, driving the decentralization thesis; the open verification problem and async RL (well-suited to heterogeneous swarms) are the key 2025–2026 advances. Caveat: decentralized RL gains have so far been concentrated in the training-data domains (math/code), with more modest broad-benchmark transfer.

### 4a. Verification Software (verifiable training & inference) — links and source

**Activation-hash verification (verifiable inference):**
- **TOPLOC** (Prime Intellect) — locality-sensitive hashing of intermediate activations; detects unauthorized modifications to models, prompts, or compute precision with 100% empirical accuracy; validation up to 100× faster than original inference; 258 bytes of storage per 32 tokens (1000× memory reduction vs raw embeddings). Used to verify all decentralized rollout workers in INTELLECT-2.
  - Code: [github.com/PrimeIntellect-ai/toploc](https://github.com/PrimeIntellect-ai/toploc)
  - Experiments: [github.com/PrimeIntellect-ai/toploc-experiments](https://github.com/PrimeIntellect-ai/toploc-experiments)
  - REST validator server: [github.com/PrimeIntellect-ai/toploc-validator](https://github.com/PrimeIntellect-ai/toploc-validator)
  - Paper: [arXiv:2501.16007](https://arxiv.org/abs/2501.16007)

**Refereed delegation (verifiable *training*):**
- **Verde + RepOps** (Gensyn) — dispute-resolution protocol that pinpoints the first disagreeing training step/operator, built on Reproducible Operators (RepOps), a library enforcing bitwise-reproducible ML ops across hardware (fixed FP operation ordering). Unlike TOPLOC, extends to training and fine-tuning. In production on the Gensyn testnet.
  - Demo code: [github.com/gensyn-ai/repops-demo](https://github.com/gensyn-ai/repops-demo)
  - Paper: [arXiv:2502.19405](https://arxiv.org/abs/2502.19405)
  - Blog: [blog.gensyn.ai/verde-verification-system-in-production](https://blog.gensyn.ai/verde-verification-system-in-production/)
- **RepDL** (Microsoft) — bitwise-reproducible deep learning ops, cited by Verde: [github.com/microsoft/RepDL](https://github.com/microsoft/RepDL)

**Optimistic / fraud-proof verification:**
- **opML** (ORA) — off-chain ML execution with an on-chain interactive dispute engine (bisection to a single MIPS instruction); runs 7B LLaMA on a common PC without GPU; targets training/fine-tuning as well as inference; deterministic execution via fixed-point arithmetic and software FP libraries.
  - Code: [github.com/ora-io/opml](https://github.com/ora-io/opml)
  - Paper: [arXiv:2401.17555](https://arxiv.org/abs/2401.17555)
- **zk-OPML** — hybrid using SP1 zkVM to optimize opML disputes: [github.com/Vid201/zk-OPML](https://github.com/Vid201/zk-OPML)

**zkML (zero-knowledge proofs of model execution):**
- **EZKL** (Zkonduit) — converts ONNX graphs into ZK-SNARK circuits (Halo2) with on-chain verifiers; Python/JS/CLI bindings; audited by Trail of Bits: [github.com/zkonduit/ezkl](https://github.com/zkonduit/ezkl)
- **zkml** (Daniel Kang) — ZK proofs of ML execution scaled to ImageNet-class models: [github.com/ddkang/zkml](https://github.com/ddkang/zkml)
- **awesome-zkml** — curated index of the zkML space: [github.com/worldcoin/awesome-zkml](https://github.com/worldcoin/awesome-zkml)

**Supporting infra (where verification plugs in):**
- [github.com/PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) — async decentralized RL using TOPLOC
- [github.com/PrimeIntellect-ai/shardcast](https://github.com/PrimeIntellect-ai/shardcast) — HTTP tree-topology weight broadcast
- [github.com/PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers) — RL environment verifier library
- [github.com/learning-at-home/hivemind](https://github.com/learning-at-home/hivemind) — volunteer training substrate underlying Gensyn RL Swarm

---

## 5. Data Curation

- **Pipelines/toolkits:** HF [datatrove](https://github.com/huggingface/datatrove) (Apache 2.0, ran the entire FineWeb pipeline), AI2 [Dolma toolkit](https://github.com/allenai/dolma) (Apache 2.0, 3T-token corpus, OLMo project), [NVIDIA NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) (Apache 2.0), RedPajama/[SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) pipelines.
- **Methodology (FineWeb/FineWeb-Edu, [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)):** URL filtering → Trafilatura extraction → FastText language ID → MassiveText + C4 + custom quality filters → **MinHash dedup** → PII reformatting; FineWeb-Edu adds a classifier-based educational-quality filter. FineWeb is ~15T tokens (ODC-By 1.0); FineWeb-Edu (1.3T) matches C4/Dolma MMLU performance with ~10× fewer tokens — the highest-leverage single intervention. Datasets: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
- **Techniques:** MinHash + exact dedup; classifier-based and perplexity quality filtering; **benchmark decontamination**; PII removal; tokenizer/chat-template considerations; instruction formats (Alpaca, ShareGPT); dataset mixing/ablation via small proxy models on lighteval. Provenance/licensing matters: prefer ODC-By/permissive corpora and document mix ratios.

---

## 6. LoRA/Adapter Ecosystem, Weights, Formats, and Quantization

**Adapters (all in HF [PEFT](https://github.com/huggingface/peft), Apache 2.0):**
- Standard **LoRA**; **QLoRA** (4-bit NF4 frozen base + BF16 adapters, ~4× memory cut, 8B fine-tune in <8 GB VRAM); **DoRA** (weight-decomposed, +1–4.4% accuracy, no inference overhead, `use_dora=True`); **rsLoRA** (scales α/√r — better at high ranks); **LoRA+**; **PiSSA** ([arXiv:2404.02948](https://arxiv.org/abs/2404.02948), SVD principal-component init, faster convergence, lower quantization error).
- 2026 practical guidance: **start at rank 16 with DoRA and `target_modules="all-linear"`, α = rank (or 2× rank), enable rsLoRA only when pushing high ranks.** Recent 2026 work shows a well-tuned learning rate often closes most of the gap between vanilla LoRA and its variants.
- **Multi-LoRA serving:** [LoRAX](https://github.com/predibase/lorax) (Apache 2.0), [vLLM multi-LoRA](https://github.com/vllm-project/vllm), [S-LoRA](https://github.com/S-LoRA/S-LoRA) serve thousands of adapters against one base.
- **Merging/composition:** [mergekit](https://github.com/arcee-ai/mergekit) + PEFT implement **TIES** (trim/elect-sign/merge), **DARE** (drop-and-rescale), task arithmetic, SLERP, DELLA; for LoRA, density ~0.5 for TIES is a good default. Note: joint data-mix training often still beats TIES/DARE for multi-skill composition.
- **Artifacts:** LoRA adapters are stored as [safetensors](https://github.com/huggingface/safetensors) on the HF Hub with adapter_config.json conventions.

**Formats & quantization:**
- **GPU:** safetensors (training/transfer), [AWQ](https://github.com/mit-han-lab/llm-awq) (4-bit, activation-aware, ~95% quality retention, vLLM-friendly), [GPTQ](https://github.com/ModelCloud/GPTQModel) (4-bit, CUDA/ExLlama), **FP8** (near-baseline quality, Hopper/Blackwell), NF4/INT4 via [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (QLoRA), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) engines, PyTorch [TorchAO](https://github.com/pytorch/ao).
- **CPU:** **GGUF** ([llama.cpp](https://github.com/ggml-org/llama.cpp)/[Ollama](https://github.com/ollama/ollama)) with k-quants (Q4_K_M is the quality/size sweet spot, ~92% retention; Q5_K_M/Q6_K near-BF16) and IQ-quants; AVX-512/AMX acceleration; CPU+GPU hybrid layer offload.
- **Apple Silicon:** [MLX](https://github.com/ml-explore/mlx) format; [mlx-lm](https://github.com/ml-explore/mlx-lm) supports LoRA/QLoRA/DoRA/full fine-tuning natively (the only on-device training path on Macs — llama.cpp is inference-only), exporting to HF/GGUF. Unified memory lets a 32 GB Mac train models a 24 GB GPU cannot.
- **Conversion pipeline:** trained checkpoint (safetensors) → fuse LoRA → `convert_hf_to_gguf.py` (CPU/GGUF) and/or AWQ/GPTQ/FP8 quantize (GPU) → optionally `torch.export` + AOTInductor to a `.pt2` shared library for Python-free C++ deployment.
- **CPU-only training feasibility:** possible (QLoRA on CPU; llama.cpp is historically inference-focused) but 1–2 orders of magnitude slower than GPU; practical mainly for tiny models or last-resort environments.

---

## 7. AOT Compilation, Reproducibility, Licensing

**AOTInductor** (PyTorch, Beta) compiles a `torch.export`-ed graph ahead of time via `torch._inductor.aoti_compile_and_package()` into a `.pt2` artifact (shared lib + optional CUDA cubins) loadable in Python or C++ with no JIT warmup at deployment — the same discipline mindXtrain applies for reproducible MI300X training. For CPU inference, `TORCHINDUCTOR_FREEZING=1` is recommended; Intel GPU is supported. Reproducibility caveats: AOT/export requires static control flow (use `torch.cond`), and compiled artifacts are sensitive to libtorch version and device/compute-capability mismatches. Licensing across the surveyed stack is overwhelmingly **Apache 2.0 / MIT / BSD** — the AMD hackathon itself requires open-source submissions with a detectable license.

---

## The End-to-End Pipeline

1. **Curate data** with datatrove or NeMo Curator: extract → language/quality filter → MinHash dedup → decontaminate against your eval set → PII scrub → format to chat template. Augment with Distilabel/Magpie synthetic data; validate mix ratios with small-proxy ablations on lighteval.
2. **Train** with Axolotl/Unsloth/torchtune (single-node) or Megatron/DeepSpeed/NeMo (multi-node), using FSDP or ZeRO-3 for sharding and tensor/pipeline/expert parallelism at scale. On AMD, run the HF stack unchanged on ROCm. Produce LoRA/DoRA rank-16 adapters (or full FT if budget allows).
3. **Automate the loop:** Optuna/Ray Tune for HPO, lm-evaluation-harness/lighteval as the quality gate, W&B/MLflow for tracking, SkyPilot/ZenML/Kubeflow for orchestration and CI-CD of model versions.
4. **Produce artifacts:** safetensors adapters on the Hub; optionally merge with mergekit (TIES/DARE).
5. **Export for deployment:** GGUF k-quants for CPU/edge (llama.cpp/Ollama), AWQ/FP8 for GPU serving (vLLM/SGLang), and `torch.export`+AOTInductor `.pt2` for compiled, reproducible artifacts.
6. **Expose as a service:** self-host multi-LoRA via LoRAX/vLLM, offer training jobs through Modal/RunPod/Together, or contribute to/borrow from decentralized networks (Prime Intellect prime-rl, Gensyn RL Swarm) — with TOPLOC or Verde-style verification of outsourced work.

---

## Recommendations

- **For the mindXtrain trajectory specifically:** keep the AOT-only discipline but formalize it on `torch.export` + AOTInductor with `save_cache_artifacts` and ROCm TunableOp/hipBLASLt offline tuning so the "60-second probe" output is a versioned, checked-in artifact. Pin libtorch/ROCm versions to avoid documented AOTInductor load-time mismatch failures. Stage next: (a) wrap training in Axolotl YAML or HF PEFT/TRL for reproducibility on ROCm; (b) add lm-evaluation-harness as a hard CI gate; (c) emit dual GGUF + AWQ/FP8 exports so artifacts are both CPU- and GPU-deployable; (d) publish adapters as safetensors with a clear Apache-2.0 license.
- **If you have one GPU:** Unsloth. **Multi-GPU/production:** Axolotl + DeepSpeed/FSDP. **PyTorch-native control or QAT:** torchtune. **RLHF/GRPO:** TRL (optionally with Unsloth kernels). **Starting out:** LLaMA-Factory GUI.
- **Thresholds that change the plan:** if a model exceeds single-node VRAM, move to Megatron/DeepSpeed 3D parallelism or a decentralized run; if eval scores regress on general tasks after fine-tuning, you've overfit — cut epochs/rank or rebalance the data mix; if broad-benchmark transfer (not just in-domain) matters, prefer centralized RL over current decentralized RL, whose gains remain domain-concentrated.

---

## Caveats

- **mindXtrain repo internals are unverified.** The README/file structure/dependency list/exact Qwen version/license could not be retrieved during research; only the AOT-probe-on-MI300X purpose is confirmed (author blog). Verify the repo directly.
- Several benchmark figures (framework speed deltas, quantization quality-retention percentages, TaaS hourly prices) come from vendor blogs and community comparisons, not peer-reviewed sources, and shift rapidly; treat them as directional. Together AI's cluster pricing in particular is quoted inconsistently across sources ($2.25–$5.49/hr H100 depending on commitment and date).
- Quantization quality retention is task-dependent — INT4 degrades most on math/code/reasoning; FP8 is closest to baseline.
- Decentralized training is real but early; efficiency losses vs co-located clusters persist, and verifiable-work mechanisms (TOPLOC, Verde, opML) are still maturing.

---

*Compiled June 2026. Sources include Prime Intellect, Gensyn, ORA, EleutherAI, Hugging Face, AMD ROCm blogs, arXiv (2501.16007, 2502.19405, 2401.17555, 2505.07291, 2412.01152, 2406.17557, 2406.08464, 2404.02948), and 2026 community framework comparisons.*
