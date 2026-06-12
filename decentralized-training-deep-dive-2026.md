# Decentralized Training: The Complete Landscape (Mid-2026)

**A deep-dive companion to the mindXtrain training-stack survey** · Compiled June 2026

---

## TL;DR

- **Decentralized training crossed its credibility threshold in 2025–2026.** Three landmark proofs: Templar's **Covenant-72B** (March 10, 2026 — 72B params, ~1.1T tokens, 70+ permissionless nodes over commodity internet, MMLU 67.1, ~LLaMA-2-70B class), Nous **Psyche/Consilience-40B** (largest internet pre-training run by parameter×token scale, coordinated on Solana), and Pluralis **Node0-7.5B** (first public *model-parallel* internet pretraining: 1,642 GPUs, 300+ participants, 198 cities, 36B tokens in 3 weeks).
- **The algorithmic unlock is communication compression**: DiLoCo-family infrequent synchronization (~500× less communication), Streaming DiLoCo (two orders of magnitude bandwidth reduction), SparseLoCo (top-k sparsification + 2-bit quantization to 1–3% density, ~97% gradient compression — what powered Covenant-72B), DeMo/DisTrO (DCT + top-k momentum decoupling, up to 85× less data per GPU), and Pluralis Protocol Models (99% activation compression enabling model parallelism over WAN).
- **Honest counterweight:** Prime Intellect — the most-funded name in the space — trained its flagship INTELLECT-3 (106B MoE) on a *centralized* 512×H200 cluster, a telling signal that for frontier-quality RL post-training, centralized still wins on engineering economics. RL post-training is the most decentralization-friendly workload; full frontier-scale pretraining over WAN remains unproven above ~100B dense.
- **For mindXtrain:** the AOT-only reproducibility discipline is *precisely* the property that verification protocols (Gensyn Verde/RepOps, TOPLOC) require. The natural integration is RL-Swarm-style participation plus an x402-payable training-job surface with checkpoint-hash verification.

---

## 1. The Algorithms: How Training Escaped the Datacenter

The core problem: datacenter training assumes NVLink/InfiniBand (100s of GB/s); the internet gives you 100–1000× less. Every viable approach attacks communication volume.

### DiLoCo family (data-parallel, low-communication)
- **DiLoCo** (DeepMind, [arXiv:2311.08105](https://arxiv.org/abs/2311.08105)) — the foundational recipe: a variant of federated averaging where each worker runs many local AdamW steps (H = hundreds), then synchronizes "pseudo-gradients" via an outer Nesterov-momentum optimizer. On C4, 8 workers matched fully synchronous training while **communicating 500× less**.
- **Streaming DiLoCo** ([arXiv:2501.18512](https://arxiv.org/abs/2501.18512)) — three upgrades: synchronize parameter *subsets* in sequence (slashing peak bandwidth), overlap communication with continued training, and quantize exchanged data. Result: billion-scale training at matching quality with **two orders of magnitude less bandwidth**. This is the blueprint for cross-datacenter training (and the suspected basis of Google's multi-campus Gemini training).
- **OpenDiLoCo** (Prime Intellect, [arXiv:2407.07852](https://arxiv.org/abs/2407.07852), [github.com/PrimeIntellect-ai/OpenDiLoCo](https://github.com/PrimeIntellect-ai/OpenDiLoCo)) — the open implementation (Hivemind-based), demonstrated at 1B+ across 3 countries at 90–95% utilization; scaled in INTELLECT-1 to 10B with int8 pseudo-gradients (~400× communication reduction, [arXiv:2412.01152](https://arxiv.org/abs/2412.01152)).
- **SparseLoCo** (Templar/Bittensor, [arXiv:2508.15706](https://arxiv.org/abs/2508.15706)) — the 2026 state of the art for data-parallel WAN pretraining: error-feedback accumulators + **Top-k sparsification + 2-bit quantization reaching 1–3% density**, which *outperforms* DiLoCo baselines on loss while compressing ~97%+. Key insight: outer momentum can be locally approximated by the error-feedback buffer, and sparse aggregation can actually *improve* performance. This is what trained Covenant-72B over home internet connections.

### Momentum-decoupling (Nous lineage)
- **DeMo — Decoupled Momentum Optimization** ([arXiv:2411.19870](https://arxiv.org/abs/2411.19870), [github.com/bloc97/DeMo](https://github.com/bloc97/DeMo)) — drop-in replacement for momentum optimizers: decouple local momentum, apply a fast DCT transform + top-k sparsification, reuse momentum as error feedback. **Up to 85× less data per GPU** than AdamW-DDP at comparable loss (shown at 300M/1B); topology-agnostic, works over plain Ethernet.
- **DisTrO** (Nous Research, [github.com/NousResearch/DisTrO](https://github.com/NousResearch/DisTrO)) — the productionized family built on DeMo's ideas, reducing inter-GPU transfer by several orders of magnitude; the engine of the Psyche network.

### Model-parallel over WAN (Pluralis)
- **SWARM Parallelism** ([arXiv:2301.11913](https://arxiv.org/abs/2301.11913)) — the precursor: pipeline-parallel training on unreliable, heterogeneous, low-bandwidth nodes (1.3B GPT over ~200Mb/s links with ~2× slowdown).
- **Protocol Models** (Pluralis, [arXiv:2506.01260](https://arxiv.org/abs/2506.01260)) — the breakthrough for *model* parallelism: unlike data-parallel (exchange weight gradients), model-parallel must compress **activations and activation gradients** flowing between layers. Pluralis confines them to a predefined low-dimensional subspace exploited via the transformer's recursive structure, achieving **up to 99% compression with no convergence degradation**. Side effect with economic teeth: no participant ever holds full model weights — the model becomes an unextractable, protocol-native asset ("Unextractable Protocol Models"). Follow-on work: async pipeline-parallel with Nesterov stale-update correction, and >95% compression for context parallelism. Blog: [pluralis.ai/blog](https://pluralis.ai/blog/).

### Other lineages
- **Federated learning** (FedAvg lineage) — the ancestor of all of this; Flower Labs ([github.com/adap/flower](https://github.com/adap/flower)) carries it forward with FlowerLLM/Photon for federated LLM pretraining.
- **Hivemind** ([github.com/learning-at-home/hivemind](https://github.com/learning-at-home/hivemind), MIT) — the P2P substrate (DHT, decentralized averaging, NAT traversal) under OpenDiLoCo, Petals, and Gensyn RL Swarm.
- **Petals** ([github.com/bigscience-workshop/petals](https://github.com/bigscience-workshop/petals)) — collaborative inference/fine-tuning of 100B+ models, BitTorrent-style layer hosting.

**Rule of thumb on bandwidth:** dense DDP needs ~GB/s-class links; DiLoCo-class needs ~100 Mb/s–1 Gb/s with minutes-scale sync windows (8-bit DiLoCo measured ~8.3 min all-reduce at 14 nodes); SparseLoCo/DeMo push viable participation down to consumer broadband.

---

## 2. The Networks: Who's Actually Training What

### Prime Intellect — the open superintelligence stack
- **Track record:** INTELLECT-1 (10B, OpenDiLoCo, 3 continents) → **INTELLECT-2** (32B, first globally distributed RL run; [arXiv:2505.07291](https://arxiv.org/abs/2505.07291)) → **INTELLECT-3** (106B MoE, 12B active, SFT+RL on GLM-4.5-Air base; [arXiv:2512.16144](https://arxiv.org/abs/2512.16144), [huggingface.co/PrimeIntellect/INTELLECT-3](https://huggingface.co/PrimeIntellect/INTELLECT-3), released Nov 27, 2025 — best-in-class math/code/reasoning for its size).
- **The catch:** INTELLECT-3 was trained on a **centralized 512× H200 cluster**, not the decentralized protocol — a candid pivot toward "open-source models + compute platform" over pure decentralization ([blog](https://www.primeintellect.ai/blog/intellect-3), critical coverage: [implicator.ai](https://www.implicator.ai/prime-intellects-intellect-3-open-source-ambition-meets-centralized-reality/)).
- **Platform (2026):** **Lab** ([blog](https://www.primeintellect.ai/blog/lab)) unifies the Environments Hub, hosted RL training, and hosted evals — 10,000+ training jobs run by hundreds of teams; opened fully May 2026. Joined the NVIDIA Nemotron coalition (June 2026). Compute Exchange aggregates global GPU supply.
- **Protocol/token:** peer-to-peer compute protocol live on internal testnet (powered SYNTHETIC-2 and INTELLECT-2); contracts on Base Sepolia with a RewardsDistributor pattern suggesting an eventual token; no token launched as of June 2026.
- **Repos (all permissive):** [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) · [protocol](https://github.com/PrimeIntellect-ai/protocol) · [toploc](https://github.com/PrimeIntellect-ai/toploc) · [shardcast](https://github.com/PrimeIntellect-ai/shardcast) · [verifiers](https://github.com/PrimeIntellect-ai/verifiers) · [OpenDiLoCo](https://github.com/PrimeIntellect-ai/OpenDiLoCo)
- **Funding:** $20M+ total — Founders Fund lead, with Karpathy, Delangue, Dylan Patel, Tri Dao, Emad Mostaque among angels.

### Nous Research / Psyche — DisTrO on Solana
- **Psyche** ([github.com/PsycheFoundation/psyche](https://github.com/PsycheFoundation/psyche), **Rust, Apache 2.0**; [docs](https://nousresearch.com/nous-psyche)) — decentralized training network with **coordination on Solana** for fault-tolerant, censorship-resistant orchestration; compute off-chain running DisTrO-compressed training.
- **Consilience-40B:** dense 40B with DeepSeek-style MLA attention, target ~20T tokens (FineWeb 14T + FineWeb-2 4T + Stack v2 upsampled to 1T) — by parameters×tokens, **the largest distributed pre-training run ever** over the internet; deliberately sized to train on one HGX and infer on a 3090. As of the ["Next Phase of Psyche"](https://nousresearch.com/the-next-phase-of-psyche) (Nov 2025), the testnet run validated internet-bandwidth training at scale and Psyche pivoted to training multiple models in parallel.
- **Token status (important):** as of April 2026 **no official Nous/Psyche token exists** — "NOUS" pairs on Solana DEXs are unofficial; don't confuse with Nosana ($NOS).
- **Funding:** $50M from Paradigm at ~$1B valuation.
- **Models:** Hermes series ([huggingface.co/NousResearch](https://huggingface.co/NousResearch)) validates the open-model credibility that underwrites the network.

### Gensyn — verification-first ML compute protocol
- **Architecture:** four primitives — execution, verification, communication, coordination — on a custom Ethereum-rollup testnet. Backed by a16z ($43M Series A).
- **Status (2026):** RL Swarm (peaked ~12,000 testnet nodes; later environments: CodeZero coding swarm) and BlockAssist/CodeAssist have been **paused/sunset**; focus consolidated on **Delphi**, a "prediction market for machine intelligence," as the first Mainnet application. **Mainnet not yet launched** as of June 2026; testnet docs: [docs.gensyn.ai/testnet](https://docs.gensyn.ai/testnet).
- **Repos:** [rl-swarm](https://github.com/gensyn-ai/rl-swarm) · [rl-swarm-contracts](https://github.com/gensyn-ai/rl-swarm-contracts) · [repops-demo](https://github.com/gensyn-ai/repops-demo). RL Swarm hardware floor was deliberately low: arm64/x86 CPU + 32GB RAM, or NVIDIA 3090/4090/5090/A100/H100; macOS, Linux, Windows-WSL2; Python 3.10–3.13.
- **Research:** Verde ([arXiv:2502.19405](https://arxiv.org/abs/2502.19405)), SAPO swarm-sampling policy optimization, NoLoCo (no-all-reduce training, [arXiv:2506.10911](https://arxiv.org/abs/2506.10911)), Gauntlet-style contribution scoring lineage.

### Templar / Bittensor SN3 — the permissionless proof
- **Covenant-72B** (announced March 10, 2026): **72B params, ~1.1T tokens, 70+ independent miners, fully permissionless** — anyone with GPUs could join/leave mid-run — over commodity internet. MMLU 67.1 (~LLaMA-2-70B class). Enabled by **SparseLoCo** (146× communication reduction claimed via sparsification + 2-bit quantization + error feedback) and the **Gauntlet** contributor-scoring system (loss-based evaluation of each node's submitted updates, with TAO/alpha incentives and slashing-style penalties for junk contributions). Apache-licensed model.
- **Ecosystem effects:** τemplar token +194% in a week; TAO ~+30–40%; Jensen Huang likened it to "folding@home for AI"; coverage from Jack Clark's Import AI. Bittensor in March 2026: ~128 active subnets, TAO ~$3.4B market cap, subnet alpha tokens ~$1.4B combined (see [arXiv risk study](https://arxiv.org/pdf/2603.29751)).
- **Links:** [tplr.ai](https://tplr.ai) · [github.com/tplr-ai/templar](https://github.com/tplr-ai/templar) · Bittensor: [github.com/opentensor/bittensor](https://github.com/opentensor/bittensor) · related training subnets: Macrocosmos IOTA ([macrocosmos.ai](https://www.macrocosmos.ai)) for pipeline-parallel pretraining experiments.
- **dTAO mechanics:** each subnet has its own alpha token bonded against TAO; miners earn by validator-scored contribution quality — the only live, fully incentivized, permissionless training market as of mid-2026.

### Pluralis Research — Protocol Learning (model parallel)
- **Node0-7.5B** ([dashboard.pluralis.ai](https://dashboard.pluralis.ai), [github.com/PluralisResearch/node0](https://github.com/PluralisResearch/node0)): the first public **model-parallel** internet pretraining run — completed after **36B tokens over 3 weeks with 300+ active participants and 1,642 GPUs across 198 cities**, joinable with a single 16GB consumer GPU (3090-class). Built on Protocol Models compression ([arXiv:2506.01260](https://arxiv.org/abs/2506.01260)).
- **Strategic differentiator:** weights are sharded such that **no participant can extract the full model** — enabling on-protocol model ownership, revenue attribution, and access gating (deeply relevant to DAIO-style on-chain asset thinking). Funding: $7.6M seed (USV, CoinFund).

### Others worth tracking
- **Flower Labs** ([flower.ai](https://flower.ai), [github.com/adap/flower](https://github.com/adap/flower), Apache 2.0) — federated LLM training (FlowerLLM; Photon system paper) with the largest federated-learning developer community.
- **Exo Labs** ([github.com/exo-explore/exo](https://github.com/exo-explore/exo)) — cluster your own heterogeneous consumer devices (Macs, mining rigs) for local training/inference; not a token network.
- **Petals / Hivemind** — see §1; research substrate more than incentive network.
- **Compute marketplaces (supply side, not training protocols):** Akash ([akash.network](https://akash.network), AKT), io.net (IO), Render, Aethir, Spheron — these price raw GPU hours; training networks sit a layer above.
- **FedML/TensorOpera, Bagel** ([bagel.net](https://bagel.net) — "Bakery" fine-tuning marketplace research) — earlier-stage or pivoted.

---

## 3. Verification: The Trust Layer

(Extends the verification section of the main survey — repos there remain canonical.)

| Approach | System | Verifies | Production status |
|---|---|---|---|
| Activation LSH | [TOPLOC](https://github.com/PrimeIntellect-ai/toploc) ([arXiv:2501.16007](https://arxiv.org/abs/2501.16007)) | Inference/rollouts | Used in INTELLECT-2 |
| Refereed delegation + bitwise-reproducible ops | Verde + RepOps ([arXiv:2502.19405](https://arxiv.org/abs/2502.19405), [repops-demo](https://github.com/gensyn-ai/repops-demo)) | **Training** steps | Gensyn testnet |
| Optimistic fraud proofs | [opML](https://github.com/ora-io/opml) ([arXiv:2401.17555](https://arxiv.org/abs/2401.17555)) | Inference (training targeted) | ORA on-chain AI |
| zkML | [EZKL](https://github.com/zkonduit/ezkl), [ddkang/zkml](https://github.com/ddkang/zkml), Lagrange DeepProve | Small-model inference proofs | Niche; cost-bound |
| Economic scoring | Templar **Gauntlet** (loss-evaluation of contributions + token slashing) | Training contributions statistically | **Live, incentivized** (SN3) |
| TEEs | NVIDIA Confidential Computing (H100), Intel TDX, AWS Nitro | Execution environment | Growing in compute markets |

**The honest state:** cryptographic verification of *pretraining* at scale remains unsolved in production. Templar's Gauntlet shows the pragmatic alternative — statistical/economic verification (does your update reduce loss?) backed by stake. Verde/RepOps is the most principled training-verification design but needs deterministic execution — which is exactly what an **AOT-only artifact policy** provides. zkML proof costs are still orders of magnitude above native compute for LLM-scale work.

---

## 4. Why RL Is the Decentralization Sweet Spot

- RL post-training = **embarrassingly parallel rollout generation** (inference-heavy, communication-light) + a small trainer. INTELLECT-2's architecture is the template: [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) async trainer ← TOPLOC-verified rollouts from untrusted inference nodes ← [shardcast](https://github.com/PrimeIntellect-ai/shardcast) weight broadcasts.
- Gensyn's RL Swarm generalized this into multi-agent collaborative RL (answer/critique/revise games; SAPO swarm sampling), demonstrating swarm-trained models learn faster than solo — and that heterogeneous, consumer hardware can contribute usefully because rollouts don't need gradient sync.
- The **environments economy** is the new commodity layer: Prime Intellect's Environments Hub + [verifiers](https://github.com/PrimeIntellect-ai/verifiers) library, Gensyn CodeZero, reasoning-gym lineage. Whoever owns high-quality verifiable environments owns RL training demand. (For PYTHAI: blockchain task environments — Foundry test-passing, contract auditing, x402 flow completion — are an unclaimed niche.)
- Caveat from the main survey still holds: decentralized RL gains concentrate in trained domains (math/code); broad transfer lags centralized RL.

---

## 5. Economics & Crypto Integration

- **Live token economics:** only Bittensor — TAO emission split across 128 subnets via dTAO; subnet alpha tokens (τemplar) reprice on demonstrated capability. Covenant-72B was the first event where a training result directly repriced a token 194%.
- **Pending:** Prime Intellect (Base testnet contracts, RewardsDistributor pattern, no token), Gensyn (testnet points → expected token at Mainnet; Delphi first), Psyche (Solana-coordinated, explicitly **no official token yet** as of April 2026 — beware impostor "NOUS" pairs).
- **Funding landscape:** a16z→Gensyn ($43M), Paradigm→Nous ($50M @ ~$1B), Founders Fund→Prime Intellect ($20M+), USV/CoinFund→Pluralis ($7.6M); DCG's Yuma accelerates Bittensor ecosystem; Grayscale holds TAO.
- **Sober read:** the only mechanism so far proven to incentivize *useful* training (not speculation) is Templar's loss-scored, slashing-backed contribution market. Everything else either pays points (Gensyn), pays nothing yet (Psyche, Pluralis Node0 — reputational/dashboard credit), or routes around tokens entirely (Prime Intellect's fiat compute exchange).
- **x402 relevance:** none of these networks natively meter per-job crypto payments; a per-training-job x402 paywall (Algorand x402-avm "Parsec" in your stack) in front of a verifiable training endpoint is genuinely unbuilt territory.

---

## 6. Hardware & Network Realities

- **Demonstrated efficiency:** INTELLECT-1 hit 83–96% utilization (14 nodes, 3 continents); OpenDiLoCo 90–95%; Pluralis cites GPT-1.3B pipeline-parallel over 200Mb/s at ~2× slowdown; SparseLoCo makes consumer broadband viable at 72B. Expect 1.2–3× wall-clock penalty vs an equivalent co-located cluster when the algorithm fits, far worse when it doesn't.
- **Consumer hardware floors:** Pluralis Node0 — single 16GB GPU (3090); Gensyn RL Swarm — even CPU+32GB RAM; Templar mining — prosumer multi-GPU favored; Psyche — 3090-class inference target, training nodes larger.
- **AMD/ROCm reality check:** every major network's node software is **NVIDIA/CUDA-first** (Gensyn lists 3090/4090/5090/A100/H100; Pluralis requires CUDA ≤12.x). MI300X participation today means either contributing through GPU marketplaces (Prime Intellect Compute Exchange lists heterogeneous supply) or running protocol-side/trainer-side infrastructure rather than mining. This is a gap — and an opening for ROCm-native node ports.
- **Networking stacks:** Hivemind DHT (+ relays/NAT traversal) dominates (OpenDiLoCo, Petals, RL Swarm); Psyche uses Solana for coordination + P2P data plane (iroh-class Rust networking); Templar uses Bittensor's axon/dendrite gossip + object storage for gradient exchange.
- **Churn tolerance:** all serious systems assume nodes join/leave mid-run — DiLoCo's infrequent sync, SWARM's stochastic rewiring, Gauntlet's per-contribution scoring, and Psyche's on-chain checkpointing all exist precisely for this.

---

## 7. Critical Assessment & Open Problems

1. **Scale ceiling:** largest decentralized pretraining = 72B dense / ~1.1T tokens (Covenant). Frontier centralized runs are training 10×+ larger models on 50×+ tokens with 100,000+ GPU clusters. The gap is closing on a log scale, not disappearing.
2. **The Prime Intellect signal:** when the best-funded decentralized lab trains its flagship centrally (512×H200) while open-sourcing the stack, the message is: decentralization currently wins on *access and sovereignty*, not on cost or speed at frontier quality.
3. **Verification gap:** pretraining verification is economic, not cryptographic. A motivated adversary inside a permissionless run is mitigated (Gauntlet slashing, Byzantine-robust aggregation), not eliminated. Data poisoning in permissionless data-parallel runs remains under-studied.
4. **Model parallelism over WAN** is the frontier — Pluralis is essentially alone in production here; if Protocol Models scales past ~10B with heterogeneous consumer cards, the "no single node has the weights" property changes the ownership game entirely.
5. **Regulatory horizon:** the "no-off problem" ([arXiv:2412.07890](https://arxiv.org/pdf/2412.07890)) — once training is a protocol, no one can stop it. Expect compute-governance and export-control attention as runs approach frontier capability.
6. **Forecast:** decentralized *post-training* (RL, fine-tuning, distillation) reaches economic parity first — arguably already there for verifiable-reward domains. Decentralized *pretraining* plausibly reaches 100B+ dense / multi-trillion tokens by 2027 via SparseLoCo-class compression + dTAO-class incentives, but frontier parity requires either an algorithmic surprise or centralized-compute commoditization.

---

## 8. Practical Integration for mindXtrain / PYTHAI

**Participate (today, ranked by fit):**
1. **Prime Intellect Lab / Environments Hub** — publish blockchain-native RL environments (Foundry-test-passing, Solidity audit, Algorand x402 flows) via the [verifiers](https://github.com/PrimeIntellect-ai/verifiers) library; train against them with hosted RL or your own prime-rl deployment. Lowest friction; AMD-agnostic since you consume the platform.
2. **Templar SN3 mining** ([github.com/tplr-ai/templar](https://github.com/tplr-ai/templar)) — the only incentivized live training market; NVIDIA prosumer hardware; real TAO/alpha yield, real slashing risk.
3. **Pluralis Node0-class events** ([github.com/PluralisResearch/node0](https://github.com/PluralisResearch/node0)) — 16GB+ NVIDIA GPU, port 49200 exposed, Docker; watch for the next run.
4. **Psyche** ([github.com/PsycheFoundation/psyche](https://github.com/PsycheFoundation/psyche)) — Rust/Apache-2.0, Solana coordination (your chain-stack adjacency is an advantage); contribution currently reputational.
5. **Gensyn** — RL Swarm paused; watch Delphi → Mainnet for the token-incentivized restart.

**Build (the mindXtrain thesis):**
- Your **AOT-only discipline is the verification primitive**: deterministic compiled artifacts + pinned ROCm/libtorch = exactly the bitwise-reproducibility Verde/RepOps demands. A mindXtrain node that ships its AOT probe artifact alongside checkpoint hashes is *natively verifiable*.
- **Training-as-a-service with x402:** front a prime-rl or Axolotl/TRL pipeline with an x402-metered endpoint (Parsec on Algorand from your stack); escrow per-job payment against TOPLOC-style rollout proofs or Verde-style checkpoint-hash spot-checks; settle on completion. Register the service as an ERC-8004 agent on AgenticPlace. Nobody has shipped this combination.
- **ROCm node ports** of rl-swarm / node0 / psyche clients are an open contribution lane with outsized visibility — every network is CUDA-locked and knows it.

---

## Network Comparison Table

| Network | Algorithm | Largest demonstrated | Verification | Token (Jun 2026) | Min hardware | Code (license) |
|---|---|---|---|---|---|---|
| Prime Intellect | OpenDiLoCo → prime-rl async RL | INTELLECT-2 32B RL (decentralized); INTELLECT-3 106B MoE (centralized) | TOPLOC | None (Base testnet contracts) | Platform consumer / any | [PrimeIntellect-ai](https://github.com/PrimeIntellect-ai) (Apache 2.0) |
| Nous Psyche | DisTrO/DeMo, Solana coordination | Consilience-40B @ 20T-token target | Solana-anchored checkpoints | None official (beware fakes) | Prosumer GPU+ | [PsycheFoundation/psyche](https://github.com/PsycheFoundation/psyche) (Apache 2.0) |
| Gensyn | RL Swarm (Hivemind), NoLoCo, SAPO | ~12K-node RL swarm (testnet) | Verde + RepOps | Testnet points; token at Mainnet | CPU+32GB RAM or 3090+ | [gensyn-ai](https://github.com/gensyn-ai) (varied OSS) |
| Templar (SN3) | SparseLoCo + Gauntlet | **Covenant-72B, 1.1T tokens, permissionless** | Economic (loss-scored, slashed) | **Live**: TAO + τemplar alpha | Prosumer/multi-GPU NVIDIA | [tplr-ai/templar](https://github.com/tplr-ai/templar) (MIT) |
| Pluralis | Protocol Models (model-parallel, 99% compression) | Node0-7.5B: 1,642 GPUs, 198 cities | Weight-sharding (unextractable) | None | 16GB GPU (3090) | [PluralisResearch/node0](https://github.com/PluralisResearch/node0) |
| Flower | Federated (Photon/FlowerLLM) | Federated LLM pretraining research | — | None | Any | [adap/flower](https://github.com/adap/flower) (Apache 2.0) |

---

## Key Papers Index

[DiLoCo 2311.08105](https://arxiv.org/abs/2311.08105) · [Streaming DiLoCo 2501.18512](https://arxiv.org/abs/2501.18512) · [OpenDiLoCo 2407.07852](https://arxiv.org/abs/2407.07852) · [SparseLoCo 2508.15706](https://arxiv.org/abs/2508.15706) · [DeMo 2411.19870](https://arxiv.org/abs/2411.19870) · [SWARM 2301.11913](https://arxiv.org/abs/2301.11913) · [Protocol Models 2506.01260](https://arxiv.org/abs/2506.01260) · [INTELLECT-1 2412.01152](https://arxiv.org/abs/2412.01152) · [INTELLECT-2 2505.07291](https://arxiv.org/abs/2505.07291) · [INTELLECT-3 2512.16144](https://arxiv.org/abs/2512.16144) · [TOPLOC 2501.16007](https://arxiv.org/abs/2501.16007) · [Verde 2502.19405](https://arxiv.org/abs/2502.19405) · [opML 2401.17555](https://arxiv.org/abs/2401.17555) · [NoLoCo 2506.10911](https://arxiv.org/abs/2506.10911) · [No-Off Problem 2412.07890](https://arxiv.org/pdf/2412.07890)

---

## Caveats

- Token prices, node counts, and network statuses shift weekly; figures here are snapshots from announcements and coverage through early June 2026.
- Covenant-72B performance claims (LLaMA-2-70B parity, 146× compression) originate from the Templar team and secondary coverage; independent replication of the full run is not yet published.
- Several "largest ever" claims (Psyche Consilience vs Covenant) measure different things — parameters, tokens processed, or parameters×tokens — and both teams claim records under their preferred metric.
- AMD/ROCm support statements reflect documented requirements as of writing; check each repo's README before provisioning hardware.
