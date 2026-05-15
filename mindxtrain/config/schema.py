"""Pydantic v2 config schema — canonical mindXtrain YAML.

Mirrors the blueprint at docs/blueprints/mindXtrain_ Production Blueprint
for the AMD and lablab.ai Hackathon.md, sections "mindXtrain architecture"
and "Critical code snippets / examples/demo_qwen3_8b_sft.yaml".

Top-level sections:
    meta       — project / run identity / seed / license
    hardware   — gpu name + gfx arch + count + HBM
    autotune   — 60s probe policy (AOT-only — JIT autotune forbidden)
    model      — base model + attention impl + dtype
    data       — HF/local dataset + dedupe + sharding
    train      — backend + method (LoRA/QLoRA/DPO/GRPO/...) + optimizer + env
    eval       — lm-evaluation-harness + regression detector
    quantize   — Quark FP8 / MXFP4 / GPTQ-ROCm
    serve      — vLLM-ROCm / SGLang + reasoning + tool-call parsers
    publish    — HF + Lighthouse + mindX + AgenticPlace + BANKON + x402
    receipt    — provenance manifest output
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, model_validator

# ---- enums / literals -------------------------------------------------------

GfxArch = Literal["gfx942", "gfx950"]
HardwareName = Literal["mi300x", "mi325x", "mi350x", "mi355x"]
TrainingBackend = Literal["axolotl", "unsloth", "torchtune", "primus", "trl_cpu"]
DType = Literal["bfloat16", "float16", "float32", "fp8_e4m3", "mxfp4"]
AttentionBackend = Literal["ck", "triton", "aiter"]
AttnImplementation = Literal["flash_attention_2", "sdpa", "eager"]
DataSource = Literal["hf", "local", "lighthouse", "mindx_dreams"]
QuantScheme = Literal["quark_fp8", "quark_mxfp4", "gptq_rocm", "none"]
ServeBackend = Literal["vllm-rocm", "sglang"]
ReasoningParser = Literal["deepseek_r1", "qwen3", "none"]
ToolCallParser = Literal["hermes", "qwen3_coder", "none"]
X402Network = Literal["algorand", "base", "base-sepolia"]
ScheduleType = Literal["cosine", "linear", "constant", "wsd"]
OptimizerName = Literal["adamw_torch_fused", "adamw_torch", "adamw_8bit", "lion", "adafactor"]
ReceiptIncludeKey = Literal[
    "rocm_version",
    "gfx_arch",
    "container_digest",
    "all_git_shas",
    "yaml_hash",
    "dataset_cids",
    "eval_report",
    "energy_kwh",
]

_DEFAULT_RECEIPT_INCLUDE: list[ReceiptIncludeKey] = [
    "rocm_version",
    "gfx_arch",
    "container_digest",
    "all_git_shas",
    "yaml_hash",
    "dataset_cids",
    "eval_report",
    "energy_kwh",
]


# ---- meta -------------------------------------------------------------------

class MetaCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project: str = Field(min_length=1)
    run_name: str = Field(min_length=1)
    seed: int = Field(default=2048, ge=0)
    license: str = Field(default="apache-2.0")
    description: str = Field(default="")


# ---- hardware ---------------------------------------------------------------

class HardwareCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: HardwareName = "mi300x"
    gfx_arch: GfxArch = "gfx942"
    gpus: Literal[0, 1, 8] = Field(
        default=1,
        description=(
            "0 = CPU lane (mindX self-training, smoke runs). "
            "1 or 8 = MI300X. 2/4 rejected: xGMI bandwidth is asymmetric."
        ),
    )
    expected_hbm_gb: int = Field(default=192, ge=64)


# ---- autotune ---------------------------------------------------------------

class AutotuneCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    plan_path: Path = Field(default=Path("./out/mindxtrain.tuned.yaml"))
    budget_seconds: int = Field(default=60, ge=10, le=600)
    policy: Literal["aot_only"] = Field(
        default="aot_only",
        description="AOT-only — JIT autotune is forbidden in production.",
    )


# ---- model ------------------------------------------------------------------

class ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="HF Hub model ID, e.g. Qwen/Qwen3-8B")
    revision: str | None = None
    attn_implementation: AttnImplementation = "flash_attention_2"
    torch_dtype: DType = "bfloat16"
    trust_remote_code: bool = False


# ---- data -------------------------------------------------------------------

class MinHashCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class SemDedupCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


class DedupeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    minhash: MinHashCfg | None = None
    semdedup: SemDedupCfg | None = None


class ShardCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    num_shards: int = Field(default=1, ge=1)


class DataCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: DataSource = "hf"
    hf_id: str = Field(
        default="",
        description="HF dataset ID (e.g. tatsu-lab/alpaca). Required when source='hf'.",
    )
    path: Path | None = Field(
        default=None,
        description=(
            "Filesystem root. Required when source='local' or 'mindx_dreams'. "
            "For mindx_dreams: the mindX `data/memory` directory."
        ),
    )
    split: str = "train"
    streaming: bool = True
    max_samples: int | None = Field(default=None, ge=1)
    seq_len: int = Field(default=4096, ge=64, le=1_048_576)
    packing: bool = True
    dedupe: DedupeCfg = Field(default_factory=DedupeCfg)
    shard: ShardCfg = Field(default_factory=ShardCfg)
    include_evolutions: bool = Field(
        default=False,
        description=(
            "When source='mindx_dreams', also pull *_evolutions.jsonl "
            "proposals alongside *_training.jsonl consolidation rows. "
            "max_samples caps the combined total."
        ),
    )

    @model_validator(mode="after")
    def _check_source_inputs(self) -> DataCfg:
        if self.source == "hf" and not self.hf_id:
            msg = "data.hf_id is required when data.source='hf'"
            raise ValueError(msg)
        if self.source in {"local", "mindx_dreams"} and self.path is None:
            msg = f"data.path is required when data.source='{self.source}'"
            raise ValueError(msg)
        return self


# ---- train (discriminated method) ------------------------------------------

class _MethodBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FullMethod(_MethodBase):
    kind: Literal["full"] = "full"


class LoraMethod(_MethodBase):
    kind: Literal["lora"] = "lora"
    r: int = Field(default=16, ge=1, le=512)
    alpha: int = Field(default=32, ge=1, le=1024)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


class QLoraMethod(_MethodBase):
    kind: Literal["qlora"] = "qlora"
    r: int = Field(default=16, ge=1, le=512)
    alpha: int = Field(default=32, ge=1, le=1024)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    quant_bits: Literal[4, 8] = 4
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
    )


class DpoMethod(_MethodBase):
    kind: Literal["dpo"] = "dpo"
    beta: float = Field(default=0.1, gt=0.0)


class OrpoMethod(_MethodBase):
    kind: Literal["orpo"] = "orpo"
    beta: float = Field(default=0.1, gt=0.0)


class GrpoMethod(_MethodBase):
    kind: Literal["grpo"] = "grpo"
    num_generations: int = Field(default=4, ge=2)
    kl_coef: float = Field(default=0.04, ge=0.0)


class GspoMethod(_MethodBase):
    """Qwen team's preferred RL algorithm for hybrid + sparse MoE stability (Qwen3-Next/3.5/3.6)."""

    kind: Literal["gspo"] = "gspo"
    num_generations: int = Field(default=4, ge=2)


class KtoMethod(_MethodBase):
    kind: Literal["kto"] = "kto"
    beta: float = Field(default=0.1, gt=0.0)


class CptMethod(_MethodBase):
    kind: Literal["cpt"] = "cpt"


TrainMethod = Annotated[
    FullMethod | LoraMethod | QLoraMethod | DpoMethod | OrpoMethod
    | GrpoMethod | GspoMethod | KtoMethod | CptMethod,
    Discriminator("kind"),
]


class OptimizerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: OptimizerName = "adamw_torch_fused"
    lr: float = Field(default=1e-4, gt=0.0)
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=0.1, ge=0.0)
    grad_clip: float = Field(default=1.0, ge=0.0)


class ScheduleCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ScheduleType = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    epochs: int = Field(default=3, ge=1, le=100)


class BatchCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    per_device: int = Field(default=8, ge=1)
    grad_accum: int = Field(default=4, ge=1)


class FlashAttentionCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: AttentionBackend = "ck"


class FsdpCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    auto_wrap: bool = True


class TrainCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: TrainingBackend = "axolotl"
    method: TrainMethod = Field(default_factory=lambda: LoraMethod())
    optimizer: OptimizerCfg = Field(default_factory=OptimizerCfg)
    schedule: ScheduleCfg = Field(default_factory=ScheduleCfg)
    batch: BatchCfg = Field(default_factory=BatchCfg)
    precision: DType = "bfloat16"
    gradient_checkpointing: bool = True
    flash_attention: FlashAttentionCfg = Field(default_factory=FlashAttentionCfg)
    fsdp: FsdpCfg = Field(default_factory=FsdpCfg)
    env: dict[str, str] = Field(
        default_factory=lambda: {
            "HSA_NO_SCRATCH_RECLAIM": "1",
            "NVTE_CK_USES_BWD_V3": "1",
            "NVTE_CK_IS_V3_ATOMIC_FP32": "1",
            "PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32": "1",
            "NCCL_MIN_NCHANNELS": "112",
            "HIP_FORCE_DEV_KERNARG": "1",
            "PYTORCH_ROCM_ARCH": "gfx942",
        },
    )


# ---- eval -------------------------------------------------------------------

class EvalHarnessCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: list[str] = Field(default_factory=lambda: ["mmlu", "gsm8k", "ifeval", "humaneval"])
    fewshot: int = Field(default=5, ge=0)


class EvalRegressionCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline: str = Field(default="", description="HF model ID of the base model for regression check")
    threshold_pct: float = Field(
        default=-1.0,
        description="fail if any task drops more than threshold_pct (negative = allowed drop)",
    )


class EvalCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    harness: EvalHarnessCfg = Field(default_factory=EvalHarnessCfg)
    regression: EvalRegressionCfg = Field(default_factory=EvalRegressionCfg)


# ---- quantize ---------------------------------------------------------------

class QuantizeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    scheme: QuantScheme = "quark_fp8"
    ptpc: bool = Field(
        default=True,
        description="Per-tensor-per-channel FP8 GEMM (15-30% faster than BlockScale on MI300X).",
    )


# ---- serve ------------------------------------------------------------------

class ServeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: ServeBackend = "vllm-rocm"
    reasoning_parser: ReasoningParser = "qwen3"
    tool_call_parser: ToolCallParser = "hermes"
    tensor_parallel: int = Field(default=1, ge=1)
    max_model_len: int = Field(default=8192, ge=512)
    port: int = Field(default=8000, ge=1024, le=65535)


# ---- publish ----------------------------------------------------------------

class HfPublishCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    repo: str = Field(description="HF Hub repo, e.g. lablab-ai-amd-developer-hackathon/mindxtrain-demo")
    private: bool = False


class LighthousePublishCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    api_key_env: str = "LIGHTHOUSE_API_KEY"


class MindxPublishCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    api_url: str = "https://mindx.pythai.net/v1/agents"
    register_as_capability: bool = True


class AgenticPlacePublishCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    api_url: str = "https://agenticplace.pythai.net/v1/listings"
    chain_map_url: str = "https://agenticplace.pythai.net/allchain.html"


class BankonPublishCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ens_parent: str = "bankon.eth"
    subname: str = ""


class X402Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    network: X402Network = "algorand"
    asset: str = "USDC"
    receiver_via: Literal["parsec_wallet", "coinbase_facilitator"] = "parsec_wallet"
    price_per_1k_tokens: float = Field(default=0.0002, ge=0.0)


class BillingPublishCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    x402: X402Cfg = Field(default_factory=X402Cfg)


class PublishCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    hf: HfPublishCfg | None = None
    lighthouse: LighthousePublishCfg = Field(default_factory=LighthousePublishCfg)
    mindx: MindxPublishCfg = Field(default_factory=MindxPublishCfg)
    agenticplace: AgenticPlacePublishCfg = Field(default_factory=AgenticPlacePublishCfg)
    bankon: BankonPublishCfg = Field(default_factory=BankonPublishCfg)
    billing: BillingPublishCfg = Field(default_factory=BillingPublishCfg)


# ---- receipt ----------------------------------------------------------------

class ReceiptCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output: Path = Field(default=Path("./out/receipt.json"))
    include: list[ReceiptIncludeKey] = Field(
        default_factory=lambda: _DEFAULT_RECEIPT_INCLUDE.copy(),
    )


# ---- root -------------------------------------------------------------------

class XTrainConfig(BaseModel):
    """Canonical mindXtrain YAML — consumed by `mindxtrain` CLI verbs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    meta: MetaCfg
    hardware: HardwareCfg = Field(default_factory=HardwareCfg)
    autotune: AutotuneCfg = Field(default_factory=AutotuneCfg)
    model: ModelCfg
    data: DataCfg
    train: TrainCfg = Field(default_factory=lambda: TrainCfg())
    eval: EvalCfg = Field(default_factory=EvalCfg)
    quantize: QuantizeCfg = Field(default_factory=QuantizeCfg)
    serve: ServeCfg = Field(default_factory=ServeCfg)
    publish: PublishCfg = Field(default_factory=PublishCfg)
    receipt: ReceiptCfg = Field(default_factory=ReceiptCfg)
