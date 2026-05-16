"""CPU throttle for the trl_cpu lane — schema validation + thread resolver."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mindxtrain.config.schema import (
    CPUThrottleCfg,
    LoraMethod,
    MetaCfg,
    ModelCfg,
    TrainCfg,
    XTrainConfig,
    resolve_thread_count,
)

# ---- resolver -------------------------------------------------------------


@pytest.mark.parametrize(
    ("percent", "total", "expected"),
    [
        (100, 4, 4),   # full host
        (50, 4, 2),    # half on a 4-core laptop
        (25, 4, 1),    # quarter; floor at 1
        (1, 4, 1),     # extreme low; still ≥ 1
        (100, 1, 1),   # single-core box
        (50, 8, 4),    # 8-core CCX → half
        (33, 16, 5),   # rounds down (16 * 33 // 100 = 5)
        (75, 12, 9),   # 12 * 75 // 100 = 9
    ],
)
def test_resolve_thread_count_matrix(percent, total, expected):
    assert resolve_thread_count(percent, total) == expected


def test_resolve_thread_count_rejects_zero_cores():
    with pytest.raises(ValueError, match="total_cores"):
        resolve_thread_count(50, 0)


def test_resolve_thread_count_rejects_out_of_range_percent():
    with pytest.raises(ValueError, match="percent"):
        resolve_thread_count(0, 4)
    with pytest.raises(ValueError, match="percent"):
        resolve_thread_count(101, 4)


# ---- schema validation ----------------------------------------------------


def test_cpu_throttle_defaults():
    """Default = full host, no nice, OMP affinity on. Matches prior behaviour."""
    cfg = CPUThrottleCfg()
    assert cfg.percent == 100
    assert cfg.nice_level == 0
    assert cfg.omp_proc_bind is True


def test_cpu_throttle_explicit_values():
    cfg = CPUThrottleCfg(percent=50, nice_level=10, omp_proc_bind=False)
    assert cfg.percent == 50
    assert cfg.nice_level == 10
    assert cfg.omp_proc_bind is False


def test_cpu_throttle_rejects_invalid_percent():
    with pytest.raises(ValidationError):
        CPUThrottleCfg(percent=0)
    with pytest.raises(ValidationError):
        CPUThrottleCfg(percent=101)
    with pytest.raises(ValidationError):
        CPUThrottleCfg(percent=-5)


def test_cpu_throttle_rejects_invalid_nice_level():
    """POSIX nice range is [-20, 19]."""
    with pytest.raises(ValidationError):
        CPUThrottleCfg(nice_level=-21)
    with pytest.raises(ValidationError):
        CPUThrottleCfg(nice_level=20)


def test_cpu_throttle_is_frozen():
    cfg = CPUThrottleCfg(percent=50)
    with pytest.raises(ValidationError):
        cfg.percent = 75  # type: ignore[misc]


def test_cpu_throttle_extra_keys_forbidden():
    with pytest.raises(ValidationError):
        CPUThrottleCfg(percent=50, mystery="value")  # type: ignore[call-arg]


# ---- TrainCfg wiring ------------------------------------------------------


def test_train_cfg_has_default_throttle():
    """Default TrainCfg gets CPUThrottleCfg with full-host defaults so the
    schema is backward compatible (existing recipes don't need the field)."""
    cfg = TrainCfg()
    assert isinstance(cfg.cpu_throttle, CPUThrottleCfg)
    assert cfg.cpu_throttle.percent == 100


def test_full_xtrain_config_with_throttle():
    """A recipe can opt into a throttled CPU run."""
    cfg = XTrainConfig(
        meta=MetaCfg(project="p", run_name="r"),
        model=ModelCfg(name="HuggingFaceTB/SmolLM2-135M"),
        data={"source": "hf", "hf_id": "tatsu-lab/alpaca"},  # type: ignore[arg-type]
        train=TrainCfg(
            backend="trl_cpu",
            method=LoraMethod(r=8, alpha=16),
            cpu_throttle=CPUThrottleCfg(percent=25, nice_level=10),
        ),
    )
    assert cfg.train.backend == "trl_cpu"
    assert cfg.train.cpu_throttle.percent == 25
    assert cfg.train.cpu_throttle.nice_level == 10


def test_recipes_with_throttle_round_trip_yaml():
    """A YAML recipe can specify cpu_throttle and round-trip through the
    schema, matching the pattern existing recipes use."""
    import yaml as _yaml
    cfg_text = _yaml.safe_dump({
        "meta": {"project": "p", "run_name": "r"},
        "model": {"name": "HuggingFaceTB/SmolLM2-135M"},
        "data": {"source": "hf", "hf_id": "tatsu-lab/alpaca"},
        "train": {
            "backend": "trl_cpu",
            "cpu_throttle": {"percent": 50, "nice_level": 5, "omp_proc_bind": False},
        },
    })
    cfg = XTrainConfig.model_validate(_yaml.safe_load(cfg_text))
    assert cfg.train.cpu_throttle.percent == 50
    assert cfg.train.cpu_throttle.nice_level == 5
    assert cfg.train.cpu_throttle.omp_proc_bind is False


# ---- Backend-side env-var application ------------------------------------


def _xtrain_with_throttle(percent: int, nice_level: int = 0,
                          omp_proc_bind: bool = True) -> XTrainConfig:
    return XTrainConfig(
        meta=MetaCfg(project="p", run_name="r"),
        model=ModelCfg(name="HuggingFaceTB/SmolLM2-135M"),
        data={"source": "hf", "hf_id": "tatsu-lab/alpaca"},  # type: ignore[arg-type]
        train=TrainCfg(
            backend="trl_cpu",
            method=LoraMethod(r=4, alpha=8),
            cpu_throttle=CPUThrottleCfg(
                percent=percent, nice_level=nice_level, omp_proc_bind=omp_proc_bind,
            ),
        ),
    )


def test_apply_cpu_throttle_sets_thread_env_vars(monkeypatch):
    """50% on a synthetic 8-core host → 4 threads in every BLAS env var."""
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM",
                "OMP_PROC_BIND", "OMP_PLACES"):
        monkeypatch.delenv(var, raising=False)

    from mindxtrain.train.backend_trl_cpu import _apply_cpu_throttle

    cfg = _xtrain_with_throttle(50)
    sink_lines: list[str] = []
    threads = _apply_cpu_throttle(cfg, sink_lines.append)

    assert threads == 4
    import os
    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert os.environ["MKL_NUM_THREADS"] == "4"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "4"
    # OMP_PROC_BIND enabled by default → close/cores set.
    assert os.environ["OMP_PROC_BIND"] == "close"
    assert os.environ["OMP_PLACES"] == "cores"
    # tokenizers parallelism left on when threads > 2.
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"
    # The sink got at least one summary line.
    assert any("throttle" in line.lower() for line in sink_lines)


def test_apply_cpu_throttle_disables_tokenizer_parallelism_for_low_thread_counts(
    monkeypatch,
):
    """≤ 2 threads → TOKENIZERS_PARALLELISM=false (Rust tokenizer thrashes
    cache at low core counts during a smoke run on a tiny model)."""
    monkeypatch.setattr("os.cpu_count", lambda: 4)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    from mindxtrain.train.backend_trl_cpu import _apply_cpu_throttle

    cfg = _xtrain_with_throttle(25)  # 25% of 4 → 1 thread
    _apply_cpu_throttle(cfg, lambda _: None)

    import os
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_apply_cpu_throttle_skips_omp_proc_bind_when_disabled(monkeypatch):
    """Recipes can opt out of CCX pinning (e.g., on Intel where it's still
    safe but unnecessary)."""
    monkeypatch.setattr("os.cpu_count", lambda: 4)
    for var in ("OMP_PROC_BIND", "OMP_PLACES"):
        monkeypatch.delenv(var, raising=False)
    from mindxtrain.train.backend_trl_cpu import _apply_cpu_throttle

    cfg = _xtrain_with_throttle(50, omp_proc_bind=False)
    _apply_cpu_throttle(cfg, lambda _: None)

    import os
    # The env vars must NOT be set when the flag is off.
    assert "OMP_PROC_BIND" not in os.environ
    assert "OMP_PLACES" not in os.environ


def test_apply_cpu_throttle_handles_nice_permission_error(monkeypatch):
    """Negative nice needs CAP_SYS_NICE. The throttle must not crash on
    refusal — it should log and continue."""
    monkeypatch.setattr("os.cpu_count", lambda: 4)
    monkeypatch.setattr("os.nice", lambda _: (_ for _ in ()).throw(PermissionError("denied")))
    from mindxtrain.train.backend_trl_cpu import _apply_cpu_throttle

    cfg = _xtrain_with_throttle(50, nice_level=-5)
    sink_lines: list[str] = []
    _apply_cpu_throttle(cfg, sink_lines.append)

    assert any("refused" in line.lower() for line in sink_lines)


def test_apply_cpu_throttle_full_host_when_percent_is_100(monkeypatch):
    """percent=100 (the default) caps at the host's actual cores."""
    monkeypatch.setattr("os.cpu_count", lambda: 4)
    from mindxtrain.train.backend_trl_cpu import _apply_cpu_throttle

    cfg = _xtrain_with_throttle(100)
    threads = _apply_cpu_throttle(cfg, lambda _: None)
    assert threads == 4
