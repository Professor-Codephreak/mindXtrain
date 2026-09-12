"""The Hugging Face extension — the Hub as mindXtrain needs it.

Design rules, learned the hard way on a live node (2026-09):

- **Membership is not write scope.** `whoami` lists orgs you belong to; only the token's own
  permissions say where it can write. `account()` reports both, separately.
- **`list_repo_tree` yields RepoFile AND RepoFolder**, both carrying `.path`; only
  `repo_info().siblings` carry `.rfilename`. Reading the wrong one makes a scan report "nothing
  on the Hub" while the repo is full.
- **The Hub checks a Space's ZeroGPU quota before it checks existence**: `create_repo(exist_ok=True,
  space_hardware=…)` on an EXISTING Space answers 402 once the account's slots are used. Check
  existence first.
- **A Space README's `short_description` must be ≤ 60 characters** or the upload is refused.
- **Never park a write-scoped token on a public Space.** Public Spaces read public repos.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_HF_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


def _token(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    for k in _HF_ENV:
        v = os.environ.get(k)
        if v:
            return v
    return None


def _api(token: Optional[str] = None):
    """(HfApi, None) or (None, {"ok": False, …}) — never raises."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None, {"ok": False, "reason": "huggingface_hub not installed — `uv sync --extra chain`"}
    tok = _token(token)
    if not tok:
        return None, {"ok": False, "reason": f"no token — set one of {', '.join(_HF_ENV)}"}
    return HfApi(token=tok), None


def tree_paths(api, repo_id: str, **kw) -> List[str]:
    """File paths of a repo tree (folders skipped). See the module docstring: the tree's entries
    carry `.path`, not `.rfilename`."""
    return [f.path for f in api.list_repo_tree(repo_id, **kw) if type(f).__name__ != "RepoFolder"]


# ── who the token is ──────────────────────────────────────────────────────────
def account(token: Optional[str] = None) -> Dict[str, Any]:
    """The identity behind the token, its role, the orgs it belongs to, and — separately — the
    namespaces it can actually write."""
    api, err = _api(token)
    if err:
        return err
    try:
        me = api.whoami()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"whoami failed: {type(e).__name__}: {str(e)[:200]}"}
    auth = (me.get("auth") or {}).get("accessToken") or {}
    perms = auth.get("fineGrained") or {}
    writable, creatable = [], []
    for scope in (perms.get("scoped") or []):
        ent = (scope.get("entity") or {})
        name = ent.get("name") or ent.get("type")
        perms_list = scope.get("permissions") or []
        if any(p.startswith("repo.write") or p == "repo.content.write" for p in perms_list):
            writable.append(name)
        if "repo.write" in perms_list:
            creatable.append(name)
    return {"ok": True, "user": me.get("name"), "type": me.get("type"), "is_pro": bool(me.get("isPro")),
            "role": auth.get("role"), "orgs": [o.get("name") for o in (me.get("orgs") or [])],
            "writable_namespaces": writable or None, "creatable_namespaces": creatable or None,
            "note": "membership is not write scope — a token can belong to an org and still be unable to write it"}


# ── bases, fetched before the run rather than during it ───────────────────────
def pull_base(model_id: str, *, token: Optional[str] = None, allow_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Download a base model to the local cache. Do this BEFORE training: a run that discovers a
    missing base three hours in has wasted three hours."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return {"ok": False, "reason": "huggingface_hub not installed — `uv sync --extra chain`"}
    t0 = time.time()
    try:
        path = snapshot_download(model_id, token=_token(token), allow_patterns=allow_patterns)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "model": model_id, "reason": f"{type(e).__name__}: {str(e)[:200]}"}
    p = Path(path)
    return {"ok": True, "model": model_id, "path": str(p), "seconds": round(time.time() - t0, 1),
            "bytes": sum(f.stat().st_size for f in p.rglob("*") if f.is_file())}


def warm(config: Path | str, *, token: Optional[str] = None) -> Dict[str, Any]:
    """Pull whatever a run.yaml says it needs (the base model) so `train` starts cold-free."""
    try:
        import yaml
        cfg = yaml.safe_load(Path(config).read_text())
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"unreadable config: {type(e).__name__}: {str(e)[:160]}"}
    base = ((cfg or {}).get("model") or {}).get("name")
    if not base:
        return {"ok": False, "reason": "config has no model.name"}
    return dict(pull_base(base, token=token), config=str(config))


# ── a finished run, published ─────────────────────────────────────────────────
def _card(repo_id: str, meta: Dict[str, Any]) -> str:
    """A model card written from the run's own numbers. No claim that is not in `meta`."""
    m = meta
    fm = {"license": m.get("license", "apache-2.0"), "library_name": "transformers",
          "pipeline_tag": "text-generation", "tags": ["mindxtrain", "lora"] + list(m.get("tags") or [])}
    if m.get("base"):
        fm["base_model"] = m["base"]
    if m.get("dataset"):
        fm["datasets"] = [m["dataset"]]
    head = "---\n" + "\n".join(
        f"{k}: {json.dumps(v) if isinstance(v, (list, dict)) else v}" for k, v in fm.items()) + "\n---\n\n"
    rows = [("base", m.get("base")), ("method", m.get("method")), ("corpus", m.get("dataset")),
            ("hardware", m.get("hardware")), ("wall", m.get("wall")), ("train loss", m.get("train_loss")),
            ("eval loss", m.get("eval_loss")), ("gate", m.get("gate")), ("framework", "mindXtrain " + str(m.get("framework_version") or ""))]
    table = "\n".join(f"| {k} | {v} |" for k, v in rows if v not in (None, ""))
    return (head + f"# {repo_id.split('/')[-1]}\n\n{m.get('summary', 'A mindXtrain run, published with its evidence.')}\n\n"
            f"| | |\n|---|---|\n{table}\n\n"
            "## Use\n\n```python\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n"
            f'tok = AutoTokenizer.from_pretrained("{repo_id}")\nm = AutoModelForCausalLM.from_pretrained("{repo_id}")\n```\n\n'
            + (f"## The gate\n\n{m['gate_note']}\n\n" if m.get("gate_note") else "")
            + "## Provenance\n\nTrained by [mindXtrain](https://github.com/professor-codephreak/mindXtrain); "
              "the training log ships beside the weights.\n")


def publish_generation(run_dir: Path | str, repo_id: str, *, token: Optional[str] = None, private: bool = False,
                       meta: Optional[Dict[str, Any]] = None, persona_system: Optional[str] = None,
                       include_merged: bool = True, dry_run: bool = False) -> Dict[str, Any]:
    """A finished run as a model repo: merged weights at the root (if present), the LoRA delta under
    `adapter/`, `train.log`, a `Modelfile` for Ollama, and a card built from `meta`.

    `dry_run=True` reports exactly what would be uploaded and touches nothing."""
    run = Path(run_dir)
    if not run.is_dir():
        return {"ok": False, "reason": f"no run dir at {run}"}
    ck = run / "checkpoint"
    merged = run / "ollama_push" / "merged"
    staged: Dict[str, Path] = {}
    if include_merged and merged.is_dir():
        for f in merged.iterdir():
            if f.is_file():
                staged[f.name] = f
    if ck.is_dir():
        for f in ck.iterdir():
            if f.is_file() and f.name != "training_args.bin" or f.name == "training_args.bin":
                staged[f"adapter/{f.name}"] = f
    log = run / "train.log"
    if log.is_file():
        staged["train.log"] = log
    if not staged:
        return {"ok": False, "reason": f"{run} holds neither a checkpoint nor merged weights"}
    meta = dict(meta or {})
    if not meta.get("base"):
        try:
            meta["base"] = json.loads((ck / "adapter_config.json").read_text()).get("base_model_name_or_path")
        except Exception:  # noqa: BLE001
            pass
    card = _card(repo_id, meta)
    modelfile = ("# ollama create <name> -f Modelfile   (from this repo's directory)\nFROM .\n"
                 + (f'SYSTEM """{persona_system}"""\n' if persona_system else "")
                 + 'PARAMETER temperature 0.7\nPARAMETER repeat_penalty 1.3\nPARAMETER stop "<|im_end|>"\n')
    plan = {"repo": repo_id, "private": private, "files": sorted(staged) + ["README.md", "Modelfile"],
            "bytes": sum(f.stat().st_size for f in staged.values())}
    if dry_run:
        return {"ok": True, "dry_run": True, "would_upload": plan, "card_preview": card[:400]}
    api, err = _api(token)
    if err:
        return err
    import tempfile
    import shutil
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for rel, src in staged.items():
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        (stage / "README.md").write_text(card)
        (stage / "Modelfile").write_text(modelfile)
        try:
            api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
            ci = api.upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="model",
                                   commit_message=meta.get("commit_message") or "mindXtrain: a run, published with its evidence")
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "repo": repo_id, "reason": f"{type(e).__name__}: {str(e)[:300]}"}
    return {"ok": True, "repo": repo_id, "url": f"https://huggingface.co/{repo_id}",
            "commit": str(getattr(ci, "oid", ci))[:12], "uploaded": plan}


# ── the corpus ────────────────────────────────────────────────────────────────
def push_dataset(path: Path | str, repo_id: str, *, token: Optional[str] = None, private: bool = False,
                 path_in_repo: str = "", manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Push a training corpus (a folder or one JSONL) with an optional manifest beside it."""
    api, err = _api(token)
    if err:
        return err
    src = Path(path)
    if not src.exists():
        return {"ok": False, "reason": f"no corpus at {src}"}
    try:
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        if src.is_dir():
            ci = api.upload_folder(folder_path=str(src), repo_id=repo_id, repo_type="dataset",
                                   path_in_repo=path_in_repo or None, commit_message="mindXtrain: corpus")
        else:
            ci = api.upload_file(path_or_fileobj=str(src), path_in_repo=f"{path_in_repo}/{src.name}".lstrip("/"),
                                 repo_id=repo_id, repo_type="dataset", commit_message="mindXtrain: corpus")
        if manifest:
            api.upload_file(path_or_fileobj=json.dumps(manifest, indent=1).encode(),
                            path_in_repo=f"{path_in_repo}/MANIFEST.json".lstrip("/"),
                            repo_id=repo_id, repo_type="dataset", commit_message="mindXtrain: corpus manifest")
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "repo": repo_id, "reason": f"{type(e).__name__}: {str(e)[:300]}"}
    return {"ok": True, "repo": repo_id, "url": f"https://huggingface.co/datasets/{repo_id}",
            "commit": str(getattr(ci, "oid", ci))[:12]}


# ── what is actually on the Hub ───────────────────────────────────────────────
_GEN = re.compile(r"(?:^|/)gen(\d+)(?:/|$)")


def lineage(repo_id: str, *, repo_type: str = "model", token: Optional[str] = None,
            local_runs: Optional[Path | str] = None) -> Dict[str, Any]:
    """Generations present in a repo, and — when `local_runs` is given — which local runs are not
    published yet. Uses `tree_paths`, so folders never break the scan."""
    api, err = _api(token)
    if err:
        return err
    try:
        paths = tree_paths(api, repo_id, repo_type=repo_type, recursive=True)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "repo": repo_id, "reason": f"{type(e).__name__}: {str(e)[:200]}"}
    gens: Dict[int, List[str]] = {}
    for p in paths:
        m = _GEN.search(p)
        if m:
            gens.setdefault(int(m.group(1)), []).append(p)
    out = {"ok": True, "repo": repo_id, "repo_type": repo_type, "files": len(paths),
           "generations": {str(g): {"files": len(v),
                                    "adapter": any(x.endswith("adapter_model.safetensors") for x in v),
                                    "merged": any(x.endswith("model.safetensors") and "adapter" not in x for x in v)}
                           for g, v in sorted(gens.items())}}
    if local_runs:
        root = Path(local_runs)
        local = sorted(int(m.group(1)) for d in root.glob("gen*") if d.is_dir() for m in [_GEN.search(d.name + "/")] if m)
        out["local"] = local
        out["unpublished"] = [g for g in local if g not in gens]
    return out


# ── Spaces ────────────────────────────────────────────────────────────────────
def push_space(folder: Path | str, space_id: str, *, token: Optional[str] = None, private: bool = True,
               hardware: Optional[str] = "zero-a10g", variables: Optional[Dict[str, str]] = None,
               secrets: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Push a Gradio folder as a Space. Existence is checked BEFORE creation (the Hub tests the
    ZeroGPU quota first and answers 402 on a repo that already exists), and a README
    `short_description` longer than 60 characters is refused by the Hub, so it is checked here."""
    api, err = _api(token)
    if err:
        return err
    src = Path(folder)
    if not (src / "app.py").is_file():
        return {"ok": False, "reason": f"{src} has no app.py"}
    readme = src / "README.md"
    if readme.is_file():
        m = re.search(r"^short_description:\s*(.+)$", readme.read_text(), re.M)
        if m and len(m.group(1).strip()) > 60:
            return {"ok": False, "reason": f"short_description is {len(m.group(1).strip())} chars; the Hub allows 60"}
    try:
        exists = api.repo_exists(space_id, repo_type="space")
        if not exists:
            api.create_repo(space_id, repo_type="space", space_sdk="gradio", private=private,
                            **({"space_hardware": hardware} if hardware else {}))
        ci = api.upload_folder(folder_path=str(src), repo_id=space_id, repo_type="space",
                               ignore_patterns=["__pycache__/*", "**/__pycache__/*", "*.pyc"],
                               commit_message="mindXtrain UI")
        for k, v in (variables or {}).items():
            api.add_space_variable(space_id, k, v)
        for k, v in (secrets or {}).items():
            api.add_space_secret(space_id, k, v)
        rt = api.get_space_runtime(space_id)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "space": space_id, "reason": f"{type(e).__name__}: {str(e)[:300]}"}
    return {"ok": True, "space": space_id, "existed": exists, "stage": str(rt.stage),
            "url": f"https://huggingface.co/spaces/{space_id}",
            "host": "https://" + space_id.replace("/", "-").replace("_", "-").lower() + ".hf.space",
            "note": "free personal accounts host 2 ZeroGPU Spaces; a free org hosts none (402)"}
