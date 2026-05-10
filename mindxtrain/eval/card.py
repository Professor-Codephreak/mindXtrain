"""Auto model-card generation from a completed run.

Renders a HuggingFace-flavored `README.md` from the run's `XTrainConfig`
plus the eval JSON. Uses Jinja2 if installed; falls back to stdlib
`string.Template` so the base install (`uv sync` no extras) still works.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from string import Template
from typing import Any

_FALLBACK_TEMPLATE = Template(
    """---
license: apache-2.0
base_model: ${base_model}
tags:
- mindxtrain
- amd-mi300x
- fine-tuned
---

# ${run_name}

Fine-tuned with [mindxtrain](https://github.com/mindx/mindxtrain) on AMD MI300X.

- **Base model:** ${base_model}
- **Trainer:** ${backend}
- **Run id:** ${run_name}

## Evaluation

```json
${eval_json}
```

## Provenance

- BLAKE3 hashes of config / dataset / checkpoint / eval are recorded in the
  accompanying `manifest.json`.
- ROCm: 7.2.1 / gfx942
"""
)


_JINJA_TEMPLATE = """---
license: apache-2.0
base_model: {{ base_model }}
tags:
- mindxtrain
- amd-mi300x
- fine-tuned
---

# {{ run_name }}

Fine-tuned with [mindxtrain](https://github.com/mindx/mindxtrain) on AMD MI300X.

- **Base model:** {{ base_model }}
- **Trainer:** {{ backend }}
- **Run id:** {{ run_name }}

{% if hyperparams %}## Hyperparameters

{% for k, v in hyperparams.items() %}- **{{ k }}**: {{ v }}
{% endfor %}{% endif %}

## Evaluation

```json
{{ eval_json }}
```

## Provenance

- BLAKE3 hashes recorded in `manifest.json`.
- ROCm 7.2.1 / gfx942 (MI300X).
"""


def render_card(cfg: Any, eval_json: Path | None, out_path: Path) -> Path:
    """Write a `README.md` model card; return the path."""
    base_model = getattr(getattr(cfg, "model", None), "name", "unknown")
    run_name = getattr(getattr(cfg, "meta", None), "run_name", "run")
    backend = getattr(getattr(cfg, "train", None), "backend", "axolotl")

    eval_payload = "{}"
    if eval_json is not None and Path(eval_json).exists():
        try:
            eval_payload = json.dumps(json.loads(Path(eval_json).read_text()), indent=2)
        except (OSError, json.JSONDecodeError):
            pass

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if importlib.util.find_spec("jinja2") is not None:
        from jinja2 import Template as JinjaTemplate

        hyperparams = {
            "learning_rate": getattr(getattr(cfg.train, "optim", None), "learning_rate", None),
            "epochs": getattr(cfg.train, "num_epochs", None),
            "micro_batch_size": getattr(cfg.train, "micro_batch_size", None),
        }
        rendered = JinjaTemplate(_JINJA_TEMPLATE).render(
            base_model=base_model,
            run_name=run_name,
            backend=backend,
            hyperparams=hyperparams,
            eval_json=eval_payload,
        )
    else:
        rendered = _FALLBACK_TEMPLATE.substitute(
            base_model=base_model,
            run_name=run_name,
            backend=backend,
            eval_json=eval_payload,
        )

    out_path.write_text(rendered)
    return out_path


__all__ = ["render_card"]
