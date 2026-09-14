# Example consumer: mindX on the Hugging Face Hub

mindXtrain is agnostic — it trains whatever corpus a consumer points it at. **mindX is its reference consumer**, so this directory shows what a complete consumer footprint on the Hub looks like: the generations mindXtrain published for mindX, the docs mindX's corpus is built from, the Spaces that serve and evaluate the results, and the base-model candidates mindX pinned by licence.

Generated 2026-09-14T20:09:53Z from the Hub's public API. **Public repos only**; mindX keeps the full map (private repos included, with the code that writes each one) in its own repository as `docs/HUGGINGFACE_MAP.md`. Machine-readable copy: [`HUGGINGFACE_MAP.json`](HUGGINGFACE_MAP.json).

- The mapping of every mindX doc: <https://huggingface.co/datasets/PYTHAI/mindX-docs/blob/main/MAPPING.md>
- mindX: <https://mindx.pythai.net> · Hub registry (live): <https://mindx.pythai.net/insight/hf/registry>

## The lineage mindXtrain produced for mindX

| repo | type | licence / sdk | created | what it is |
|---|---|---|---|---|
| [`Gregory-L/machine.dream`](https://huggingface.co/datasets/Gregory-L/machine.dream) | dataset | — | 2026-09-02 | Overflow dream corpus. |
| [`Gregory-L/mindX-ascend-weights`](https://huggingface.co/datasets/Gregory-L/mindX-ascend-weights) | dataset | — | 2026-09-02 | Overflow weights zone. |
| [`PYTHAI/mindXascension`](https://huggingface.co/datasets/PYTHAI/mindXascension) | dataset | — | 2026-09-02 | The weights zone (weights/genN/: full merged model + checkpoint) and the curated dream corpus (machine.dream/). Public — a token-less Space can load only what is here. |
| [`PYTHAI/mindXtrain39`](https://huggingface.co/PYTHAI/mindXtrain39) | model | apache-2.0 | 2026-09-12 | Generation 39 published: merged weights, adapter/, train.log, Modelfile with the persona SYSTEM prompt, THOT.json, inft/ ERC-7857 facets, educational policy and bootcamp impression. The last generation that passed its imprint. |

## mindX's docs, mapped for training

| repo | type | licence / sdk | created | what it is |
|---|---|---|---|---|
| [`PYTHAI/mindX-docs`](https://huggingface.co/datasets/PYTHAI/mindX-docs) | dataset | other | 2026-09-14 | mindX's documentation mapped for mindXtrain — the public tier in full (NAV, THESIS, MANIFESTO: what corpus.doc_rows turns into first-person training rows when no checkout is present), MAPPING.md / mapping.json (every non-private doc: title, tier, words, sha256, page on mindx.pythai.net) and hub_map.json. Public. |

## Spaces mindX runs

| repo | type | licence / sdk | created | what it is |
|---|---|---|---|---|
| [`Gregory-L/mindX-ascend`](https://huggingface.co/spaces/Gregory-L/mindX-ascend) | space | static | 2026-09-02 | Overflow copy of the static lineage dashboard. |
| [`Gregory-L/mindXhfgradio`](https://huggingface.co/spaces/Gregory-L/mindXhfgradio) | space | gradio | 2026-09-11 | mindXhfgradio — the coach, mindXtrain and the Hub as one Gradio app. ZeroGPU, OAuth (visitors spend their own quota), MCP. Free ZeroGPU slot 2 of 2. |
| [`Gregory-L/Savante`](https://huggingface.co/spaces/Gregory-L/Savante) | space | static | 2026-09-14 | Savante (sAGI) public office — the static edition (Ask Savante on the visitor's own token, Hub, Office, Verdict, Integrity, Skills). The Gradio edition waits on PRO. |
| [`PYTHAI/mindX`](https://huggingface.co/spaces/PYTHAI/mindX) | space | static | 2026-09-02 | Static lineage dashboard reading mindx.pythai.net JSON live; the org-branded frame the Gradio Spaces embed in. No compute. |

## Licence-locked pointer forks (base-model candidates)

| repo | type | licence / sdk | created | what it is |
|---|---|---|---|---|
| [`PYTHAI/GLM-5.2-fork`](https://huggingface.co/PYTHAI/GLM-5.2-fork) | model | mit | 2026-09-13 | Licence-locked pointer fork of the upstream `GLM-5.2` release: LICENSE, config, tokenizer and code at a pinned commit, FORK.json provenance, no weights. |
| [`PYTHAI/GLM-5.3-Flash-fork`](https://huggingface.co/PYTHAI/GLM-5.3-Flash-fork) | model | mit | 2026-09-13 | Licence-locked pointer fork of the upstream `GLM-5.3-Flash` release: LICENSE, config, tokenizer and code at a pinned commit, FORK.json provenance, no weights. |
| [`PYTHAI/GLM-5.3-fork`](https://huggingface.co/PYTHAI/GLM-5.3-fork) | model | other | 2026-09-13 | Licence-locked pointer fork of the upstream `GLM-5.3` release: LICENSE, config, tokenizer and code at a pinned commit, FORK.json provenance, no weights. |
| [`PYTHAI/granite-4.2-30b-fork`](https://huggingface.co/PYTHAI/granite-4.2-30b-fork) | model | apache-2.0 | 2026-09-13 | Licence-locked pointer fork of the upstream `granite-4.2-30b` release: LICENSE, config, tokenizer and code at a pinned commit, FORK.json provenance, no weights. |
| [`PYTHAI/granite-4.2-3b-fork`](https://huggingface.co/PYTHAI/granite-4.2-3b-fork) | model | apache-2.0 | 2026-09-13 | Licence-locked pointer fork of the upstream `granite-4.2-3b` release: LICENSE, config, tokenizer and code at a pinned commit, FORK.json provenance, no weights. |
| [`PYTHAI/granite-4.2-8b-fork`](https://huggingface.co/PYTHAI/granite-4.2-8b-fork) | model | apache-2.0 | 2026-09-13 | Licence-locked pointer fork of the upstream `granite-4.2-8b` release: LICENSE, config, tokenizer and code at a pinned commit, FORK.json provenance, no weights. |
| [`PYTHAI/Kimi-K2.7-Code-fork`](https://huggingface.co/PYTHAI/Kimi-K2.7-Code-fork) | model | other | 2026-09-13 | Licence-locked pointer fork of the upstream `Kimi-K2.7-Code` release: LICENSE, config, tokenizer and code at a pinned commit, FORK.json provenance, no weights. |
| [`PYTHAI/Kimi-K3-fork`](https://huggingface.co/PYTHAI/Kimi-K3-fork) | model | other | 2026-09-13 | Licence-locked pointer fork of the upstream `Kimi-K3` release: LICENSE, config, tokenizer and code at a pinned commit, FORK.json provenance, no weights. |
| [`PYTHAI/Qwen3.8-2.4T-A95B-fork`](https://huggingface.co/PYTHAI/Qwen3.8-2.4T-A95B-fork) | model | other | 2026-09-13 | Licence-locked pointer fork of the upstream `Qwen3.8-2.4T-A95B` release: LICENSE, config, tokenizer and code at a pinned commit, FORK.json provenance, no weights. |
| [`PYTHAI/Qwen3.8-27B-fork`](https://huggingface.co/PYTHAI/Qwen3.8-27B-fork) | model | apache-2.0 | 2026-09-13 | Licence-locked pointer fork of the upstream `Qwen3.8-27B` release: LICENSE, config, tokenizer and code at a pinned commit, FORK.json provenance, no weights. |
| [`PYTHAI/Qwen3.8-Flash-Next-fork`](https://huggingface.co/PYTHAI/Qwen3.8-Flash-Next-fork) | model | other | 2026-09-13 | Licence-locked pointer fork of the upstream `Qwen3.8-Flash-Next` release: LICENSE, config, tokenizer and code at a pinned commit, FORK.json provenance, no weights. |

## Earlier demos (2023, pre-mindXtrain lineage)

| repo | type | licence / sdk | created | what it is |
|---|---|---|---|---|
| [`Gregory-L/EleutherAI-gpt-neo-1.3B`](https://huggingface.co/spaces/Gregory-L/EleutherAI-gpt-neo-1.3B) | space | gradio | 2023-06-23 | Thin model demo named after the model (gr.load / gr.Interface.load, or a Streamlit sibling). |
| [`Gregory-L/mrm8488-santacoder-finetuned-the-stack-bash-shell`](https://huggingface.co/spaces/Gregory-L/mrm8488-santacoder-finetuned-the-stack-bash-shell) | space | gradio | 2023-07-11 | Thin model demo named after the model (gr.load / gr.Interface.load, or a Streamlit sibling). |
| [`Gregory-L/openlm-research-open_llama_3b`](https://huggingface.co/spaces/Gregory-L/openlm-research-open_llama_3b) | space | gradio | 2023-07-14 | Thin model demo named after the model (gr.load / gr.Interface.load, or a Streamlit sibling). |
| [`Gregory-L/WizardLM-WizardCoder-15B-V1.0`](https://huggingface.co/spaces/Gregory-L/WizardLM-WizardCoder-15B-V1.0) | space | gradio | 2023-07-11 | Thin model demo named after the model (gr.load / gr.Interface.load, or a Streamlit sibling). |
| [`Gregory-L/ZENMLOS`](https://huggingface.co/spaces/Gregory-L/ZENMLOS) | space | docker | 2023-06-02 | Docker Space (2023). |

## Reading this as a template

A consumer needs the same four things mindX has: a **corpus dataset** mindXtrain can read without a checkout, a **lineage repo** that receives each generation (and records the ones the imprint gate refused), a **surface** that serves or evaluates the result, and a **licence record** for every base model it may train on. Replace `PYTHAI` with your namespace; nothing in the framework assumes mindX.
