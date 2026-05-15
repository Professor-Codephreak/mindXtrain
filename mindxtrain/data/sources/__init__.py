"""Dataset source adapters.

`load_streaming_dataset` (`mindxtrain.data.curate`) dispatches to one of these
based on `DataCfg.source`. Each adapter yields rows in the canonical shape
expected downstream:

    {"messages": [{"role": "...", "content": "..."}, ...]}

or a flat `{"text": "..."}` for completion-style corpora.

Adapters are intentionally small and have no GPU/heavy-dep imports at module
load time, so the CLI/Coach can validate configs against arbitrary sources
without pulling `datasets` or `transformers`.
"""
