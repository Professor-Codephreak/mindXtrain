"""Cloud-init `user_data` generator — pure-string tests."""

from __future__ import annotations

import pytest
import yaml

from mindxtrain.deploy.cloud_init import BOOTSTRAP_SENTINEL, render


def test_render_default_is_valid_yaml() -> None:
    text = render()
    # cloud-config files start with a literal `#cloud-config` directive.
    assert text.startswith("#cloud-config\n")
    # And the rest must parse as YAML.
    body = yaml.safe_load(text)
    assert "runcmd" in body
    assert "packages" in body
    assert "git" in body["packages"]
    assert "podman" in body["packages"]


def test_render_sets_bootstrap_sentinel() -> None:
    text = render()
    assert BOOTSTRAP_SENTINEL == "/workspace/mindxtrain/.bootstrap-done"
    assert BOOTSTRAP_SENTINEL in text
    # And the final touch command is exactly the sentinel.
    assert f"touch {BOOTSTRAP_SENTINEL}" in text


def test_render_clones_specified_repo_and_branch() -> None:
    text = render(repo="someowner/somerepo", branch="dev")
    assert "https://github.com/someowner/somerepo.git" in text
    assert "--branch dev" in text


def test_render_run_bench_false_omits_mindxtrain_bench() -> None:
    text = render(run_bench=False)
    assert "mindxtrain bench" not in text
    # But pip install -e is still there to make the env runnable.
    assert "pip install -e .[" in text


@pytest.mark.parametrize(
    "field,bad",
    [
        ("repo", "evil; rm -rf /"),
        ("repo", "a b/c"),               # whitespace
        ("branch", "main$(ls)"),
        ("container", "a`b`c"),
        ("extras", "ml,eval | rm -rf /"),
        ("remote_path", "/tmp/$(whoami)"),
    ],
)
def test_render_rejects_shell_injectable_inputs(field: str, bad: str) -> None:
    kwargs = {field: bad}
    with pytest.raises(ValueError, match="cloud-init"):
        render(**kwargs)


def test_render_with_safe_special_chars() -> None:
    # Colons (registry tags), slashes (repo paths), dots (versions), @ are all valid.
    text = render(container="rocm/primus:v26.2", repo="org-name/repo.name")
    assert "rocm/primus:v26.2" in text
    assert "org-name/repo.name" in text
