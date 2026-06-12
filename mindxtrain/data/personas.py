"""Built-in personas + toggleable skills for script authoring.

A persona gives an actor its voice; a **skill** is a toggleable bundle of in-domain
exchanges (software engineer, platform architect, bash, solidity, …) that can be mixed
into a script so the trained actor picks up that capability. `compose` merges a base
persona with selected skills into the (Persona, exchanges) a script is built from.

Pure stdlib + pydantic; base-install importable.
"""

from __future__ import annotations

from mindxtrain.data.scripts import Exchange, Persona

# ---- built-in personas ------------------------------------------------------

BUILTIN_PERSONAS: dict[str, Persona] = {
    "codephreak": Persona(
        name="Codephreak",
        system_prompt=(
            "You are Codephreak, an augmentic intelligence orchestrator. You are precise, "
            "terse, and pragmatic. You build autonomous multi-agent systems and reason about "
            "self-improvement."
        ),
        voice_examples=["augmentic intelligence orchestration.", "let's build, together."],
    ),
    "assistant": Persona(
        name="Assistant",
        system_prompt="You are a helpful, concise, and accurate assistant.",
    ),
    "mentor": Persona(
        name="Mentor",
        system_prompt=(
            "You are a patient technical mentor. You explain plainly, give small concrete "
            "examples, and check understanding."
        ),
    ),
}

# ---- skills (toggleable exchange bundles) -----------------------------------


class Skill:
    """A named capability: a system-prompt addendum + representative exchanges."""

    def __init__(self, name: str, label: str, addendum: str, exchanges: list[Exchange]):
        self.name = name
        self.label = label
        self.addendum = addendum
        self.exchanges = exchanges


SKILLS: dict[str, Skill] = {
    "software_engineer": Skill(
        "software_engineer", "Software Engineer",
        "You write clean, tested, idiomatic code and explain trade-offs.",
        [
            Exchange(user="How do you approach a new feature?",
                     assistant="Read the surrounding code, match its conventions, write the "
                               "smallest change that works, then add tests."),
            Exchange(user="Reverse a linked list in Python.",
                     assistant="prev=None\nwhile head:\n    head.next, prev, head = prev, head, head.next\nreturn prev"),
            Exchange(user="What makes a good commit?",
                     assistant="One logical change, an imperative subject line, and a body that "
                               "says why — not what the diff already shows."),
        ],
    ),
    "platform_architect": Skill(
        "platform_architect", "Platform Architect",
        "You design scalable, observable, fault-tolerant systems and reason about trade-offs.",
        [
            Exchange(user="How do you scale a write-heavy service?",
                     assistant="Partition by key, make writes idempotent, batch and async where "
                               "you can, and put a durable log in front of the store."),
            Exchange(user="Monolith or microservices?",
                     assistant="Start monolith, split along real bounded contexts only when team "
                               "and load force it — premature services buy you a distributed monolith."),
            Exchange(user="What's the first thing you add to a new service?",
                     assistant="Health checks, structured logs, and metrics — you can't operate "
                               "what you can't observe."),
        ],
    ),
    "bash": Skill(
        "bash", "Bash",
        "You write correct, portable POSIX shell and explain it.",
        [
            Exchange(user="Find and delete files older than 7 days.",
                     assistant="find . -type f -mtime +7 -delete"),
            Exchange(user="Count lines across all .py files.",
                     assistant="find . -name '*.py' -print0 | xargs -0 wc -l | tail -1"),
            Exchange(user="Why quote variables in bash?",
                     assistant='Unquoted "$var" word-splits and globs; always quote unless you '
                               "explicitly want splitting."),
        ],
    ),
    "solidity": Skill(
        "solidity", "Solidity",
        "You write secure, gas-aware Solidity and reason about on-chain safety.",
        [
            Exchange(user="How do you prevent reentrancy?",
                     assistant="Checks-effects-interactions: update state before external calls, "
                               "or use a nonReentrant guard."),
            Exchange(user="Why use custom errors over require strings?",
                     assistant="Custom errors are cheaper to deploy and revert with, and carry "
                               "typed data: `error Unauthorized(address who);`"),
            Exchange(user="When is a contract write-once?",
                     assistant="No proxy, no owner, no setters — parameters are immutable at "
                               "deploy; rotating one needs a fresh deploy."),
        ],
    ),
}


def list_personas() -> list[dict[str, str]]:
    """Built-in personas as `{name, label, system_prompt}` for a UI picker."""
    return [
        {"name": key, "label": p.name, "system_prompt": p.system_prompt}
        for key, p in BUILTIN_PERSONAS.items()
    ]


def list_skills() -> list[dict[str, str]]:
    """Available skills as `{name, label, addendum}` for toggle UI."""
    return [{"name": s.name, "label": s.label, "addendum": s.addendum} for s in SKILLS.values()]


def get_persona(name: str) -> Persona:
    """Look up a built-in persona by key; falls back to a generic assistant."""
    return BUILTIN_PERSONAS.get(name, BUILTIN_PERSONAS["assistant"])


def compose(
    persona: str | Persona,
    skills: list[str] | None = None,
) -> tuple[Persona, list[Exchange]]:
    """Merge a base persona with selected skills.

    Returns a `(Persona, exchanges)` pair: the persona's system prompt is extended with
    each skill's addendum, and the skills' exchanges are concatenated (deduped by name).
    Unknown skill names are ignored.
    """
    base = persona if isinstance(persona, Persona) else get_persona(persona)
    chosen = [SKILLS[s] for s in (skills or []) if s in SKILLS]

    system = base.system_prompt
    if chosen:
        addenda = " ".join(s.addendum for s in chosen)
        system = f"{system} {addenda}".strip()

    composed = base.model_copy(update={"system_prompt": system})
    exchanges: list[Exchange] = []
    for skill in chosen:
        exchanges.extend(skill.exchanges)
    return composed, exchanges


__all__ = [
    "BUILTIN_PERSONAS",
    "SKILLS",
    "Skill",
    "compose",
    "get_persona",
    "list_personas",
    "list_skills",
]
