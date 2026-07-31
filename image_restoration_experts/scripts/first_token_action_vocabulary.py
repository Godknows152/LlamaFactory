"""Model-facing action aliases with one unique initial per restoration action."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


# The 0721 reasoning catalog uses these human-readable names. The new SFT
# replaces exactly one occurrence in every target with the model-facing alias.
THINKING_ACTION_SURFACES: Mapping[str, tuple[str, ...]] = {
    "real_esrgan": ("Real-ESRGAN",),
    "scunet": ("SCUNet",),
    "retinexformer_fivek": ("Retinexformer-FiveK", "Retinexformer trained on FiveK"),
    "hvicidnet": ("HVI-CIDNet",),
    "lightdiff": ("LightenDiffusion",),
    "turbo_rain": ("Turbo-Rain",),
    "s2former": ("S2Former",),
    "idt": ("IDT",),
    "ridcp": ("RIDCP",),
    "kanet": ("KA-Net",),
    "turbo_snow": ("Turbo-Snow",),
    "snowmaster": ("SnowMaster",),
    "nafnet_denoise": ("NAFNet-Denoise",),
    "focalnet_dehaze": ("FocalNet-Dehaze",),
    "focalnet_desnow": ("FocalNet-Desnow",),
    "mb_taylorformer_dehaze": ("MB-TaylorFormer-Dehaze",),
}


def validate_action_aliases(
    action_aliases: Mapping[str, str], canonical_actions: Sequence[str]
) -> None:
    """Require complete, reversible aliases with unique first characters."""

    expected = set(canonical_actions)
    actual = set(action_aliases)
    if expected != actual:
        raise ValueError(
            "first-token action alias mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )

    aliases = list(action_aliases.values())
    if len(aliases) != len(set(aliases)):
        raise ValueError("first-token action aliases must be unique")

    initials = [alias[0] for alias in aliases]
    if len(initials) != len(set(initials)):
        raise ValueError("first-token action aliases must have unique first characters")


def alias_thinking_text(action: str, text: str, action_aliases: Mapping[str, str]) -> str:
    """Replace the one human-readable action mention with its model alias."""

    surfaces = THINKING_ACTION_SURFACES[action]
    matches = [(surface, text.count(surface)) for surface in surfaces if surface in text]
    if sum(count for _, count in matches) != 1:
        raise ValueError(
            f"thinking target for {action!r} must contain exactly one known action surface: {text!r}"
        )
    surface = matches[0][0]
    return text.replace(surface, action_aliases[action], 1)
