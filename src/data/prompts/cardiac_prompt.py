"""
Definition:
Contract-based prompt helpers for turning canonical cardiac metadata into MINIM text.
---
Results:
Provides EF normalization, payload validation, and segment-based prompt generation.
"""

from __future__ import annotations

from data.utilities.dataset_utilities import (
    compute_age_group,
    compute_bmi_group,
    compute_disease_label,
    compute_sex_label,
)
from data.prompts.prompt_contract import (
    PromptCapabilities,
    PromptField,
    PromptPayload,
    PromptSegment,
    PromptSpec,
)


def to_ef_percentage(ef: float) -> float:
    """
    ########################################
    Definition:
    Normalize an EF value into percentage form.
    ---
    Params:
    ef: Ejection fraction expressed either as a ratio or percentage.
    ---
    Results:
    Returns EF as a percentage in the `[0, 100]` style range.
    ---
    Other Information:
    Values less than or equal to `1.0` are interpreted as ratios.
    ########################################
    """
    return ef * 100.0 if ef <= 1.0 else ef


def classify_ef(ef: float) -> str:
    """
    ########################################
    Definition:
    Map an EF value to the prompt category used by the project.
    ---
    Params:
    ef: Ejection fraction expressed as ratio or percentage.
    ---
    Results:
    Returns one of the supported EF class labels.
    ########################################
    """
    ef_pct = to_ef_percentage(float(ef))

    if ef_pct <= 40.0:
        return "reduced EF"
    if ef_pct <= 49.0:
        return "mildly reduced EF"
    return "normal EF"


CARDIAC_PROMPT_SPEC = PromptSpec(
    hard_requirements=frozenset({PromptField.MODALITY, PromptField.VIEW}),
    segments=(
        PromptSegment(
            required_fields=frozenset({PromptField.MODALITY}),
            render=lambda payload: payload.modality,
        ),
        PromptSegment(
            required_fields=frozenset({PromptField.VIEW}),
            render=lambda payload: payload.view,
        ),
        PromptSegment(
            required_fields=frozenset({PromptField.FRAME}),
            render=lambda payload: str(payload.frame),
        ),
        PromptSegment(
            required_fields=frozenset({PromptField.SEX}),
            render=lambda payload: str(payload.sex),
        ),
        PromptSegment(
            required_fields=frozenset({PromptField.AGE_GROUP}),
            render=lambda payload: str(payload.age_group),
        ),
        PromptSegment(
            required_fields=frozenset({PromptField.BMI_GROUP}),
            render=lambda payload: str(payload.bmi_group),
        ),
        PromptSegment(
            required_fields=frozenset({PromptField.EF_GROUP}),
            render=lambda payload: str(payload.ef_group),
        ),
        PromptSegment(
            required_fields=frozenset({PromptField.DISEASE_LABEL}),
            render=lambda payload: str(payload.disease_label),
        ),
    ),
)


def validate_prompt_contract(
    payload: PromptPayload,
    capabilities: PromptCapabilities,
    spec: PromptSpec = CARDIAC_PROMPT_SPEC,
) -> None:
    """
    ########################################
    Definition:
    Validate hard requirements and declared capabilities before rendering a prompt.
    ---
    Params:
    payload: Canonical prompt payload.
    capabilities: Declared dataset prompt capabilities.
    spec: Prompt specification to validate against.
    ---
    Results:
    Raises ValueError when a required or declared field is missing.
    ########################################
    """
    for field in spec.hard_requirements:
        if not payload.has(field):
            raise ValueError(f"Prompt contract violation: missing required field '{field.value}'.")

    for field in capabilities:
        if not payload.has(field):
            raise ValueError(
                f"Prompt contract violation: capability '{field.value}' was declared without a value."
            )


def generate_prompt(
    payload: PromptPayload,
    capabilities: PromptCapabilities,
    spec: PromptSpec = CARDIAC_PROMPT_SPEC,
) -> str:
    """
    ########################################
    Definition:
    Generate one deterministic prompt from a canonical payload and declared capabilities.
    ---
    Params:
    payload: Canonical prompt payload.
    capabilities: Declared dataset prompt capabilities.
    spec: Prompt specification that controls order and formatting.
    ---
    Results:
    Returns the final prompt string used in CSV export.
    ########################################
    """
    validate_prompt_contract(payload, capabilities, spec=spec)

    rendered_segments: list[str] = []
    for segment in spec.segments:
        if not segment.required_fields.issubset(capabilities):
            continue
        if not all(payload.has(field) for field in segment.required_fields):
            continue
        rendered_segments.append(segment.render(payload))

    if not rendered_segments:
        raise ValueError("Prompt generation failed: no prompt segments were rendered.")

    return f"{spec.separator.join(rendered_segments)}{spec.suffix}"


def build_cardiac_prompt_payload(
    metadata: dict,
    ef: float,
    modality: str = "Cardiac MRI",
    view: str = "short-axis view",
    frame: str | None = None,
) -> PromptPayload:
    """
    ########################################
    Definition:
    Build the canonical cardiac prompt payload from raw metadata and EF.
    ---
    Params:
    metadata: Patient or study metadata with demographic and disease information.
    ef: Ejection fraction value to classify.
    modality: Required modality descriptor.
    view: Required view descriptor.
    frame: Optional frame descriptor.
    ---
    Results:
    Returns the canonical prompt payload.
    ########################################
    """
    try:
        bmi_group = compute_bmi_group(metadata)
    except (KeyError, ValueError):
        bmi_group = None

    return PromptPayload(
        modality=modality,
        view=view,
        frame=frame,
        sex=compute_sex_label(metadata),
        age_group=compute_age_group(metadata),
        bmi_group=bmi_group,
        ef_group=classify_ef(float(ef)),
        disease_label=compute_disease_label(metadata),
    )


def infer_capabilities(payload: PromptPayload) -> PromptCapabilities:
    """
    ########################################
    Definition:
    Infer capabilities from the fields currently populated in one payload.
    ---
    Params:
    payload: Canonical prompt payload.
    ---
    Results:
    Returns the set of populated fields, including required fields.
    ########################################
    """
    capabilities: PromptCapabilities = {PromptField.MODALITY, PromptField.VIEW}
    for field in (
        PromptField.FRAME,
        PromptField.SEX,
        PromptField.AGE_GROUP,
        PromptField.BMI_GROUP,
        PromptField.EF_GROUP,
        PromptField.DISEASE_LABEL,
    ):
        if payload.has(field):
            capabilities.add(field)
    return capabilities


def build_cardiac_prompt(metadata: dict, ef: float) -> str:
    """
    ########################################
    Definition:
    Backward-compatible wrapper that builds a cardiac prompt from raw metadata.
    ---
    Params:
    metadata: Patient or study metadata with body measurements and disease labels.
    ef: Ejection fraction value to classify.
    ---
    Results:
    Returns the final prompt string used in CSV export.
    ########################################
    """
    payload = build_cardiac_prompt_payload(metadata, ef)
    capabilities = infer_capabilities(payload)
    return generate_prompt(payload, capabilities)
