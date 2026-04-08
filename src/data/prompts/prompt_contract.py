"""
Definition:
Canonical contract types for dataset-agnostic prompt generation.
---
Results:
Provides prompt fields, payloads, capabilities, and segment/spec definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class PromptField(str, Enum):
    """
    ########################################
    Definition:
    Enumerate the canonical fields supported by the prompt contract.
    ---
    Results:
    Provides stable field identifiers for dataset capabilities and prompt segments.
    ########################################
    """

    MODALITY = "modality"
    VIEW = "view"
    FRAME = "frame"
    SEX = "sex"
    AGE_GROUP = "age_group"
    BMI_GROUP = "bmi_group"
    EF_GROUP = "ef_group"
    DISEASE_LABEL = "disease_label"


PromptCapabilities = set[PromptField]


@dataclass(frozen=True)
class PromptPayload:
    """
    ########################################
    Definition:
    Store canonical prompt metadata after dataset-specific adaptation.
    ---
    Params:
    modality: Required imaging modality label.
    view: Required anatomical/acquisition view label.
    frame: Optional temporal/frame descriptor.
    sex: Optional normalized sex label.
    age_group: Optional normalized age-group label.
    bmi_group: Optional normalized BMI-group label.
    ef_group: Optional normalized EF-group label.
    disease_label: Optional normalized disease label.
    ---
    Results:
    Provides the validated input contract consumed by the prompt generator.
    ########################################
    """

    modality: str
    view: str
    frame: str | None = None
    sex: str | None = None
    age_group: str | None = None
    bmi_group: str | None = None
    ef_group: str | None = None
    disease_label: str | None = None

    def get(self, field: PromptField) -> str | None:
        """
        ########################################
        Definition:
        Retrieve the payload value associated with one prompt field.
        ---
        Params:
        field: Canonical prompt field identifier.
        ---
        Results:
        Returns the stored value or `None`.
        ########################################
        """
        return getattr(self, field.value)

    def has(self, field: PromptField) -> bool:
        """
        ########################################
        Definition:
        Check whether the payload contains a non-empty value for one prompt field.
        ---
        Params:
        field: Canonical prompt field identifier.
        ---
        Results:
        Returns `True` when the field is present and non-empty.
        ########################################
        """
        value = self.get(field)
        return value is not None and bool(str(value).strip())


RenderFn = Callable[[PromptPayload], str]


@dataclass(frozen=True)
class PromptSegment:
    """
    ########################################
    Definition:
    Describe one ordered prompt fragment with its field requirements.
    ---
    Results:
    Provides the composition unit used by the central prompt generator.
    ########################################
    """

    required_fields: frozenset[PromptField]
    render: RenderFn


@dataclass(frozen=True)
class PromptSpec:
    """
    ########################################
    Definition:
    Store the ordered prompt-composition policy for one prompt family.
    ---
    Results:
    Provides hard requirements, ordered segments, and formatting policy.
    ########################################
    """

    hard_requirements: frozenset[PromptField]
    segments: tuple[PromptSegment, ...]
    separator: str = ", "
    suffix: str = "."
