"""Validated, source-grounded output contracts for every assistant capability."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from generation.intent_classifier import RequestType


class OutputValidationError(ValueError):
    """Raised when an LLM response does not conform to the grounding contract."""


@dataclass(frozen=True)
class ValidatedOutput:
    display_text: str
    citation_ids: tuple[str, ...]


def validate_output(raw: str, request_type: RequestType, allowed_sources: dict[str, tuple[str, int]]) -> ValidatedOutput:
    """Parse strict JSON and reject any citation not present in retrieved context."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise OutputValidationError("پاسخ مدل JSON معتبر نیست.") from error
    if not isinstance(payload, dict):
        raise OutputValidationError("پاسخ مدل باید یک شیء JSON باشد.")
    citations = _citation_ids(payload.get("citations"), allowed_sources)
    if request_type is RequestType.QA:
        answer = _string(payload.get("answer"), "answer")
        return ValidatedOutput(answer, citations)
    if request_type is RequestType.SCENARIO:
        title = _string(payload.get("title"), "title")
        situation = _string(payload.get("situation"), "situation")
        roles = _strings(payload.get("roles"), "roles")
        steps = _strings(payload.get("steps"), "steps")
        safety_notes = _strings(payload.get("safety_notes"), "safety_notes")
        text = "\n".join([f"عنوان: {title}", f"وضعیت: {situation}", "نقش‌ها: " + "؛ ".join(roles),
                          "مراحل اقدام:\n" + "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1)),
                          "نکات ایمنی: " + "؛ ".join(safety_notes)])
        return ValidatedOutput(text, citations)
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise OutputValidationError("Quiz باید حداقل یک سؤال داشته باشد.")
    rendered: list[str] = []
    for index, question in enumerate(questions, 1):
        if not isinstance(question, dict): raise OutputValidationError("ساختار سؤال نامعتبر است.")
        prompt = _string(question.get("question"), "question")
        options = question.get("options")
        if not isinstance(options, dict) or set(options) != {"الف", "ب", "ج", "د"}:
            raise OutputValidationError("هر سؤال باید چهار گزینهٔ الف تا د داشته باشد.")
        option_text = [f"{key}) {_string(options[key], 'option')}" for key in ("الف", "ب", "ج", "د")]
        correct = _string(question.get("correct_option"), "correct_option")
        if correct not in options: raise OutputValidationError("گزینهٔ صحیح معتبر نیست.")
        explanation = _string(question.get("explanation"), "explanation")
        question_citations = _citation_ids(question.get("citation_ids"), allowed_sources)
        citations = tuple(dict.fromkeys((*citations, *question_citations)))
        rendered.append(f"سؤال {index}: {prompt}\n" + "\n".join(option_text) +
                        f"\nپاسخ صحیح: {correct}\nتوضیح: {explanation}")
    return ValidatedOutput("\n\n".join(rendered), citations)


def _citation_ids(value: Any, allowed: dict[str, tuple[str, int]]) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise OutputValidationError("پاسخ باید citationهای غیرخالی داشته باشد.")
    unknown = set(value) - set(allowed)
    if unknown: raise OutputValidationError("citation خارج از منابع بازیابی‌شده است.")
    return tuple(dict.fromkeys(value))


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip(): raise OutputValidationError(f"فیلد {field} نامعتبر است.")
    return value.strip()


def _strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise OutputValidationError(f"فیلد {field} نامعتبر است.")
    return [item.strip() for item in value]
