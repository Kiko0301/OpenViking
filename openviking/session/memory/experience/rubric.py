# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Domain rubric definitions and immutable, per-execution result snapshots."""

from __future__ import annotations

import json
import math
from typing import Any

import yaml

from openviking_cli.utils import get_logger

logger = get_logger(__name__)


def parse_case_rubric(value: Any) -> list[dict[str, str]]:
    """Read the structured rubric carried in the schema's string field."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("Case rubric must be a JSON list of id/description objects")
    result = []
    seen = set()
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "description"}
            or any(not isinstance(text, str) or not text.strip() for text in item.values())
        ):
            raise ValueError("Each Case rubric item requires non-empty id and description strings")
        if item["id"] in seen:
            raise ValueError(f"Duplicate Case rubric id: {item['id']}")
        seen.add(item["id"])
        result.append(dict(item))
    return result


def normalize_case_rubric(value: Any) -> str:
    """Validate after patch application, before persisting the complete rubric."""
    return json.dumps(
        parse_case_rubric([] if value is None else value), ensure_ascii=False, indent=2
    )


def snapshot_rubric_results(value: Any, rubric: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Keep valid judgments without letting auxiliary results block the factual trajectory.

    The Case rubric is guidance, not an ID whitelist. Preserve structurally valid
    assessments in model order, including unknown/repeated IDs and model descriptions.
    Case definitions only fill missing descriptions; no judgments or ID mappings are invented.
    """
    definitions = {item["id"]: item for item in rubric}
    if isinstance(value, str):
        try:
            value = yaml.safe_load(value)
        except (yaml.YAMLError, RecursionError) as exc:
            mark = getattr(exc, "problem_mark", None)
            logger.warning(
                "Discarding unparseable evaluation_criteria YAML: %s (line=%s, column=%s)",
                type(exc).__name__,
                mark.line + 1 if mark else None,
                mark.column + 1 if mark else None,
            )
            return []
    if not isinstance(value, list):
        logger.warning("Discarding non-list evaluation_criteria: %s", type(value).__name__)
        return []

    results = []
    for item in value:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not item["id"].strip()
        ):
            continue
        rubric_id = item["id"]
        if "passed" not in item or (
            item["passed"] is not None and type(item["passed"]) is not bool
        ):
            continue
        if not isinstance(item.get("evidence"), str) or not item["evidence"].strip():
            continue
        result: dict[str, Any] = {"id": rubric_id}
        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            description = definitions.get(rubric_id, {}).get("description")
        if description is not None:
            result["description"] = description
        result.update(passed=item["passed"], evidence=item["evidence"])
        if "score" in item:
            score = item["score"]
            if (
                score is None
                or type(score) is int
                or (type(score) is float and math.isfinite(score))
            ):
                result["score"] = score
        results.append(result)
    if len(results) != len(value):
        logger.warning(
            "Discarded %d of %d evaluation_criteria entries (invalid fields)",
            len(value) - len(results),
            len(value),
        )
    return results


def validate_external_rubric_results(
    results: list[dict[str, Any]], external_evaluation: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Check reported per-criterion judgments without injecting results or mapping Case IDs.

    Overall outcomes and unstructured evaluator evidence do not imply per-criterion results.
    The provider returns mismatches to the existing extraction repair loop before any write.
    """
    reports = (external_evaluation or {}).get("criterion_results")
    if not isinstance(reports, list):
        return []
    errors = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        name = report.get("criterion_name")
        if not isinstance(name, str) or not name.strip():
            continue
        judgments = {key: report[key] for key in ("passed", "score") if key in report}
        if not judgments:
            continue
        expected = {"id": name, **judgments}
        matches = [result for result in results if result["id"] == name]
        if len(matches) != 1 or any(
            key not in matches[0] or matches[0][key] != value for key, value in judgments.items()
        ):
            errors.append(
                {
                    "error": "Extract exactly one evaluation_criteria entry for this external "
                    "criterion. Copy criterion_name as id and the reported passed/score unchanged; "
                    "do not substitute a Case ID or rejudge the result. Explain discrepancies "
                    "only in evidence.",
                    "expected": expected,
                }
            )
    return errors
