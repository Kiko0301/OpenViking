# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

import json

import pytest
from test_content_gate import batch, file, operation, patch
from test_edit_repair import repairer, reply, run

from openviking.session.memory.edit_repair_candidates import (
    _find_best_substring_match,
    edit_match_hints,
    get_similarity,
)
from openviking.session.memory.merge_op import FieldType, PatchOp

ACTUAL = "   d. 若同时存在符合取消资格和不符合取消资格的预订，先完成符合资格预订的确认和取消，再处理不符合资格的预订。"
WRONG = ACTUAL.replace("若同时存在", "如同时存在").replace(
    "取消，再处理不符合资格", "取消，再处理不符合取消资格"
)


def test_reuses_legacy_levenshtein_and_substring_scoring():
    assert get_similarity("abc", "adc") == pytest.approx(2 / 3)
    assert get_similarity("“text”\u200b", '"text"') == 1
    assert get_similarity("abc", "") == 0
    score, text = _find_best_substring_match("a much longer prefix green", "grean")
    assert score == pytest.approx(0.8) and text == "green"


def test_real_failure_returns_verbatim_candidate_and_character_differences():
    original = "header\n" + ACTUAL + "\nfooter"
    hints = edit_match_hints(original, WRONG)
    candidate = hints["candidates"][0]
    assert hints["reason"] == "not_found"
    assert candidate["text"] == ACTUAL
    assert candidate["start_line"] == candidate["end_line"] == 2
    assert candidate["match_count"] == 1
    assert {"supplied": "如", "original": "若"} in candidate["differences"]
    assert {"supplied": "取", "original": ""} in candidate["differences"]
    assert {"supplied": "消", "original": ""} in candidate["differences"]
    assert original.count(candidate["text"]) == 1


@pytest.mark.parametrize("wrong", ["3. Transfer to an agent", "85\t3) Transfer to an agent"])
def test_numbering_and_copied_display_line_numbers(wrong):
    actual = "3) Transfer to an agent"
    hint = edit_match_hints(actual, wrong)
    assert hint["candidates"][0]["text"] == actual


def test_multiline_candidates_preserve_raw_mixed_line_endings():
    original = "header\nfirst actual\r\nsecond actual\nfooter"
    hints = edit_match_hints(original, "first actua1\r\nsecond actual")
    best = hints["candidates"][0]
    assert best["text"] == "first actual\r\nsecond actual"
    assert best["text"] in original
    assert (best["start_line"], best["end_line"]) == (2, 3)


def test_candidates_are_ranked_bounded_and_not_truncated():
    original = "red cart\nred cast\nred carp\nred care\nred cart"
    hints = edit_match_hints(original, "red carx")
    assert len(hints["candidates"]) == 3
    assert hints["candidates"][0]["text"] == "red cart"
    assert hints["candidates"][0]["match_count"] == 2
    small = edit_match_hints(original, "red carx", max_chars=9)
    assert len(small["candidates"]) == 1
    assert sum(len(c["text"]) for c in small["candidates"]) <= 9
    assert all(c["text"] in original for c in small["candidates"])
    assert edit_match_hints(original, "red carx", max_chars=0) is None


@pytest.mark.parametrize("original,old", [("hello\r\nworld", "hello\nworld"), ("“text”", '"text"')])
def test_existing_pi_normalization_is_not_misreported_as_missing(original, old):
    assert edit_match_hints(original, old) is None


def test_ambiguous_match_is_reported_instead_of_arbitrarily_choosing_one():
    assert edit_match_hints("same\nsame", "same") == {
        "reason": "non_unique",
        "match_count": 2,
        "candidates": [],
    }


def test_search_limits_are_explicit_and_no_match_does_not_invent_source():
    hints = edit_match_hints("x\n" * 2100 + "3) Actual text", "3. Actual text")
    assert hints["search_limited"] is True and hints["candidates"] == []
    assert edit_match_hints("abc", "z" * 2049)["search_limited"] is True
    hints = edit_match_hints("xyz", "abc")
    assert hints["candidates"] == []


@pytest.mark.asyncio
async def test_model_can_copy_candidate_but_matcher_never_applies_it_automatically():
    source = 'dag = workflow("test")\nreminder = tell("""\n' + ACTUAL + '\n""")'
    ops = batch(operation(patch(WRONG, ACTUAL + " 已确认。"), file(source)))
    before = ops.model_copy(deep=True)
    # Candidate scores alone must not repair the operation. An unchanged model
    # reply is still rejected by the same Pi matcher, even at high similarity.
    repair, model = repairer(reply(WRONG, ACTUAL + " 已确认。"))
    failed = await run(repair, ops)
    assert not failed.repaired and failed.operations is ops and ops == before
    payload = json.loads(model.get_completion_async.await_args.kwargs["messages"][1]["content"])
    hints = payload["targets"][0]["edits"][0]["match_hints"]
    assert hints["candidates"][0]["text"] == ACTUAL

    repair, model = repairer(reply(ACTUAL, ACTUAL + " 已确认。"))
    fixed = await run(repair, ops)
    assert fixed.repaired and ops == before
    fixed_patch = fixed.operations.upsert_operations[0].memory_fields["content"]
    result = await PatchOp(FieldType.STRING).apply(source, fixed_patch)
    assert result == source.replace(ACTUAL, ACTUAL + " 已确认。")
    model.get_completion_async.assert_awaited_once()
    assert (
        "possible locations"
        in model.get_completion_async.await_args.kwargs["messages"][0]["content"]
    )


@pytest.mark.asyncio
async def test_candidate_text_budget_is_shared_across_fields():
    from openviking.session.memory.dataclass import (
        MemoryField,
        MemoryFile,
        MemoryTypeSchema,
        ResolvedOperation,
    )

    uri = "viking://user/u/memories/records/example.md"
    source = "\n".join(str(i) + "a" * 1490 + str(i) for i in range(4))
    wrong = "x" + "a" * 1490 + "x"
    previous = MemoryFile(
        uri=uri,
        memory_type="records",
        extra_fields={"description": source, "notes": source},
    )
    schema = MemoryTypeSchema(
        memory_type="records",
        fields=[
            MemoryField(name=name, field_type=FieldType.STRING) for name in ["description", "notes"]
        ],
    )
    ops = batch(
        ResolvedOperation(
            memory_type="records",
            uris=[uri],
            old_memory_file_content=previous,
            memory_fields={"description": patch(wrong, "new"), "notes": patch(wrong, "new")},
        )
    )
    repair, model = repairer({"status": "cannot_repair", "reason": "Intent is unclear"})
    result = await run(repair, ops, schemas=[schema], errors=[{"uri": uri, "error": "Not found"}])
    assert not result.repaired and result.operations is ops
    payload = json.loads(model.get_completion_async.await_args.kwargs["messages"][1]["content"])
    groups = [
        t["edits"][0].get("match_hints", {}).get("candidates", []) for t in payload["targets"]
    ]
    assert len(groups[0]) == 3 and len(groups[1]) < 3
    assert sum(len(c["text"]) for group in groups for c in group) <= 6000
    assert all(c["text"] in source for group in groups for c in group)
