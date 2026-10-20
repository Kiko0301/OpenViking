from __future__ import annotations

import json

import pytest

from openviking.message import Message, TextPart
from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations
from openviking.session.memory.experience.trajectory_context import (
    SYSTEM_STEP_MAX_CHARS,
    AgentTrajectoryContextProvider,
    TrajectoryTarget,
    trajectory_source_steps,
)


def message(text: str, role: str = "user") -> Message:
    return Message(id="m", role=role, parts=[TextPart(text=text)])


def operations(
    content: str, *, context_rules: str = "", count: int = 1, memory_type: str = "trajectories"
):
    return ResolvedOperations(
        upsert_operations=[
            ResolvedOperation(
                memory_type=memory_type,
                uris=["viking://user/u/memories/trajectories/model_case/model_name.md"],
                memory_fields={
                    "steps": content,
                    "context_rules": context_rules,
                    "trajectory_name": "model_name",
                    "case_name": "case",
                    "outcome": "success",
                },
            )
            for _ in range(count)
        ],
        delete_file_contents=[],
        errors=[],
    )


@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize("source_steps", [None, []])
def test_trajectory_instruction_keeps_the_target_language(language, source_steps):
    provider = AgentTrajectoryContextProvider(
        messages=[], source_steps=source_steps, targets=[TrajectoryTarget(case={"name": "case"})]
    )
    provider._output_language = language

    instruction = provider.instruction()

    assert provider.get_output_language() == language
    assert instruction.startswith(f"Write new prose in {language}.\n")
    if source_steps is not None:
        assert "ASD-STE100 writing principles for new prose" in instruction
        assert "short sentences, direct wording" in instruction
        assert "consistent terms" in instruction
        assert "Keep source quotes and code unchanged" in instruction
        assert "Copy complete original clauses verbatim" in instruction
        assert "800 characters" in instruction


def test_trajectory_prompt_keeps_effect_evidence_without_domain_specific_examples():
    instruction = " ".join(
        AgentTrajectoryContextProvider(
            messages=[], source_steps=[], targets=[TrajectoryTarget(case={"name": "case"})]
        )
        .instruction()
        .split()
    )

    assert "state changes, side effects, and important facts that did not change" in instruction
    assert "Compare supplied values only for the same entity" in instruction
    assert "Do not infer results from missing evidence" in instruction
    assert "not the Case's entire business scope" in instruction
    assert "minimum complete evidence" in instruction
    assert "both successful and failed runs" in instruction
    assert "facts the agent ignored or misused" in instruction
    assert "source values and relevant candidate alternatives" in instruction
    assert "entity, unit, time and scope" in instruction
    assert "keep original and updated values distinct" in instruction
    assert "If scope is ambiguous, retain evidence for the supported interpretations" in instruction
    assert "Do not add unobserved totals, corrected answers or causes" in instruction
    assert "Omit unrelated fields, wrappers, and full histories" in instruction
    assert "Do not copy raw JSON" in instruction
    for example in ("payment_history", "refund", "baggage"):
        assert example not in instruction


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh-CN"])
async def test_trajectory_style_does_not_rewrite_source_evidence_or_content(language):
    source = '原文 "yes"; code `state["count"] = 0`; ID A-123; value 50.00.'
    steps = [{"type": "user", "content": source}]
    provider = AgentTrajectoryContextProvider(
        messages=[message(source)],
        source_steps=steps,
        targets=[
            TrajectoryTarget(case={"name": "case"}, trajectory_name="trial", outcome="failure")
        ],
    )
    provider._output_language = language
    payload = json.loads((await provider.prefetch())[0]["content"])
    assert payload["source_steps"] == steps

    label = "The user supplied" if language == "en" else "用户提供"
    content = f"# Trajectory\n1. `user` {label}: {source}"
    result = operations(content)
    assert await provider.validate_resolved_operations(result) == []
    assert result.upsert_operations[0].memory_fields["steps"] == content


@pytest.mark.parametrize(
    "content",
    [
        "",
        "The user asked to cancel and the agent checked eligibility.",
        "# Trajectory\n1. Missing context section",
        "# Session Context\nRules without any trajectory",
        "# Trajectory\n1. step\n# Session Context\nreversed",
        "# Session Context\n# Trajectory\n1. step\n# Session Context\nextra",
        "# Session Context\n# Trajectory\n1. step\n# Trajectory\n2. extra",
        "# Session Context\n1. Rule only\n# Trajectory\n",
        "# Session Context\n# Trajectory\n2. first\n5. second",
        '1. `tool:update` flights=[{"flight_number":"HAT072"}], payment_id=gift',
        "```markdown\n# Session Context\nRule\n# Trajectory\n1. step\n```",
        "\n".join(f"{index}. step" for index in range(1, 301)),
    ],
)
async def test_content_is_kept_without_heading_numbering_count_or_type_binding(content):
    provider = AgentTrajectoryContextProvider(
        messages=[message("Cancel the booking")],
        source_steps=[{"type": "user", "content": "Cancel the booking"}],
        targets=[
            TrajectoryTarget(
                case={"name": "case"},
                trajectory_name="server_fixed",
                outcome="failure",
                target_uri="viking://user/u/memories/trajectories/case/server_fixed.md",
            )
        ],
    )
    result = operations(content)

    assert await provider.validate_resolved_operations(result) == []

    operation = result.upsert_operations[0]
    assert operation.memory_fields == {
        "steps": content,
        "context_rules": "",
        "trajectory_name": "server_fixed",
        "case_name": "case",
        "outcome": "failure",
        "action_scores": [],
        "action_score_sources": {},
        "evaluation_criteria": [],
    }
    assert operation.uris == [provider.targets["case"].target_uri]


@pytest.mark.parametrize("count", [0, 2])
async def test_single_trajectory_constraint_is_retained(count):
    provider = AgentTrajectoryContextProvider(
        messages=[], source_steps=[], targets=[TrajectoryTarget(case={"name": "case"})]
    )
    errors = await provider.validate_resolved_operations(operations("plain text", count=count))
    assert errors[0]["error"] == "Create exactly one Trajectory per supplied Case"


async def test_trajectory_memory_type_constraint_is_retained():
    provider = AgentTrajectoryContextProvider(
        messages=[], source_steps=[], targets=[TrajectoryTarget(case={"name": "case"})]
    )
    errors = await provider.validate_resolved_operations(
        operations("plain text", memory_type="experiences")
    )
    assert errors[0]["error"] == "Every operation must create a Trajectory"


def test_source_steps_preserve_system_control_dialogue_and_duplicate_messages():
    messages = [
        message("system:\nOriginal policy."),
        message("Original policy.", "system"),
        message("Original policy.", "system"),
        message("# OpenViking Policy\nObtain authorization.", "system"),
        message("# OpenViking Batch Training CaseSpec v1\ncontrol", "system"),
        message("system:\n# OpenViking OutcomeEvaluation\ncontrol"),
        message("task_success: true", "system"),
        message("[Rollout Evaluation] passed", "system"),
        message("My request is to cancel."),
        message("I think cancellation is forbidden.", "assistant"),
        message(""),
        message("[Experience Reminder] Verify eligibility.", "assistant"),
    ]

    steps = trajectory_source_steps(messages)
    assert [step["content"] for step in steps] == [item.content for item in messages]
    assert [step["type"] for step in steps] == [item.role for item in messages[:-1]] + [
        "experience_reminder"
    ]


@pytest.mark.parametrize("role,prefix", [("system", ""), ("user", "system:\n")])
def test_system_rules_keep_larger_budget_independent_of_record_count(role, prefix):
    rules = (
        prefix
        + "Eligibility rules.\n"
        + "policy " * 3_000
        + "\nException: duplicate bookings qualify."
    )
    messages = [message(rules, role)] + [message("ordinary " * 500) for _ in range(100)]
    steps = trajectory_source_steps(messages)

    assert 16_384 < len(rules) < SYSTEM_STEP_MAX_CHARS
    assert steps[0] == {"type": role, "content": rules}
    assert all(len(step["content"]) <= 64_000 // len(messages) for step in steps[1:])
    assert messages[0].content == rules


@pytest.mark.parametrize("role,prefix", [("system", ""), ("user", "system:\n")])
def test_oversized_system_rules_are_still_bounded_with_head_and_tail(role, prefix):
    text = prefix + "RULE_START\n" + "x" * (SYSTEM_STEP_MAX_CHARS * 2) + "\nEXCEPTION_END"
    steps = trajectory_source_steps([message(text, role)])

    assert len(steps[0]["content"]) == SYSTEM_STEP_MAX_CHARS
    assert steps[0]["content"].startswith(prefix + "RULE_START\n")
    assert steps[0]["content"].endswith("\nEXCEPTION_END")
    assert "truncated" in steps[0]["content"]


def test_empty_source_steps_stay_empty():
    assert trajectory_source_steps([]) == []


def test_source_can_exceed_256_records_with_shared_text_budget():
    records = trajectory_source_steps([message("x" * 2_000) for _ in range(300)])
    assert len(records) == 300
    assert all(record["type"] == "user" for record in records)
    assert sum(len(record["content"]) for record in records) <= 64_000


@pytest.mark.asyncio
async def test_system_rules_reach_model_once_through_source_steps():
    messages = [
        message("system:\nCancel only after confirmation."),
        message("Cancel the booking"),
    ]
    steps = trajectory_source_steps(messages)
    provider = AgentTrajectoryContextProvider(
        messages=messages, source_steps=steps, targets=[TrajectoryTarget(case={"name": "case"})]
    )
    payload = json.loads((await provider.prefetch())[0]["content"])

    assert set(payload) == {"cases", "source_steps"}
    assert set(payload["cases"][0]) == {
        "case",
        "task_spec",
        "injected_actions",
        "external_evaluation",
    }
    assert payload["source_steps"] == steps
    assert payload["source_steps"][0] == {
        "type": "user",
        "content": "system:\nCancel only after confirmation.",
    }
    assert not hasattr(provider, "source_context")
    prompt = provider.instruction()
    assert "session_context" not in prompt
    assert "All source messages are supplied once in source_steps" in prompt
    assert "context_rules" in prompt
    assert "template adds the Session Context heading" in prompt
    assert "training metadata, not execution-time rules or actions" in prompt
    content = "Confirmation precedes cancellation.\nThe user asked to cancel."
    result = operations(content)
    assert await provider.validate_resolved_operations(result) == []
    assert result.upsert_operations[0].memory_fields["steps"] == content


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("text", [None, "", "   "])
def test_tool_transport_has_no_empty_dialogue_step(role, text):
    from openviking.message import ToolPart

    tool = ToolPart(
        tool_id="t",
        tool_name="done",
        tool_input={},
        tool_output="Error: USER_CONVERSATION_ACTIVE",
        tool_status="error",
    )
    parts = [tool] if text is None else [TextPart(text=text), tool]
    steps = trajectory_source_steps([Message(id="transport", role=role, parts=parts)])

    assert len(steps) == 1
    assert steps[0]["type"] == "tool:done"
    evidence = json.loads(steps[0]["content"])
    assert evidence["tool_status"] == "error"
    assert evidence["tool_output"] == "Error: USER_CONVERSATION_ACTIVE"


def test_mixed_tool_dialogue_empty_user_confirmations_and_stop_keep_order():
    from openviking.message import ToolPart

    texts = ["task_success: please explain this field", "The policy says: ask before cancelling."]
    steps = trajectory_source_steps(
        [
            message(texts[0]),
            message(texts[1]),
            message("Yes, cancel A."),
            Message(
                id="mixed",
                role="assistant",
                parts=[
                    TextPart(text="I will now cancel A."),
                    ToolPart(
                        tool_id="a",
                        tool_name="cancel_reservation",
                        tool_input={"id": "A"},
                        tool_output="cancelled",
                        tool_status="completed",
                    ),
                    ToolPart(
                        tool_id="b",
                        tool_name="get_booking",
                        tool_input={"id": "B"},
                        tool_status="running",
                    ),
                ],
            ),
            message(""),
            message("###STOP###"),
            message("Task Terminated"),
        ]
    )

    assert [step["type"] for step in steps] == [
        "user",
        "user",
        "user",
        "assistant",
        "tool:cancel_reservation",
        "tool:get_booking",
        "user",
        "user",
        "user",
    ]
    assert [step["content"] for step in steps[:4]] == [
        *texts,
        "Yes, cancel A.",
        "I will now cancel A.",
    ]
    assert [step["content"] for step in steps[-3:]] == ["", "###STOP###", "Task Terminated"]
    assert json.loads(steps[4]["content"])["tool_input"] == {"id": "A"}
    assert json.loads(steps[5]["content"])["tool_status"] == "running"


@pytest.mark.asyncio
async def test_system_rules_are_context_in_single_source_input_not_output_steps():
    rules = "system:\nObtain explicit confirmation. Exception: read-only queries need none."
    dialogue = "The policy says confirmation is needed; please explain why."
    provider = AgentTrajectoryContextProvider(
        messages=[],
        source_steps=trajectory_source_steps([message(rules), message(dialogue)]),
        targets=[TrajectoryTarget(case={"name": "case"}, trajectory_name="t", outcome="unknown")],
    )
    payload = json.loads((await provider.prefetch())[0]["content"])
    assert payload["source_steps"] == [
        {"type": "user", "content": rules},
        {"type": "user", "content": dialogue},
    ]
    assert "session_context" not in payload
    instruction = provider.instruction()
    assert "System rules in source_steps are context, not dialogue steps" in instruction
    assert "Genuine user policy quotations" in instruction
    assert "rejected done call" in instruction


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_request",
    [
        "Choose the smallest sufficient gift card for a $59 charge.",
        "Correct the saved passenger's date of birth.",
    ],
)
@pytest.mark.parametrize("outcome", ["success", "failure"])
async def test_tool_summary_selection_is_task_guidance_not_input_field_filtering(
    user_request, outcome
):
    from copy import deepcopy

    from openviking.message import ToolPart

    profile = {
        "user_id": "user_a",
        "membership": "silver",
        "dob": "1990-01-01",
        "address": "Example address",
        "payment_methods": {
            "gift_card_a": {"amount": 6.0},
            "gift_card_b": {"amount": 113.0},
            "gift_card_c": {"amount": 157.0},
        },
    }
    messages = [
        message(user_request),
        Message(
            id="tool",
            role="assistant",
            parts=[
                ToolPart(
                    tool_id="lookup",
                    tool_name="get_user_details",
                    tool_input={"user_id": "user_a"},
                    tool_output=json.dumps(profile),
                    tool_status="completed",
                )
            ],
        ),
    ]
    original = deepcopy(messages)
    provider = AgentTrajectoryContextProvider(
        messages=messages,
        source_steps=trajectory_source_steps(messages),
        targets=[TrajectoryTarget(case={"name": "case"}, outcome=outcome)],
    )
    prompt = " ".join(provider.instruction().split())
    for requirement in (
        "Select facts for this Case's part of the execution",
        "not the Case's entire business scope",
        "Keep exact task-relevant values",
        "source values and relevant candidate alternatives",
        "not just selected results, totals or differences",
        "both successful and failed runs",
        "facts the agent ignored or misused",
        "keep original and updated values distinct",
        "Do not infer results from missing evidence",
    ):
        assert requirement in prompt
    assert "Rewrite only what happened" in prompt
    assert "Preserve explicit confirmations, tool errors" in prompt

    # Selection belongs to the extraction model, not a hard-coded field filter.
    # This checks available evidence and instructions, not the model's semantic choices.
    payload = json.loads((await provider.prefetch())[0]["content"])
    assert payload["source_steps"][0]["content"] == user_request
    tool = next(s for s in payload["source_steps"] if s["type"] == "tool:get_user_details")
    record = json.loads(tool["content"])
    assert json.loads(record["tool_output"]) == profile
    assert record["tool_input"] == {"user_id": "user_a"}
    assert record["tool_status"] == "completed"
    assert messages == original

    # The new wording does not add a semantic validator or rewrite model-produced text.
    result = operations("1. `tool:get_user_details` observed profile facts.")
    content = result.upsert_operations[0].memory_fields["steps"]
    assert await provider.validate_resolved_operations(result) == []
    assert result.upsert_operations[0].memory_fields["steps"] == content


@pytest.mark.asyncio
async def test_named_reminders_reach_extraction_with_copy_header_guidance():
    note = "[Experience Reminder] require_chinese, ask_reason\n\n**建议**\nKeep the full guidance."
    steps = trajectory_source_steps([message(note)])
    provider = AgentTrajectoryContextProvider(
        messages=[], source_steps=steps, targets=[TrajectoryTarget(case={"name": "case"})]
    )
    instruction = provider.instruction()
    assert "copy its first line exactly" in instruction
    assert "[Experience Reminder] advice_a, advice_b" in instruction
    assert "Do not translate or invent advice names" in instruction
    payload = json.loads((await provider.prefetch())[0]["content"])
    assert payload["source_steps"] == [{"type": "experience_reminder", "content": note}]
