"""User control markers are validated before committing a simulator reply to history."""

from __future__ import annotations

import asyncio
import json
import threading
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall, UserMessage
from tau2.user import user_simulator
from tau2.user.user_simulator import UserSimulator, UserState

from benchmark.tau2.common.tau2_env import tau2_environment as env_module
from benchmark.tau2.train.rollout_executor_vikingbot import _tau2_user_reply_terminates
from openviking.session.train import (
    Case,
    ExecutionContext,
    ExperienceSet,
    PipelineEvaluationResult,
    Rollout,
    Rubric,
    RubricEvaluation,
)
from openviking.session.train.components.report_builder import (
    PipelineReportBuilder,
    _analyses_from_rollout_evaluations,
)
from openviking.session.train.components.reporter import ConsolePipelineReporter
from openviking.session.train.components.session_commit import SessionCommitPolicyTrainer


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    monkeypatch.setattr(env_module.time, "sleep", lambda _: None)


def reply(content, *, tool_calls=None):
    return AssistantMessage(role="assistant", content=content, tool_calls=tool_calls)


def case(index=0):
    return Case(
        name=f"tau2_case_{index}",
        task_signature=f"tau2:airline:train:{index}",
        input={
            "domain": "airline",
            "task_id": str(index),
            "task_no": index,
            "data_split": "airline_train",
            "eval_trial": 0,
        },
        rubric=Rubric("rubric", "", []),
    )


def call_user(generate):
    messages = [
        SystemMessage(role="system", content="Simulate a customer."),
        reply("Please cancel my booking."),  # Prior customer turn, after Tau2's role flip.
        UserMessage(role="user", content="The cancellation is complete."),
    ]
    result = env_module._wrap_tau2_generate_with_rate_limit_retry(generate)(
        model="offline", messages=messages, call_name="user_simulator_response"
    )
    assert len(messages) == 3
    return result


@pytest.mark.parametrize("marker", ["###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"])
def test_exact_markers_and_whitespace_are_terminal_but_mixed_text_is_not(monkeypatch, marker):
    monkeypatch.setattr(UserSimulator, "is_stop", classmethod(env_module._tau2_user_is_stop))
    for text in (marker, f" \n{marker}\n "):
        assert UserSimulator.is_stop(UserMessage(role="user", content=text))
        assert _tau2_user_reply_terminates(text)
        generate = Mock(spec=[], return_value=reply(text))
        assert call_user(generate).content == text
        generate.assert_called_once()
    for text in (f"Yes, proceed.\n{marker}", f'Quoted: "{marker}"', f"{marker}\n{marker}"):
        assert not UserSimulator.is_stop(UserMessage(role="user", content=text))
        assert not _tau2_user_reply_terminates(text)
        assert env_module._tau2_user_format_diagnostics(reply(text)) is not None


def test_tool_call_is_not_a_stop_even_with_exact_marker(monkeypatch):
    monkeypatch.setattr(UserSimulator, "is_stop", classmethod(env_module._tau2_user_is_stop))
    calls = [ToolCall(id="call", name="task", arguments={})]
    assert not UserSimulator.is_stop(
        UserMessage(role="user", content="###STOP###", tool_calls=calls)
    )
    assert env_module._tau2_user_format_diagnostics(reply("###STOP###", tool_calls=calls))
    assert env_module._tau2_user_format_diagnostics(reply(None, tool_calls=calls)) is None


def test_invalid_output_retries_current_generation_without_polluting_history(monkeypatch):
    inputs = []
    rejected = reply("Yes, proceed.\n###STOP###")
    accepted = reply("Yes, proceed.")

    def generate(**kwargs):
        inputs.append(deepcopy(kwargs))
        return rejected if len(inputs) < 3 else accepted

    monkeypatch.setattr(
        user_simulator, "generate", env_module._wrap_tau2_generate_with_rate_limit_retry(generate)
    )
    user = UserSimulator(
        llm="offline", instructions="Confirm then wait for completion.", tools=None
    )
    previous = UserMessage(role="user", content="Please change my reservation.")
    state = user.get_init_state([previous])
    original_system = deepcopy(state.system_messages)
    incoming = AssistantMessage(role="assistant", content="Please confirm the changes.")

    actual, same_state = user.generate_next_message(incoming, state)

    assert actual.content == accepted.content
    assert same_state is state
    assert state.messages == [previous, incoming, actual]
    assert state.system_messages == original_system
    assert len(inputs) == 3
    assert (
        len(inputs[1]["messages"]) == len(inputs[2]["messages"]) == len(inputs[0]["messages"]) + 1
    )
    assert inputs[0]["messages"][-1].content == env_module._TAU2_USER_OUTPUT_RULE
    for attempt in inputs[1:]:
        assert attempt["messages"][-1].content == env_module._TAU2_USER_OUTPUT_CORRECTION
    assert rejected.content not in [msg.content for msg in state.messages]


def test_exhaustion_is_bounded_redacted_and_does_not_commit_a_rejected_reply(monkeypatch):
    content = 'api_key="private-value" ' + "x" * 5000 + "###STOP###"
    generate = Mock(spec=[], return_value=reply(content))
    monkeypatch.setattr(
        user_simulator, "generate", env_module._wrap_tau2_generate_with_rate_limit_retry(generate)
    )
    user = UserSimulator(llm="offline", instructions="Wait for completion", tools=None)
    previous = UserMessage(role="user", content="Please change my reservation.")
    state = UserState(system_messages=[], messages=[previous])
    incoming = AssistantMessage(role="assistant", content="Please confirm.")

    with pytest.raises(env_module.Tau2UserSimulatorFormatError, match="after 3 attempts") as caught:
        user.generate_next_message(incoming, state)

    assert generate.call_count == 3
    assert state.messages == [previous, incoming]
    assert state.system_messages == []
    assert "private-value" not in str(caught.value)
    assert len(str(caught.value)) < 2500
    diagnostics = json.loads(str(caught.value).split("diagnostics=", 1)[1])
    assert diagnostics["content_chars"] == len(content)
    assert diagnostics["truncated"] is True
    assert diagnostics["markers"] == ["###STOP###"]


@pytest.mark.parametrize("call_name", [None, "agent_response", "evaluation"])
def test_user_rules_do_not_change_other_model_callers(call_name):
    bad_for_user = reply("Here is a quote: ###STOP###")
    generate = Mock(spec=[], return_value=bad_for_user)
    messages = [SystemMessage(role="system", content="Not a simulator")]
    wrapped = env_module._wrap_tau2_generate_with_rate_limit_retry(generate)

    assert wrapped(model="offline", messages=messages, call_name=call_name) is bad_for_user
    generate.assert_called_once_with(model="offline", messages=messages, call_name=call_name)


def test_reasoning_only_and_format_failures_share_one_budget():
    thinking = AssistantMessage(
        role="assistant",
        content=None,
        raw_data={
            "choices": [{"finish_reason": "stop", "message": {"reasoning_content": "thinking"}}]
        },
    )
    generate = Mock(spec=[], side_effect=[reply("Yes\n###STOP###"), thinking, thinking])

    with pytest.raises(env_module.Tau2UserSimulatorFormatError, match="after 3 attempts"):
        call_user(generate)
    assert generate.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vikingbot", "native"])
@pytest.mark.parametrize(
    "error_type",
    [env_module.Tau2UserSimulatorFormatError, env_module.Tau2UserSimulatorInitializationError],
)
async def test_exhausted_simulator_only_fails_one_rollout_and_skips_learning(
    monkeypatch, backend, error_type
):
    if backend == "vikingbot":
        from benchmark.tau2.train.rollout_executor_vikingbot import VikingBotTau2RolloutExecutor

        executor = VikingBotTau2RolloutExecutor()
        target = "_execute_one_async"
    else:
        from benchmark.tau2.train.rollout_executor_native import NativeTau2RolloutExecutor

        executor = NativeTau2RolloutExecutor()
        target = "_execute_one_sync"
    policy = ExperienceSet("viking://user/default/memories/experiences", [])
    context = ExecutionContext("snapshot", {"training": False, "stage": "eval_train_rollout"})
    good = Rollout(case(1), [], "snapshot", RubricEvaluation(True, 1.0, [], []))
    error = error_type("invalid user output; attempts exhausted")

    def outcome(case, *args):
        if case.input["task_no"] == 0:
            raise error
        return good

    monkeypatch.setattr(
        type(executor),
        target,
        AsyncMock(spec=[], side_effect=outcome)
        if backend == "vikingbot"
        else Mock(spec=[], side_effect=outcome),
    )

    results = await executor.execute([case(0), case(1)], policy, context)

    failed = results[0]
    assert results[1] is good
    assert failed.evaluation.score == 0.0 and not failed.evaluation.passed
    assert failed.metadata["execution_error"]["type"] == error_type.__name__
    assert failed.metadata["execution_error"]["source"] == "tau2_simulator"
    assert failed.metadata["eval_trial"] == 0
    assert failed.case.name == "tau2_case_0"
    client = SimpleNamespace(create_session=AsyncMock())
    result = await SessionCommitPolicyTrainer(client).train_rollouts([failed], policy)
    assert result.apply_result.metadata["skipped_rollout_count"] == 1
    client.create_session.assert_not_awaited()

    builder = PipelineReportBuilder(trial_index_key="eval_trial")
    analysis = _analyses_from_rollout_evaluations(results)
    report = builder.trial_evaluation_report(
        PipelineEvaluationResult(epoch=0, analyses=analysis, policy_snapshot_ids=["snapshot"]),
        trial_count=1,
    )
    assert report["case_count"] == 2 and report["accuracy"] == 0.5
    assert report["execution_error_count"] == 1 and report["execution_error_rate"] == 0.5
    assert report["execution_error_types"] == {f"tau2_simulator:{error_type.__name__}": 1}
    assert report["trials"][0]["execution_error_count"] == 1


def test_console_exposes_execution_errors_without_changing_score(capsys):
    reporter = ConsolePipelineReporter(use_rich=False)
    reporter.on_eval_report(
        label="eval_train_rollout",
        report={
            "epoch": 0,
            "case_count": 2,
            "accuracy": 0.5,
            "passed_count": 1,
            "execution_error_count": 1,
        },
        context=None,
    )
    output = capsys.readouterr().out
    assert "execution_errors=1" in output
    assert "passed=1/2" in output


@pytest.fixture
def failing_gym(monkeypatch):
    from tau2.gym.gym_agent import AgentGymEnv

    generate = Mock(spec=[], return_value=reply("Yes, proceed.\n###STOP###"))
    monkeypatch.setattr(
        user_simulator, "generate", env_module._wrap_tau2_generate_with_rate_limit_retry(generate)
    )
    monkeypatch.setattr(UserSimulator, "is_stop", classmethod(env_module._tau2_user_is_stop))
    user = UserSimulator(llm="offline", instructions="Wait for completion.", tools=None)
    previous = UserMessage(role="user", content="Please change my reservation.")
    state = user.get_init_state([previous])
    incoming = AssistantMessage(role="assistant", content="Please confirm.")

    class FakeGym:
        _run_orchestrator = AgentGymEnv._run_orchestrator

        def __init__(self, **kwargs):
            self._orchestrator = None
            self._simulation_done = threading.Event()
            self._simulation_run = None
            self._agent = SimpleNamespace(_lock=threading.Lock())
            self._format_observation = Mock()
            self._log = Mock()

        def _get_orchestrator(self):
            return SimpleNamespace(
                run=lambda: user.generate_next_message(incoming, state),
                user_state=state,
                termination_reason=None,
                step_count=0,
                num_errors=0,
            )

        def reset(self):
            self._orchestrator = self._get_orchestrator()
            self._run_orchestrator()
            return "", {}

        def step(self, action):
            self._run_orchestrator()
            return "do not deliver invalid output", 0.0, True, False, {}

    monkeypatch.setattr(env_module, "AgentGymEnv", FakeGym)
    monkeypatch.setattr(env_module, "_install_tau2_litellm_rate_limit_retry", lambda: None)
    monkeypatch.setattr(env_module, "_install_tau2_litellm_unknown_cost_suppression", lambda: None)
    return env_module._GymTau2BenchEnv("airline", "offline"), generate, state, incoming


def test_gym_rethrows_typed_format_failure_after_background_swallow(failing_gym):
    env, generate, state, incoming = failing_gym
    history = list(state.messages)
    with pytest.raises(env_module.Tau2UserSimulatorFormatError) as caught:
        env.reset()
    assert env.env._simulation_done.is_set()
    assert generate.call_count == 3
    assert state.messages == [*history, incoming]
    assert "3 attempts" in str(caught.value)
    assert env.execution_error["type"] == "Tau2UserSimulatorFormatError"


def test_gym_mid_dialog_failure_ends_without_delivering_rejected_text(failing_gym):
    env, generate, state, incoming = failing_gym
    history = list(state.messages)
    env.env._orchestrator = env.env._get_orchestrator()
    assert (
        env.tool_call("communicate_with_user", {"content": "Please confirm."}) == "Task Terminated"
    )
    assert env.terminated
    assert generate.call_count == 3
    assert state.messages == [*history, incoming]
    assert env.execution_error["type"] == "Tau2UserSimulatorFormatError"
    assert env.tool_call("communicate_with_user", {"content": "Again?"}) == "Task Terminated"
    assert generate.call_count == 3


def test_rate_limit_retries_do_not_reset_the_output_budget():
    rate_limit = RuntimeError("TPM (Tokens Per Minute) limit of the model is exceeded")
    bad = reply("Yes, proceed.\n###STOP###")
    generate = Mock(spec=[], side_effect=[bad, rate_limit, bad, rate_limit, bad])
    with pytest.raises(env_module.Tau2UserSimulatorFormatError, match="after 3 attempts"):
        call_user(generate)
    assert generate.call_count == 5


@pytest.mark.asyncio
async def test_concurrent_user_generations_do_not_share_corrections_or_retry_counts():
    calls = {}

    def generate(**kwargs):
        model = kwargs["model"]
        calls[model] = calls.get(model, 0) + 1
        if model == "needs-retry" and calls[model] == 1:
            return reply("Yes. ###STOP###")
        if model == "ordinary":
            assert not any(
                msg.content == env_module._TAU2_USER_OUTPUT_CORRECTION for msg in kwargs["messages"]
            )
        return reply("Yes.")

    wrapped = env_module._wrap_tau2_generate_with_rate_limit_retry(generate)
    messages = [SystemMessage(role="system", content="original")]
    results = await asyncio.gather(
        *(
            asyncio.to_thread(
                wrapped, model=model, messages=messages, call_name="user_simulator_response"
            )
            for model in ["needs-retry", "ordinary"]
        )
    )
    assert all(result.content == "Yes." for result in results)
    assert calls == {"needs-retry": 2, "ordinary": 1}
    assert len(messages) == 1


@pytest.mark.parametrize("marker", ["###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"])
def test_initial_terminal_reply_retries_once_without_polluting_history(monkeypatch, marker):
    from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE

    attempts = []
    accepted = reply("My cancelled flight caused me to miss a meeting; I want compensation.")

    def generate(**kwargs):
        attempts.append(deepcopy(kwargs["messages"]))
        return reply(marker) if len(attempts) == 1 else accepted

    monkeypatch.setattr(
        user_simulator, "generate", env_module._wrap_tau2_generate_with_rate_limit_retry(generate)
    )
    user = UserSimulator(llm="offline", instructions="Ask for compensation.", tools=None)
    state = user.get_init_state()
    before = deepcopy(state.system_messages)
    message, result = user.generate_next_message(DEFAULT_FIRST_AGENT_MESSAGE, state)

    assert message.content == accepted.content
    assert result is state
    assert state.messages == [DEFAULT_FIRST_AGENT_MESSAGE, message]
    assert state.system_messages == before
    assert len(attempts) == 2
    assert "No customer request has been delivered yet" in attempts[1][-1].content
    assert marker not in [m.content for m in state.messages]


@pytest.mark.parametrize("second", ["###OUT-OF-SCOPE###", "###STOP###", "", "Okay ###STOP###"])
def test_initial_invalid_outputs_share_two_attempt_budget(monkeypatch, second):
    from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE

    generate = Mock(spec=[], side_effect=[reply("###OUT-OF-SCOPE###"), reply(second)])
    monkeypatch.setattr(
        user_simulator, "generate", env_module._wrap_tau2_generate_with_rate_limit_retry(generate)
    )
    user = UserSimulator(llm="offline", instructions="Ask for compensation.", tools=None)
    state = user.get_init_state()
    with pytest.raises(env_module.Tau2UserSimulatorInitializationError, match="after 2 attempts"):
        user.generate_next_message(DEFAULT_FIRST_AGENT_MESSAGE, state)
    assert generate.call_count == 2
    assert state.messages == [DEFAULT_FIRST_AGENT_MESSAGE]


def test_initial_valid_customer_message_does_not_retry():
    generate = Mock(spec=[], return_value=reply("Please cancel my flight."))
    wrapped = env_module._wrap_tau2_generate_with_rate_limit_retry(generate)
    assert (
        wrapped(
            model="offline",
            messages=[SystemMessage(role="system", content="Customer scenario")],
            call_name="user_simulator_response",
        ).content
        == "Please cancel my flight."
    )
    generate.assert_called_once()


def test_initial_failure_survives_gym_thread_error_capture(failing_gym):
    env, generate, state, _ = failing_gym
    state.messages.clear()
    generate.return_value = reply("###OUT-OF-SCOPE###")
    # The fake Gym starts with its existing greeting; no user reply has been committed.
    with pytest.raises(env_module.Tau2UserSimulatorInitializationError, match="after 2 attempts"):
        env.reset()
    assert generate.call_count == 2
    assert env.execution_error["type"] == "Tau2UserSimulatorInitializationError"
    assert not any(isinstance(m, UserMessage) for m in state.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vikingbot", "native"])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_unknown_errors_and_cancellation_are_not_isolated(monkeypatch, backend, error_type):
    if backend == "vikingbot":
        from benchmark.tau2.train.rollout_executor_vikingbot import VikingBotTau2RolloutExecutor

        executor = VikingBotTau2RolloutExecutor()
        monkeypatch.setattr(
            type(executor), "_execute_one_async", AsyncMock(side_effect=error_type())
        )
    else:
        from benchmark.tau2.train.rollout_executor_native import NativeTau2RolloutExecutor

        executor = NativeTau2RolloutExecutor()
        monkeypatch.setattr(type(executor), "_execute_one_sync", Mock(side_effect=error_type()))
    with pytest.raises(error_type):
        await executor.execute(
            [case()],
            ExperienceSet("viking://user/default/memories/experiences", []),
            ExecutionContext("snapshot", {"training": False, "stage": "train_rollout"}),
        )
