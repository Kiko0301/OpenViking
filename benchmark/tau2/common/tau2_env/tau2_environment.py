#!/usr/bin/env python3

from __future__ import annotations

import importlib
import json
import os
import time
from collections.abc import Callable
from functools import wraps
from typing import Any
from uuid import uuid4

from openviking.session.train.components.dataset_service import redact_sensitive
from openviking.telemetry import tracer
from openviking.utils.model_retry import is_retryable_rate_limit_error, rate_limit_retry_delay
from openviking_cli.utils import get_logger

logger = get_logger(__name__)

DEFAULT_TAU2_USER_LLM = "openai/doubao-seed-2-0-code-preview-260215"
_TAU2_OUTPUT_MAX_ATTEMPTS = 3
_TAU2_INITIAL_OUTPUT_MAX_ATTEMPTS = 2
_TAU2_USER_OUTPUT_RULE = (
    "Output either ordinary customer text without control markers, or exactly one of "
    "###STOP###, ###TRANSFER###, ###OUT-OF-SCOPE###. Never combine a marker with text "
    "or tool calls. Authorizing a pending action is not completion: reply normally "
    "without STOP, and wait for the agent to report the result."
)
_TAU2_USER_OUTPUT_CORRECTION = (
    "Your previous output mixed a control marker with text or tool calls. "
    "Regenerate this user reply following the output rule."
)


class Tau2UserSimulatorError(ValueError):
    """A known simulator output failure, isolated from other rollouts."""


class Tau2UserSimulatorFormatError(Tau2UserSimulatorError):
    """The user simulator exhausted its current-turn output-format budget."""


class Tau2UserSimulatorInitializationError(Tau2UserSimulatorError):
    """The simulator failed to produce an opening message after one retry."""


_GYM_INITIAL_OBSERVATION_TIMEOUT_SECONDS = 120.0
_GYM_INITIAL_OBSERVATION_POLL_SECONDS = 0.01
_TAU2_DONE_RULE = (
    "Finish only after the user conversation has ended. After completing operations, "
    "report the results via communicate_with_user and handle the user's follow-up requests. "
    "Do not call done while the user conversation is still active."
)

_TAU2_GENERATE_REFERENCE_MODULES = (
    "tau2.agent.llm_agent",
    "tau2.user.user_simulator",
    "tau2.evaluator.evaluator_nl_assertions",
    "tau2.environment.utils.interface_agent",
)


def _is_tau2_retryable_rate_limit_error(exc: BaseException) -> bool:
    return is_retryable_rate_limit_error(exc)


def _tau2_rate_limit_retry_delay(attempt: int) -> float:
    return rate_limit_retry_delay(attempt)


def _tau2_reasoning_only_diagnostics(response: Any) -> dict[str, Any] | None:
    """Identify thinking without a usable reply in Tau2's normalized LiteLLM result."""
    if (
        (getattr(response, "content", None) or "").strip()
        or (getattr(response, "audio_content", None) or "").strip()
        or getattr(response, "tool_calls", None)
    ):
        return None
    raw = getattr(response, "raw_data", None) or {}
    choices = raw.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    message = choice.get("message") or {}
    if choice.get("finish_reason") == "content_filter" or message.get("refusal"):
        return None
    reasoning = message.get("reasoning_content") or ""
    thinking_blocks = message.get("thinking_blocks") or []
    if not reasoning.strip() and not thinking_blocks:
        return None
    return {
        "response_id": raw.get("id"),
        "model": raw.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "reasoning_chars": len(reasoning),
        "thinking_block_count": len(thinking_blocks),
        "usage": raw.get("usage"),
    }


def _wrap_tau2_generate_with_rate_limit_retry(generate: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(generate, "_openviking_tau2_rate_limit_retry", False):
        return generate

    @wraps(generate)
    def generate_with_rate_limit_retry(*args: Any, **kwargs: Any) -> Any:
        user_call = kwargs.get("call_name") == "user_simulator_response"
        # Tau2 flips customer replies to assistant messages in the simulator's input.
        # The default agent greeting is a user message; system prompts alone are not
        # sufficient to identify startup. Existing conversation history must be respected.
        initial_user_turn = user_call and not any(
            message.role == "assistant" for message in kwargs["messages"]
        )
        max_outputs = (
            _TAU2_INITIAL_OUTPUT_MAX_ATTEMPTS if initial_user_turn else _TAU2_OUTPUT_MAX_ATTEMPTS
        )
        if user_call:
            from tau2.data_model.message import SystemMessage

            # This is a per-call copy: retries must not append to UserState history.
            user_messages = [
                *kwargs["messages"],
                SystemMessage(role="system", content=_TAU2_USER_OUTPUT_RULE),
            ]
            kwargs = {**kwargs, "messages": user_messages}
        attempt = 1
        invalid_outputs = 0
        last_format_error = None
        while True:
            try:
                response = generate(*args, **kwargs)
            except Exception as exc:
                if not _is_tau2_retryable_rate_limit_error(exc):
                    raise
                delay = _tau2_rate_limit_retry_delay(attempt)
                logger.warning(
                    "tau2 LiteLLM generate rate limited; retrying attempt=%d delay=%.1fs error=%s",
                    attempt,
                    delay,
                    exc,
                )
                time.sleep(delay)
                attempt += 1
                continue

            # Both output failures share one budget; nested retry loops could multiply it.
            diagnostics = _tau2_reasoning_only_diagnostics(response)
            format_error = _tau2_user_format_diagnostics(response) if user_call else None
            if format_error is not None:
                last_format_error = format_error
                diagnostics = format_error
                kwargs = {
                    **kwargs,
                    "messages": [
                        *user_messages,
                        SystemMessage(role="system", content=_TAU2_USER_OUTPUT_CORRECTION),
                    ],
                }
            if initial_user_turn:
                from tau2.user.user_simulator import OUT_OF_SCOPE, STOP, TRANSFER

                text = (response.content or "").strip()
                if not response.tool_calls and (
                    text in (STOP, TRANSFER, OUT_OF_SCOPE)
                    or (not text and not getattr(response, "audio_content", None))
                ):
                    diagnostics = {
                        "reason": "initial_user_terminated"
                        if text
                        else "empty_initial_user_message",
                        "stop_marker": text or None,
                    }
                if diagnostics is not None:
                    kwargs = {
                        **kwargs,
                        "messages": [
                            *user_messages,
                            SystemMessage(
                                role="system",
                                content=(
                                    "No customer request has been delivered yet. Begin the "
                                    "conversation using the supplied scenario, without inventing "
                                    "missing facts. A request the agent may reject is still a "
                                    "request to make; do not end the simulation before asking it."
                                ),
                            ),
                        ],
                    }
            if diagnostics is None:
                return response
            invalid_outputs += 1
            detail = json.dumps(diagnostics, ensure_ascii=False)
            tracer.info(f"tau2 invalid model output attempt={invalid_outputs} diagnostics={detail}")
            if invalid_outputs >= max_outputs:
                if initial_user_turn:
                    error = Tau2UserSimulatorInitializationError(
                        "tau2 user simulator produced no valid opening message "
                        f"after {invalid_outputs} attempts; diagnostics={detail}"
                    )
                elif last_format_error is not None:
                    error = Tau2UserSimulatorFormatError(
                        "tau2 user simulator returned invalid stop format "
                        f"after {invalid_outputs} attempts; "
                        f"diagnostics={json.dumps(last_format_error, ensure_ascii=False)}"
                    )
                else:
                    error = ValueError(
                        "tau2 LiteLLM returned reasoning-only responses without content or tool calls "
                        f"after {invalid_outputs} attempts; diagnostics={detail}"
                    )
                tracer.error(str(error), e=error, console=False)
                raise error
            if format_error is None and not initial_user_turn:
                time.sleep(_tau2_rate_limit_retry_delay(invalid_outputs))

    generate_with_rate_limit_retry._openviking_tau2_rate_limit_retry = True
    generate_with_rate_limit_retry._openviking_original_generate = generate
    return generate_with_rate_limit_retry


def _tau2_user_format_diagnostics(response: Any) -> dict[str, Any] | None:
    from tau2.user.user_simulator import OUT_OF_SCOPE, STOP, TRANSFER

    text = response.content or ""
    markers = [marker for marker in (STOP, TRANSFER, OUT_OF_SCOPE) if marker in text]
    if not markers or (text.strip() in markers and response.tool_calls is None):
        return None
    preview = redact_sensitive(text)
    return {
        "content": preview[:2000],
        "content_chars": len(text),
        "truncated": len(preview) > 2000,
        "markers": markers,
        "tool_call_count": len(response.tool_calls or []),
    }


def _tau2_user_is_stop(cls: Any, message: Any) -> bool:
    from tau2.user.user_simulator import OUT_OF_SCOPE, STOP, TRANSFER

    return not message.is_tool_call() and (message.content or "").strip() in (
        STOP,
        TRANSFER,
        OUT_OF_SCOPE,
    )


def _install_tau2_litellm_rate_limit_retry() -> None:
    """Retry rate limits and invalid outputs in Tau2's sync generate path.

    AgentGymEnv's user simulator and orchestrator call tau2.utils.llm_utils.generate
    through synchronous module globals imported with ``from ... import generate``.
    Those calls run in tau2's own worker thread, so a sync sleep-based retry is
    safe and does not block the OpenViking service event loop.
    """
    try:
        llm_utils = importlib.import_module("tau2.utils.llm_utils")
    except Exception as exc:
        logger.debug("tau2 llm_utils unavailable for rate-limit retry patch: %s", exc)
        return

    original = getattr(llm_utils, "_openviking_original_generate", None)
    current = getattr(llm_utils, "generate", None)
    if not callable(current):
        return
    if getattr(current, "_openviking_tau2_rate_limit_retry", False):
        wrapped = current
        original = getattr(current, "_openviking_original_generate", original)
    else:
        original = current
        wrapped = _wrap_tau2_generate_with_rate_limit_retry(original)
        llm_utils.generate = wrapped
        llm_utils._openviking_original_generate = original

    for module_name in _TAU2_GENERATE_REFERENCE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        module_generate = getattr(module, "generate", None)
        if (
            module_generate is original
            or module_generate is current
            or getattr(module_generate, "_openviking_tau2_rate_limit_retry", False)
        ):
            module.generate = wrapped
        if module_name == "tau2.user.user_simulator":
            module.UserSimulator.is_stop = classmethod(_tau2_user_is_stop)


def _install_tau2_litellm_unknown_cost_suppression() -> None:
    """Suppress noisy LiteLLM cost lookup errors for private/proxy model names.

    tau2-bench logs an ERROR whenever LiteLLM cannot find a public price entry
    for models such as Doubao private gateway names. Cost accounting is not used
    by our rollout evaluator, and at high concurrency those repeated ERROR logs
    add significant IO noise. Replace cost lookup failures with a zero-cost
    result while preserving normal behavior for mapped models.
    """

    try:
        llm_utils = importlib.import_module("tau2.utils.llm_utils")
    except Exception as exc:
        logger.debug("tau2 llm_utils unavailable for cost suppression patch: %s", exc)
        return

    for name in ("get_response_cost", "get_cost"):
        current = getattr(llm_utils, name, None)
        if not callable(current) or getattr(current, "_openviking_tau2_cost_suppressed", False):
            continue

        @wraps(current)
        def suppressed_cost(*args: Any, __fn: Callable[..., Any] = current, **kwargs: Any) -> Any:
            try:
                return __fn(*args, **kwargs)
            except Exception as exc:
                text = str(exc)
                if "model isn't mapped" not in text and "not mapped" not in text:
                    raise
                logger.debug("suppressed tau2 LiteLLM cost lookup failure: %s", exc)
                return 0.0

        suppressed_cost._openviking_tau2_cost_suppressed = True
        suppressed_cost._openviking_original_cost_fn = current
        setattr(llm_utils, name, suppressed_cost)


try:
    from tau2.gym.gym_agent import AgentGymEnv
except ModuleNotFoundError:
    AgentGymEnv = None


class CommunicateWithUser:
    """The agent's only channel for talking to the user.

    tau2's environment has no native "speak to the user" action, so we add this
    tool: whatever ``content`` the agent passes is delivered to the tau2 user
    simulator, and the simulator's reply comes back as the tool result. This class
    owns both the tool's schema (``openai_schema``, consumed by
    ``Tau2BenchToolProvider``) and its execution (``forward``, invoked by
    ``Tau2BenchEnv.tool_call``).
    """

    name = "communicate_with_user"
    description = (
        "say something to the user. Note that the customer cannot see the answer "
        "returned in `final_answer`. You must communicate with the customer "
        "exclusively through this tool."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "the content to say to the user",
            }
        },
        "required": ["content"],
    }

    def __init__(self, env):
        # ``env`` is the underlying tau2 AgentGymEnv.
        self.env = env

    def forward(self, content: str):
        """Deliver ``content`` to the tau2 user simulator.

        Returns the raw gym step tuple ``(obs, reward, terminated, truncated, info)``;
        ``Tau2BenchEnv.tool_call`` cleans the observation and tracks termination.
        """
        response = self.env.tool_call(self.name, {"content": content})
        return response

    @classmethod
    def openai_schema(cls) -> dict:
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": cls.parameters,
            },
        }


class Tau2BenchEnv:
    def __init__(self, domain: str, task_id: str):
        if AgentGymEnv is not None:
            self._impl = _GymTau2BenchEnv(domain, task_id)
        else:
            self._impl = _NativeTau2BenchEnv(domain, task_id)

    @property
    def execution_error(self) -> dict[str, str] | None:
        return self._impl.execution_error

    def reset(self):
        self._impl.reset()
        self.env = self._impl.env
        self.terminated = self._impl.terminated
        self.user_query = self._impl.user_query
        self.task = self._impl.task
        self.simulation_run = self._impl.simulation_run
        self.policy = self._impl.policy
        self.tool_schemas = self._impl.tool_schemas
        self.ground_truth = self._impl.ground_truth
        self.user_scenario = self._impl.user_scenario

    def tool_call(self, tool_name: str, arguments: dict) -> str:
        response = self._impl.tool_call(tool_name, arguments)
        self.terminated = self._impl.terminated
        return response

    def append_agent_message(self, content: str) -> None:
        append_message = getattr(self._impl, "append_agent_message", None)
        if callable(append_message):
            append_message(content)

    def _get_reward(self):
        return self._impl._get_reward()


class _GymTau2BenchEnv:
    def __init__(self, domain: str, task_id: str):
        _install_tau2_litellm_rate_limit_retry()
        _install_tau2_litellm_unknown_cost_suppression()
        self.env = AgentGymEnv(
            domain=domain,
            task_id=task_id,
            user_llm=os.getenv("TAU2_USER_LLM") or DEFAULT_TAU2_USER_LLM,
        )
        self.terminated = False
        self._create_orchestrator = self.env._get_orchestrator
        self.env._get_orchestrator = self._get_orchestrator

    @property
    def execution_error(self) -> dict[str, str] | None:
        return getattr(self.env._orchestrator, "_openviking_execution_error", None)

    def _get_orchestrator(self):
        orchestrator = self._create_orchestrator()
        run = orchestrator.run
        orchestrator._openviking_execution_error = None

        @wraps(run)
        def run_with_error_tracking(*args, **kwargs):
            try:
                return run(*args, **kwargs)
            except Exception as exc:
                # Gym catches this exception and otherwise exposes an ordinary zero reward.
                # Keep the marker on this instance so concurrent rollouts/resets cannot mix it.
                orchestrator._openviking_execution_error = {
                    "source": "tau2_simulator",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                tracer.error("Tau2 simulator execution failed", e=exc, console=False)
                raise

        orchestrator.run = run_with_error_tracking
        return orchestrator

    def reset(self):
        user_query, info_dict = self.env.reset()
        try:
            user_query = _wait_for_gym_initial_observation(self.env, user_query)
        except RuntimeError as exc:
            details = json.dumps(_gym_initial_observation_diagnostics(self.env), ensure_ascii=False)
            for error_type in (Tau2UserSimulatorFormatError, Tau2UserSimulatorInitializationError):
                if (self.execution_error or {}).get("type") == error_type.__name__:
                    raise error_type(f"{exc}; diagnostics={details}") from exc
            raise RuntimeError(f"{exc}; diagnostics={details}") from exc
        self.user_query = user_query.lstrip("user: ")
        self.task = info_dict["task"]
        self.simulation_run = info_dict["simulation_run"]
        self.policy = info_dict["policy"]
        self.tool_schemas = [tool.openai_schema for tool in info_dict["tools"]]
        for schema in self.tool_schemas:
            if schema["function"]["name"] == "done":
                schema["function"]["description"] = _TAU2_DONE_RULE
        self.tool_schemas.append(CommunicateWithUser.openai_schema())
        self.ground_truth = str(self.task.evaluation_criteria)
        self.user_scenario = self.task.user_scenario

    def tool_call(self, tool_name: str, arguments: dict) -> str:
        if self.terminated:
            return "Task Terminated"
        if tool_name == "done":
            # Reuse Gym's terminal state; a write result is not a finished conversation.
            # Error results keep AgentLoop running instead of triggering its stop-tool path.
            return f"Error: USER_CONVERSATION_ACTIVE. {_TAU2_DONE_RULE}"

        if tool_name == CommunicateWithUser.name:
            obs, reward, terminated, truncated, info = self.env.step(arguments["content"])
        else:
            action = {"name": tool_name, "arguments": arguments}
            obs, reward, terminated, truncated, info = self.env.step(json.dumps(action))

        self.terminated = terminated
        if terminated and self.execution_error is not None:
            return "Task Terminated"
        return _clean_obs(obs)

    def append_agent_message(self, content: str) -> None:
        if not content.strip():
            return
        simulation = self._simulation_run_from_env_info()
        if simulation is None:
            return

        from tau2.data_model.message import AssistantMessage

        simulation.messages.append(AssistantMessage(role="assistant", content=content))
        self.env._simulation_run = simulation
        self.simulation_run = simulation.model_dump_json(indent=2)

    def _get_reward(self):
        reward, reward_info = self.env._get_reward()
        try:
            return reward, json.loads(reward_info)
        except (TypeError, json.JSONDecodeError):
            return reward, reward_info

    def _simulation_run_from_env_info(self):
        simulation_run_json = self.env._get_info().get("simulation_run")
        if not simulation_run_json:
            return None
        try:
            from tau2.data_model.simulation import SimulationRun

            return SimulationRun.model_validate_json(simulation_run_json)
        except Exception:
            return None


def _wait_for_gym_initial_observation(env: Any, observation: Any) -> str:
    """Bound first-message publication after reset returns, not reset/step themselves."""
    initial = str(observation or "")
    if initial.strip():
        return initial

    agent = getattr(env, "_agent", None)
    agent_lock = getattr(agent, "_lock", None)
    format_observation = getattr(env, "_format_observation", None)
    simulation_done = getattr(env, "_simulation_done", None)
    if agent is None or agent_lock is None or not callable(format_observation):
        raise RuntimeError("Tau2 AgentGymEnv returned an empty initial observation")

    deadline = time.monotonic() + _GYM_INITIAL_OBSERVATION_TIMEOUT_SECONDS
    while True:
        if simulation_done is not None and simulation_done.is_set():
            raise RuntimeError("Tau2 AgentGymEnv ended before publishing its initial observation")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Tau2 AgentGymEnv timed out waiting for its initial observation")

        acquired = agent_lock.acquire(timeout=remaining)
        if not acquired:
            raise RuntimeError("Tau2 AgentGymEnv timed out waiting for its initial observation")
        try:
            candidate = str(format_observation(list(agent.observation)) or "")
        finally:
            agent_lock.release()
        if candidate.strip():
            return candidate

        time.sleep(min(_GYM_INITIAL_OBSERVATION_POLL_SECONDS, remaining))


def _gym_initial_observation_diagnostics(env: Any) -> dict[str, Any]:
    """Snapshot the user side only; never log prompts, raw responses or tool arguments."""
    from tau2.user.user_simulator import OUT_OF_SCOPE, STOP, TRANSFER

    def preview(text: str, limit: int = 2000) -> str:
        text = redact_sensitive(text)  # Redact before clipping, including long quoted credentials.
        return text[:limit] + ("...[truncated]" if len(text) > limit else "")

    orchestrator = getattr(env, "_orchestrator", None)
    user_state = getattr(orchestrator, "user_state", None)
    # UserSimulator appends a reply here before validation can reject it. Such a
    # reply never reaches the orchestrator trajectory or the final SimulationRun.
    messages = list(getattr(user_state, "messages", ()))
    last_user = next((message for message in reversed(messages) if message.role == "user"), None)
    last_user_message = None
    if last_user is not None:
        content = last_user.content or ""
        tool_calls = last_user.tool_calls or []
        last_user_message = {
            "content": preview(content) if last_user.content is not None else None,
            "content_chars": len(content),
            "stop_markers": [
                marker for marker in (STOP, TRANSFER, OUT_OF_SCOPE) if marker in content
            ],
            "tool_calls": [preview(call.name, 100) for call in tool_calls[:10]],
            "tool_call_count": len(tool_calls),
        }

    execution_error = getattr(orchestrator, "_openviking_execution_error", None)
    if execution_error is not None:
        execution_error = {**execution_error, "message": preview(execution_error["message"])}
    return {
        "termination_reason": getattr(orchestrator, "termination_reason", None),
        "step_count": getattr(orchestrator, "step_count", None),
        "num_errors": getattr(orchestrator, "num_errors", None),
        "last_user_message": last_user_message,
        "execution_error": execution_error,
    }


class _NativeTau2BenchEnv:
    # This adapter has no background simulator whose errors can be swallowed by Gym.
    execution_error: dict[str, str] | None = None

    def __init__(self, domain: str, task_id: str):
        self.domain = domain
        self.task_id = task_id
        self.env = None
        self.terminated = False
        self.simulation_run = None

    def reset(self):
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
        from tau2.registry import registry

        self._evaluate_simulation = evaluate_simulation
        self._evaluation_type = EvaluationType.ALL
        self.env = registry.get_env_constructor(self.domain)()
        tasks = registry.get_tasks_loader(self.domain)()
        task_by_id = {str(task.id): task for task in tasks}
        self.task = task_by_id[self.task_id]
        self.env.set_state(
            initialization_data=(
                self.task.initial_state.initialization_data
                if self.task.initial_state is not None
                else None
            ),
            initialization_actions=(
                self.task.initial_state.initialization_actions
                if self.task.initial_state is not None
                else None
            ),
            message_history=(
                self.task.initial_state.message_history
                if self.task.initial_state is not None
                and self.task.initial_state.message_history is not None
                else []
            ),
        )
        self.policy = self.env.get_policy()
        self.tool_schemas = [tool.openai_schema for tool in self.env.get_tools()]
        self.tool_schemas.append(CommunicateWithUser.openai_schema())
        self.user_query = str(self.task.user_scenario)
        self.ground_truth = str(self.task.evaluation_criteria)
        self.user_scenario = self.task.user_scenario
        self._messages = []

    def tool_call(self, tool_name: str, arguments: dict) -> str:
        from tau2.data_model.message import AssistantMessage, ToolCall

        if self.terminated:
            return "Task Terminated"

        if tool_name == CommunicateWithUser.name:
            # tau2 evaluates required customer-facing information by scanning
            # AssistantMessage text content. Record this synthetic communication
            # as assistant text so the native fallback matches gym trajectories.
            self._messages.append(
                AssistantMessage(role="assistant", content=str(arguments["content"]))
            )
            return (
                "User simulator is unavailable in this tau2 version; "
                "continue using tools and final answer."
            )

        tool_call = ToolCall(
            id=f"call_{uuid4().hex}",
            name=tool_name,
            arguments=arguments,
            requestor="assistant",
        )
        assistant_message = AssistantMessage(role="assistant", tool_calls=[tool_call])
        tool_message = self.env.get_response(tool_call)
        self._messages.extend([assistant_message, tool_message])
        return _clean_obs(tool_message.content or "")

    def append_agent_message(self, content: str) -> None:
        from tau2.data_model.message import AssistantMessage

        if content.strip():
            self._messages.append(AssistantMessage(role="assistant", content=content))

    def _get_reward(self):
        from tau2.data_model.simulation import SimulationRun
        from tau2.utils.utils import get_now

        now = get_now()
        simulation = SimulationRun(
            id=f"native_tau2_{self.domain}_{self.task_id}_{uuid4().hex}",
            task_id=self.task.id,
            start_time=now,
            end_time=now,
            duration=0.0,
            termination_reason="agent_stop",
            reward_info=None,
            messages=self._messages,
        )
        reward_info = self._evaluate_simulation(
            domain=self.domain,
            task=self.task,
            simulation=simulation,
            evaluation_type=self._evaluation_type,
            solo_mode=False,
        )
        simulation.reward_info = reward_info
        self.simulation_run = simulation
        return reward_info.reward, reward_info


def _clean_obs(obs: str) -> str:
    if "tool: " in obs:
        obs = obs.removeprefix("tool: ")
    if "user: " in obs:
        obs = obs.removeprefix("user: ")
    return obs
