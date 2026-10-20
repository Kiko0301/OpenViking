# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_memory_create_guard import CTX, MemoryFS
from test_memory_create_guard import schema as record_schema

from openviking.session.memory.dataclass import MemoryFile, ResolvedOperation, ResolvedOperations
from openviking.session.memory.experience.compiler import compile_dag
from openviking.session.memory.extract_loop import ExtractLoop
from openviking.session.memory.memory_type_registry import (
    MemoryTypeRegistry,
    resolve_memory_templates_dir,
)
from openviking.session.memory.memory_updater import ExtractContext, MemoryUpdater
from openviking.session.memory.merge_op import (
    DeleteBlock,
    FieldType,
    PatchOp,
    SearchReplaceBlock,
    StrPatch,
)
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.memory.validation import content as content_gate
from tests.session.train.test_fakes import render_trajectory_template

URI = "viking://user/alice/memories/experiences/case_a.md"
OTHER = "viking://user/alice/memories/records/a.md"
SOURCE = """dag = workflow("review")
condition = check("Current request applies")
reminder = tell("old reminder")
condition.then(reminder)"""


def registry():
    value = MemoryTypeRegistry(load_schemas=False)
    value.load_from_yaml(str(resolve_memory_templates_dir() / "experiences.yaml"))
    value.register(record_schema())
    return value


def file(source=SOURCE, uri=URI):
    return MemoryFile(
        uri=uri,
        memory_type="experiences",
        content=source,
        extra_fields={"experience_name": "case_a", "version": 1},
    )


def operation(content, old=None, uri=URI, memory_type="experiences", create_only=False):
    return ResolvedOperation(
        memory_type=memory_type,
        uris=[uri],
        page_id=1,
        memory_fields={"content": content, "experience_name": "case_a"},
        old_memory_file_content=old,
        create_only=create_only,
    )


def batch(*ops, deletes=()):
    return ResolvedOperations(
        upsert_operations=list(ops), delete_file_contents=list(deletes), errors=[]
    )


def patch(search="old reminder", replace="new reminder"):
    return StrPatch(blocks=[SearchReplaceBlock(search=search, replace=replace)])


class Store(MemoryFS):
    def __init__(self, files=None):
        super().__init__(files)
        self.writes = []
        self.deletes = []
        self.on_lock = None

    async def acquire(self, paths, **kwargs):
        lease = await super().acquire(paths, **kwargs)
        if self.on_lock:
            self.on_lock()
            self.on_lock = None
        return lease

    async def write_file(self, uri, content, **kwargs):
        await super().write_file(uri, content, **kwargs)
        self.writes.append(uri)

    async def rm(self, uri, **kwargs):
        assert self.lock.locked()
        self.deletes.append(uri)
        self.files.pop(uri, None)


def updater(fs):
    result = MemoryUpdater(registry=registry())
    result._viking_fs = fs
    result._vectorize_memories = AsyncMock(return_value=0)
    result.generate_overview = AsyncMock()
    result._sync_resource_refs_for_result = AsyncMock()
    return result


def loop():
    value = object.__new__(ExtractLoop)
    value.ctx = CTX
    value._extract_context = ExtractContext([])
    schemas = registry().list_all()
    value.context_provider = SimpleNamespace(get_memory_schemas=lambda _: schemas)
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("as_dict", [False, True])
async def test_patch_is_materialized_before_gate_and_preserved_until_locked_apply(as_dict):
    old = file()
    update = patch()
    update = update.model_dump() if as_dict else update
    op = operation(update, old)
    ops = batch(op)
    assert await loop()._validate_content_gates(ops) == []
    assert op.memory_fields["content"] is update  # Never flatten a patch before the lock.
    fs = Store({URI: MemoryFileUtils.write(old)})
    result = await updater(fs).apply_operations(ops, CTX)
    assert not result.errors
    stored = MemoryFileUtils.read(fs.files[URI], uri=URI)
    assert stored.content == SOURCE.replace("old reminder", "new reminder")
    assert stored.extra_fields["version"] == 2
    assert compile_dag(stored.content)
    assert not fs.lock.locked()


@pytest.mark.asyncio
async def test_latest_locked_content_preserves_concurrent_unrelated_change():
    old = file()
    latest = SOURCE.replace('"Current request applies"', '"Updated applicability"')
    fs = Store({URI: MemoryFileUtils.write(old)})
    fs.on_lock = lambda: fs.files.update({URI: MemoryFileUtils.write(file(latest))})
    result = await updater(fs).apply_operations(batch(operation(patch(), old)), CTX)
    assert not result.errors
    content = MemoryFileUtils.read(fs.files[URI], uri=URI).content
    assert "Updated applicability" in content and "new reminder" in content


@pytest.mark.asyncio
async def test_latest_content_gate_catches_invalid_graph_despite_valid_preview():
    original = 'dag = workflow("review")\ncondition = check("ready")\nreminder = tell("old")'
    update = patch('reminder = tell("old")', 'reminder = tell("new")\ncondition.then(reminder)')
    op = operation(update, file(original))
    assert await loop()._validate_content_gates(batch(op)) == []
    deleted = file(uri=URI.replace("case_a.md", "obsolete.md"))
    fs = Store(
        {
            URI: MemoryFileUtils.write(file(original)),
            OTHER: "keep this",
            deleted.uri: MemoryFileUtils.write(deleted),
        }
    )
    latest = 'dag = workflow("review")\nreminder = tell("old")'
    fs.on_lock = lambda: fs.files.update({URI: MemoryFileUtils.write(file(latest))})
    result = await updater(fs).apply_operations(
        batch(
            operation("changed", memory_type="records", uri=OTHER),
            op,
            deletes=[deleted],
        ),
        CTX,
    )
    assert result.errors
    assert fs.writes == [] and fs.deletes == []
    assert MemoryFileUtils.read(fs.files[URI], uri=URI).content == latest
    assert not fs.lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "latest",
    [
        SOURCE.replace("old reminder", "almost old reminder!"),
        SOURCE + '\nother = tell("old reminder")',
    ],
)
async def test_stale_or_ambiguous_anchor_does_not_use_fuzzy_matching(latest):
    # Match the full declaration, so a changed string cannot match as a substring.
    update = patch('reminder = tell("old reminder")', 'reminder = tell("new reminder")')
    if "other =" in latest:
        update = patch("old reminder", "new reminder")
    fs = Store({URI: MemoryFileUtils.write(file(latest))})
    before = dict(fs.files)
    result = await updater(fs).apply_operations(batch(operation(update, file())), CTX)
    assert result.errors and "exactly once" in str(result.errors[0][1])
    assert fs.files == before and fs.writes == []
    assert not fs.lock.locked()


@pytest.mark.asyncio
async def test_invalid_candidate_blocks_unrelated_write_and_dependent_delete():
    deleted = file(uri=URI.replace("case_a.md", "obsolete.md"))
    fs = Store(
        {
            URI: MemoryFileUtils.write(file()),
            OTHER: "keep",
            deleted.uri: MemoryFileUtils.write(deleted),
        }
    )
    result = await updater(fs).apply_operations(
        batch(
            operation("changed", memory_type="records", uri=OTHER),
            operation('dag = workflow("broken")', file()),
            deletes=[deleted],
        ),
        CTX,
    )
    assert result.errors
    assert not fs.writes and not fs.deletes


@pytest.mark.asyncio
async def test_delete_patch_checks_the_complete_result_and_allows_safe_node_removal():
    source = SOURCE + '\nextra = tell("Unrelated reminder")'
    good = StrPatch(blocks=[DeleteBlock(delete='extra = tell("Unrelated reminder")')])
    fs = Store({URI: MemoryFileUtils.write(file(source))})
    result = await updater(fs).apply_operations(batch(operation(good, file(source))), CTX)
    assert not result.errors
    assert MemoryFileUtils.read(fs.files[URI], uri=URI).content == SOURCE
    bad = StrPatch(blocks=[DeleteBlock(delete='reminder = tell("old reminder")')])
    result = await updater(fs).apply_operations(batch(operation(bad, file())), CTX)
    assert result.errors
    assert MemoryFileUtils.read(fs.files[URI], uri=URI).content == SOURCE


@pytest.mark.asyncio
async def test_sequential_patches_share_a_staged_view_before_any_write():
    fs = Store({URI: MemoryFileUtils.write(file())})
    result = await updater(fs).apply_operations(
        batch(
            operation(patch(replace="intermediate"), file()),
            operation(patch(search="intermediate", replace="final"), file()),
        ),
        CTX,
    )
    assert not result.errors
    assert MemoryFileUtils.read(fs.files[URI], uri=URI).content == SOURCE.replace(
        "old reminder", "final"
    )


@pytest.mark.asyncio
async def test_erased_content_is_not_validated_using_the_old_graph():
    content = StrPatch(blocks=[DeleteBlock(delete=SOURCE)])
    fs = Store({URI: MemoryFileUtils.write(file())})
    result = await updater(fs).apply_operations(batch(operation(content, file())), CTX)
    assert result.errors and not fs.writes


@pytest.mark.asyncio
async def test_missing_or_unreadable_latest_file_never_falls_back_to_prefetch():
    for error in [FileNotFoundError(URI), PermissionError("denied")]:
        fs = Store()
        fs.read_file = AsyncMock(side_effect=error)
        result = await updater(fs).apply_operations(batch(operation(patch(), file())), CTX)
        assert result.errors and not fs.writes
        assert not fs.lock.locked()


@pytest.mark.asyncio
async def test_new_memory_and_full_replacement_use_same_gate():
    fs = Store()
    result = await updater(fs).apply_operations(batch(operation(SOURCE, create_only=True)), CTX)
    assert not result.errors
    result = await updater(fs).apply_operations(
        batch(operation(SOURCE.replace("old", "new"), file())), CTX
    )
    assert not result.errors
    assert "new reminder" in MemoryFileUtils.read(fs.files[URI], uri=URI).content


@pytest.mark.asyncio
async def test_gate_dispatch_is_type_generic_and_independent_of_quality_gate(
    monkeypatch,
):
    calls = []

    def check(content):
        calls.append(content)
        if content != "safe":
            raise ValueError("record content rejected")
        return content

    monkeypatch.setitem(content_gate._FIELD_GATES, "records", {"content": check})
    fs = Store({OTHER: "before"})
    result = await updater(fs).apply_operations(
        batch(operation("unsafe", uri=OTHER, memory_type="records")), CTX
    )
    assert calls == ["unsafe"] and result.errors and not fs.writes
    assert content_gate.get_field_gates("experiences")["content"] is not None


@pytest.mark.asyncio
async def test_ordinary_and_checked_patch_share_pi_validation():
    # Every Pi edit batch is validated in full, even before a type-owned gate.
    merge = PatchOp(FieldType.STRING)
    content = "unchanged"
    empty = StrPatch(blocks=[SearchReplaceBlock(search="", replace="ignored")])
    with pytest.raises(ValueError, match="search must not be empty"):
        await merge.apply(content, empty)
    with pytest.raises(ValueError, match="search must not be empty"):
        await merge.apply_checked(content, empty)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", None])
async def test_empty_optional_patch_value_keeps_existing_merge_semantics(content):
    fs = Store({URI: MemoryFileUtils.write(file())})
    result = await updater(fs).apply_operations(batch(operation(content, file())), CTX)
    assert not result.errors
    assert MemoryFileUtils.read(fs.files[URI], uri=URI).content == SOURCE


@pytest.mark.asyncio
async def test_concurrent_disjoint_patches_preserve_both_updates():
    fs = Store({URI: MemoryFileUtils.write(file())})
    results = await asyncio.gather(
        updater(fs).apply_operations(batch(operation(patch(), file())), CTX),
        updater(fs).apply_operations(
            batch(operation(patch("Current request applies", "Updated condition"), file())), CTX
        ),
    )
    assert all(not result.errors for result in results)
    content = MemoryFileUtils.read(fs.files[URI], uri=URI).content
    assert "new reminder" in content and "Updated condition" in content
    assert not fs.lock.locked()


def extraction_case(monkeypatch, responses, *, source=SOURCE, output_format="python"):
    from openviking.session.memory.experience.experience_context import (
        AgentExperienceContextProvider,
    )
    from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler
    from openviking.session.train import Case, Rubric, Trajectory
    from openviking.session.train.components.trajectory_observation import (
        TRAJECTORY_FORMAT,
    )

    old = file(source)
    old.extra_fields["case_name"] = "case_a"
    fs = Store({URI: MemoryFileUtils.write(old)})
    provider = AgentExperienceContextProvider(
        messages=[],
        case=Case("case_a", "signature", {}, Rubric("reward", "", [])),
        target_uri=URI,
        source_session_uri="viking://session/s/history/archive_001",
        memory_registry=registry(),
        trajectories=[
            Trajectory(
                name="observation",
                uri="viking://user/alice/memories/trajectories/observation.md",
                outcome="failure",
                content="The old reminder needs correction",
                metadata={
                    "case_name": "case_a",
                    "source_session_uri": "viking://session/s/history/archive_001",
                },
            )
        ],
    )
    provider.read_file_contents[URI] = old
    provider.get_extract_context().page_id_map.get_page_id(URI)
    for trajectory in provider.trajectories:
        trajectory.metadata["trajectory_format"] = TRAJECTORY_FORMAT
        trajectory.content = render_trajectory_template(trajectory)
    provider._eval_memory = AsyncMock(
        return_value={
            "experience": {
                "uri": URI,
                "memory_type": "experiences",
                "experience_name": "case_a",
                "case_name": "case_a",
                "content": source,
            }
        }
    )
    isolation = MemoryIsolationHandler(
        CTX, provider.get_extract_context(), allowed_memory_types={"experiences"}
    )
    provider._isolation_handler = isolation
    model = SimpleNamespace(model="mock", get_completion_async=AsyncMock(side_effect=responses))
    extraction = ExtractLoop(
        vlm=model,
        viking_fs=fs,
        ctx=CTX,
        context_provider=provider,
        isolation_handler=isolation,
        max_iterations=1,
    )
    config = SimpleNamespace(
        memory=SimpleNamespace(link_enabled=False, extraction_output_format=output_format)
    )
    monkeypatch.setattr(
        "openviking.session.memory.extract_loop.get_openviking_config", lambda: config
    )
    monkeypatch.setattr("openviking_cli.utils.config.get_openviking_config", lambda: config)
    return extraction, model, fs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bad_edit", "expected_search", "expected_count"),
    [
        pytest.param(
            """drop(text='condition = check("Current request applies")')""",
            None,
            None,
            id="broken_graph",
        ),
        pytest.param(
            'edit(search="missing reminder", replace="new reminder")',
            "missing reminder",
            0,
            id="missing_anchor",
        ),
        pytest.param(
            'edit(search="reminder", replace="new reminder")',
            "reminder",
            3,
            id="ambiguous_anchor",
        ),
        pytest.param(
            r"""edit(search='reminder = tell(\n    "old reminder"\n)', replace="new reminder")""",
            'reminder = tell(\n    "old reminder"\n)',
            0,
            id="reformatted_anchor",
        ),
        pytest.param(
            r'edit(search="old reminder", replace="new\nreminder")',
            None,
            None,
            id="invalid_dag_string",
        ),
    ],
)
@pytest.mark.parametrize("repaired", [True, False])
async def test_extraction_repair_and_writer_share_the_content_gate(
    monkeypatch, bad_edit, expected_search, expected_count, repaired
):
    bad = f"experience.content.{bad_edit}\nsdk.commit()"
    good = 'experience.content.edit(search="old reminder", replace="new reminder")\nsdk.commit()'
    is_drop = bad_edit.startswith("drop(")
    repair_reply = json.dumps(
        {
            "status": "repaired",
            "repairs": [{"edit_index": 0, "search": "old reminder", "replace": "new reminder"}],
        }
        if repaired
        else {"status": "cannot_repair", "reason": "No safe correction"}
    )
    responses = [bad, (good if repaired else bad) if is_drop else repair_reply]
    if not is_drop and not repaired:
        responses.extend([bad, repair_reply])  # The regenerated Edit also gets local repair.
    extraction, model, fs = extraction_case(monkeypatch, responses)
    result, _ = await extraction.run()
    assert model.get_completion_async.await_count == len(responses)
    if not is_drop and not repaired:
        fallback = model.get_completion_async.await_args_list[2].kwargs["messages"]
        assert any("cannot_repair: No safe correction" in m["content"] for m in fallback)
    retry_messages = model.get_completion_async.await_args_list[1].kwargs["messages"]
    if is_drop:
        # Drop-only edits still use the original content-gate retry.
        assert {"role": "assistant", "content": bad} in retry_messages
    else:
        assert len(retry_messages) == 2
        assert all(message["role"] != "assistant" for message in retry_messages)
        payload = json.loads(retry_messages[1]["content"])
        target = payload["targets"][0]
        assert target["uri"] == URI and target["field"] == "content"
        assert target["original_text"] == SOURCE
        if expected_search is not None:
            assert target["edits"][0]["search"] == expected_search
            diagnostic = target["error"]
            assert f"{URI}#content" in diagnostic
            assert (
                f"Found {expected_count} occurrences"
                if expected_count
                else "Could not find the exact text"
            ) in diagnostic
        if expected_search == "missing reminder":
            details = next(error for error in payload["errors"] if "block_index" in error)
            assert details["block_index"] == 1
            assert details["reason"] == "not_found" and details["match_count"] == 0
    if not repaired:
        assert not result.upsert_operations
        assert MemoryFileUtils.read(fs.files[URI], uri=URI).content == SOURCE
        assert not fs.writes
        return
    assert isinstance(result.upsert_operations[0].memory_fields["content"], StrPatch)
    applied = await updater(fs).apply_operations(result, CTX)
    assert not applied.errors
    assert MemoryFileUtils.read(fs.files[URI], uri=URI).content == SOURCE.replace(
        "old reminder", "new reminder"
    )


@pytest.mark.asyncio
async def test_case_rubric_gate_validates_patched_field_without_requiring_content():
    from openviking.session.compressor_v3 import _memory_file_to_case
    from openviking.session.memory.experience.rubric import normalize_case_rubric

    reg = registry()
    reg.load_from_yaml(str(resolve_memory_templates_dir() / "cases.yaml"))
    uri = "viking://user/alice/memories/cases/library.md"
    before = normalize_case_rubric([{"id": "scope", "description": "Respect scope."}])
    old = MemoryFile(
        uri=uri,
        memory_type="cases",
        content="",
        extra_fields={
            "case_name": "library",
            "domain": "library",
            "description": "",
            "rubric": before,
        },
    )
    fs = Store({uri: MemoryFileUtils.write(old)})
    target = updater(fs)
    target._registry = reg
    candidate = ResolvedOperation(
        memory_type="cases",
        uris=[uri],
        page_id=1,
        memory_fields={"rubric": patch("Respect scope.", "Respect authorized scope.")},
        old_memory_file_content=old,
    )
    extractor = loop()
    extractor.context_provider = SimpleNamespace(get_memory_schemas=lambda _: reg.list_all())
    assert await extractor._validate_content_gates(batch(candidate)) == []
    result = await target.apply_operations(batch(candidate), CTX)
    assert not result.errors
    memory = MemoryFileUtils.read(fs.files[uri], uri=uri)
    assert (
        _memory_file_to_case(memory).rubric.criteria[0].description == "Respect authorized scope."
    )
    saved = fs.files[uri]
    candidate.memory_fields = {"rubric": patch('"id": "scope"', '"id": false')}
    candidate.old_memory_file_content = memory
    assert await extractor._validate_content_gates(batch(candidate))
    assert (await target.apply_operations(batch(candidate), CTX)).errors
    assert fs.files[uri] == saved
    assert not fs.lock.locked()


@pytest.mark.asyncio
async def test_checked_patch_rejects_overlapping_original_ranges():
    changes = StrPatch(
        blocks=[
            SearchReplaceBlock(search="old reminder", replace="updated reminder"),
            SearchReplaceBlock(search="old reminder", replace="final reminder"),
        ]
    )
    with pytest.raises(ValueError) as caught:
        await PatchOp(FieldType.STRING).apply_checked(SOURCE, changes)
    error = str(caught.value)
    assert "edits[0] and edits[1] overlap in string field" in error
    assert "Merge them into one edit or target disjoint regions" in error


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["python", "json"])
@pytest.mark.parametrize(
    ("original_fragment", "approximate_text"),
    [
        ("old “reminder”", 'old "reminder"'),
        (
            "退款原路退回，且退款账户必须为原支付方式。",
            "退款原路退回,且退款账户必须为原支付方式。",
        ),
    ],
    ids=["quotes", "unicode-width"],
)
async def test_patch_formats_share_fuzzy_preview_and_locked_write(
    monkeypatch, output_format, original_fragment, approximate_text
):
    source = SOURCE.replace("old reminder", original_fragment)
    search = f'reminder = tell("{approximate_text}")'
    replacement = 'reminder = tell("new reminder")'
    change = patch(search, replacement)
    assert search not in source
    expected = await PatchOp(FieldType.STRING).apply(source, change)
    assert expected == source.replace(original_fragment, "new reminder")
    if output_format == "python":
        response = (
            f"experience.content.edit(search={search!r}, replace={replacement!r})\nsdk.commit()"
        )
    else:
        response = json.dumps(
            {
                "experiences": [
                    {
                        "page_id": 1,
                        "experience_name": "case_a",
                        "case_name": "case_a",
                        "content": change.model_dump(),
                    }
                ],
                "delete_ids": [],
            }
        )
    extraction, model, fs = extraction_case(
        monkeypatch, [response], source=source, output_format=output_format
    )
    result, _ = await extraction.run()
    model.get_completion_async.assert_awaited_once()
    assert isinstance(result.upsert_operations[0].memory_fields["content"], StrPatch)
    assert not fs.writes
    applied = await updater(fs).apply_operations(result, CTX)
    assert not applied.errors
    assert MemoryFileUtils.read(fs.files[URI], uri=URI).content == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("as_dict", [False, True])
async def test_failed_block_after_fuzzy_match_does_not_persist_partial_edits(as_dict):
    change = StrPatch(
        blocks=[
            SearchReplaceBlock(
                search='reminder = tell("old reminder")   ',
                replace='reminder = tell("new reminder")',
            ),
            SearchReplaceBlock(search="UNRELATED_MISSING_ANCHOR_12345", replace="never saved"),
        ]
    )
    # Both the raw engine and the gated writer reject the whole batch, not just
    # the failing second edit. No partial first edit may escape either path.
    with pytest.raises(ValueError, match=r"Could not find edits\[1\]"):
        await PatchOp(FieldType.STRING).apply(SOURCE, change)
    deleted = file(uri=URI.replace("case_a.md", "obsolete.md"))
    fs = Store(
        {
            URI: MemoryFileUtils.write(file()),
            OTHER: "keep",
            deleted.uri: MemoryFileUtils.write(deleted),
        }
    )
    before = dict(fs.files)
    candidate = operation(change.model_dump() if as_dict else change, file())
    errors = await loop()._validate_content_gates(batch(candidate))
    assert len(errors) == 1 and "edits[1]" in errors[0]["error"]
    result = await updater(fs).apply_operations(
        batch(
            operation("changed", memory_type="records", uri=OTHER),
            candidate,
            deletes=[deleted],
        ),
        CTX,
    )
    assert result.errors and "edits[1]" in str(result.errors[0][1])
    assert fs.files == before and fs.writes == [] and fs.deletes == []
    assert not fs.lock.locked()


@pytest.mark.asyncio
async def test_fuzzy_edit_does_not_rebase_over_a_concurrently_changed_field():
    change = patch('reminder = tell("old reminder")   ', 'reminder = tell("new reminder")')
    candidate = operation(change, file())
    assert await loop()._validate_content_gates(batch(candidate)) == []
    latest = SOURCE.replace("old reminder", "independently changed reminder")
    fs = Store({URI: MemoryFileUtils.write(file())})
    fs.on_lock = lambda: fs.files.update({URI: MemoryFileUtils.write(file(latest))})
    result = await updater(fs).apply_operations(batch(candidate), CTX)
    assert result.errors and "changed since it was read" in str(result.errors[0][1])
    assert MemoryFileUtils.read(fs.files[URI], uri=URI).content == latest
    assert not fs.writes and not fs.lock.locked()


@pytest.mark.asyncio
async def test_fuzzy_match_still_requires_a_valid_final_dag():
    change = patch(
        'reminder = tell("old reminder")   ',
        'reminder = tell("new reminder")\nmissing.then(reminder)',
    )
    candidate = operation(change, file())
    errors = await loop()._validate_content_gates(batch(candidate))
    assert len(errors) == 1 and "Unsupported DAG relation" in errors[0]["error"]
    fs = Store({URI: MemoryFileUtils.write(file())})
    before = dict(fs.files)
    result = await updater(fs).apply_operations(batch(candidate), CTX)
    assert result.errors and fs.files == before and not fs.writes
    assert not fs.lock.locked()


@pytest.mark.asyncio
async def test_pi_touched_line_normalization_cannot_persist_an_invalid_dag():
    source = SOURCE.replace("old reminder", "请确认 “yes”，备注“保留”")
    change = patch('请确认 "yes"', "请确认 “no”")
    candidate = operation(change, file(source))
    assert await loop()._validate_content_gates(batch(candidate))
    fs = Store({URI: MemoryFileUtils.write(file(source))})
    before = dict(fs.files)
    result = await updater(fs).apply_operations(batch(candidate), CTX)
    assert result.errors
    assert fs.files == before and not fs.writes
    assert not fs.lock.locked()
