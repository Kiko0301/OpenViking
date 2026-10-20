# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from typing import Any

import pytest
from test_fakes import fake_request_context

from openviking.session.memory.dataclass import MemoryFile, StoredLink
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.skill.session_skill_context_provider import (
    SESSION_SKILL_MEMORY_TYPE,
    load_skill_extract_registry,
)
from openviking.session.train import (
    ContentHashPolicySnapshotter,
    DryRunPolicyUpdater,
    Experience,
    ExperienceSet,
    ExperienceSetLoader,
    MemoryFilePolicyUpdater,
    PatchMergePolicyOptimizer,
    PatchMergePolicyOptimizerContext,
    PatchSemanticGradient,
    PolicyUpdatePlan,
)
from openviking.storage.errors import LockAcquisitionError


class FakePathlockClient:
    def __init__(self, acquire_error: Exception | None = None):
        self.acquire_error = acquire_error
        self.acquire_calls = []
        self.release_calls = []

    async def pathlock_acquire_exact_tree_batch(
        self,
        exact_paths,
        tree_paths,
        timeout_secs=0.0,
        owner_lease_ref=None,
    ):
        call = {
            "exact_paths": list(exact_paths),
            "tree_paths": list(tree_paths),
            "timeout_secs": timeout_secs,
            "owner_lease_ref": owner_lease_ref,
        }
        self.acquire_calls.append(call)
        if self.acquire_error is not None:
            raise self.acquire_error
        lease = {"lease_ref": f"combined-lease-{len(self.acquire_calls)}"}
        call["lease"] = lease
        return lease

    async def pathlock_release(self, lease):
        self.release_calls.append(lease)


class FakeVikingFS:
    def __init__(
        self,
        files: dict[str, str],
        *,
        lock_acquire_error: Exception | None = None,
    ):
        self.files = files
        self.rm_lock_handles = []
        self.write_lock_handles = []
        self._async_agfs = FakePathlockClient(lock_acquire_error)

    def _uri_to_path(self, uri: str, ctx=None) -> str:
        account_id = getattr(getattr(ctx, "user", None), "account_id", None) or "default"
        return f"/local/{account_id}/{uri.removeprefix('viking://')}"

    async def ls(self, uri: str, output: str = "original", ctx=None, **kwargs):
        assert output == "original"
        prefix = uri.rstrip("/") + "/"
        entries = {}
        for path in sorted(self.files):
            if not path.startswith(prefix):
                continue
            rest = path.removeprefix(prefix)
            name, separator, _ = rest.partition("/")
            entries[name] = {"name": name, "uri": prefix + name, "isDir": bool(separator)}
        offset = kwargs.get("offset", 0)
        return list(entries.values())[offset : offset + kwargs.get("node_limit", 1000)]

    async def read_file(self, uri: str, ctx=None):
        return self.files[uri]

    async def write_file(self, uri: str, content: str, ctx=None, lease_ref=None):
        self.write_lock_handles.append((uri, lease_ref))
        self.files[uri] = content

    async def rm(self, uri: str, recursive: bool = False, ctx=None, lease_ref=None):
        del recursive, ctx
        self.rm_lock_handles.append(lease_ref)
        self.files.pop(uri, None)
        return {"estimated_deleted_count": 1}


class FakeVikingDB:
    def __init__(self):
        self.embedding_messages = []

    async def enqueue_embedding_msg(self, embedding_msg):
        self.embedding_messages.append(embedding_msg)
        return True


def _experience_set() -> ExperienceSet:
    return ExperienceSet(
        root_uri="viking://user/u/memories/experiences",
        policies=[
            Experience(
                name="booking_duplicate_handling",
                uri="viking://user/u/memories/experiences/booking_duplicate_handling.md",
                version=1,
                status="production",
                content="content",
            )
        ],
    )


def _memory_file(
    *,
    name: str,
    uri: str | None,
    content: str,
    version: int | None = 1,
    status: str = "production",
) -> MemoryFile:
    fields: dict[str, Any] = {
        "memory_type": "experiences",
        "experience_name": name,
        "status": status,
    }
    if version is not None:
        fields["version"] = version
    return MemoryFile(
        uri=uri,
        content=content,
        memory_type="experiences",
        extra_fields=fields,
    )


def _patch_gradient(
    *,
    name: str = "booking_duplicate_handling",
    uri: str | None = "viking://user/u/memories/experiences/booking_duplicate_handling.md",
    before: str | None = "content",
    after: str = "new content",
    base_version: int | None = 1,
    rationale: str = "r",
    links: list[StoredLink] | None = None,
    confidence: float = 0.8,
    metadata: dict[str, Any] | None = None,
) -> PatchSemanticGradient:
    return PatchSemanticGradient(
        before_file=(
            _memory_file(name=name, uri=uri, content=before, version=base_version)
            if before is not None
            else None
        ),
        after_file=_memory_file(name=name, uri=uri, content=after, version=base_version),
        base_version=base_version,
        rationale=rationale,
        links=(
            links
            if links is not None
            else [
                StoredLink(
                    from_uri=uri or "",
                    to_uri="viking://user/u/memories/trajectories/traj1.md",
                    link_type="derived_from",
                    weight=1.0,
                )
            ]
        ),
        confidence=confidence,
        metadata=metadata or {},
    )


def _plan_from_gradient(gradient: PatchSemanticGradient) -> PolicyUpdatePlan:
    return PolicyUpdatePlan(
        items=[
            _plan_item_from_gradient(gradient),
        ]
    )


def _plan_item_from_gradient(gradient: PatchSemanticGradient):
    from openviking.session.train import PolicyPlanItem

    return PolicyPlanItem(
        kind="upsert",
        memory_type="experiences",
        target_name=gradient.target_name,
        target_uri=gradient.target_uri,
        before_content=(
            gradient.before_file.plain_content() if gradient.before_file is not None else None
        ),
        after_content=gradient.after_file.plain_content(),
        base_version=gradient.base_version,
        confidence=gradient.confidence,
        links=list(gradient.links),
        metadata={"rationale": gradient.rationale},
    )


def _delete_plan(*, uri: str, before_content: str = "content") -> PolicyUpdatePlan:
    from openviking.session.train import PolicyPlanItem

    return PolicyUpdatePlan(
        items=[
            PolicyPlanItem(
                kind="delete",
                memory_type="experiences",
                target_name="booking_duplicate_handling",
                target_uri=uri,
                before_content=before_content,
                after_content=None,
                base_version=1,
                confidence=0.8,
                links=[
                    StoredLink(
                        from_uri=uri,
                        to_uri="viking://user/u/memories/trajectories/traj1.md",
                        link_type="derived_from",
                        weight=1.0,
                    )
                ],
                metadata={"rationale": "delete duplicate experience"},
            )
        ]
    )


@pytest.mark.asyncio
async def test_experience_set_loader_reads_memory_files():
    root = "viking://user/u/memories/experiences"
    fs = FakeVikingFS(
        {
            f"{root}/booking_duplicate_handling.md": '## Situation\n- test\n\n<!-- MEMORY_FIELDS\n{"memory_type": "experiences", "experience_name": "booking_duplicate_handling", "version": 3, "status": "staging"}\n-->',
            f"{root}/.overview.md": "hidden",
        }
    )

    ctx = fake_request_context()
    loaded = await ExperienceSetLoader(viking_fs=fs).load(root, ctx=ctx)

    assert loaded.root_uri == root
    assert loaded.viking_fs is fs
    assert loaded.request_context is ctx
    assert len(loaded.policies) == 1
    policy = loaded.policies[0]
    assert policy.name == "booking_duplicate_handling"
    assert policy.version == 3
    assert policy.status == "staging"
    assert policy.content == "## Situation\n- test"
    assert policy.metadata["memory_type"] == "experiences"


@pytest.mark.asyncio
async def test_experience_set_loader_requires_request_context():
    root = "viking://user/u/memories/experiences"
    fs = FakeVikingFS({})

    with pytest.raises(ValueError, match="requires request_context ctx"):
        await ExperienceSetLoader(viking_fs=fs).load(root)


@pytest.mark.asyncio
async def test_content_hash_snapshotter_is_deterministic():
    snapshotter = ContentHashPolicySnapshotter()
    policy_set = _experience_set()

    first = await snapshotter.snapshot(policy_set)
    second = await snapshotter.snapshot(policy_set)

    assert first == second
    assert first.startswith("policy-snapshot:")


@pytest.mark.asyncio
async def test_dry_run_policy_updater_does_not_mutate_policy_set():
    policy_set = _experience_set()
    plan = PolicyUpdatePlan(metadata={"hello": "world"})

    result = await DryRunPolicyUpdater().apply(plan, policy_set)

    assert result.updated_policy_set is policy_set
    assert result.written_uris == []
    assert result.deleted_uris == []
    assert result.metadata["dry_run"] is True
    assert result.metadata["simulated"] is True
    assert result.metadata["plan"] == {"hello": "world"}


@pytest.mark.asyncio
async def test_dry_run_policy_updater_simulates_patch_plan_items():
    policy_set = _experience_set()
    gradient = _patch_gradient(
        uri=policy_set.policies[0].uri,
        before="content",
        after='dag = workflow("new content")\nstep = tell("new content")',
    )
    plan = _plan_from_gradient(gradient)

    result = await DryRunPolicyUpdater().apply(plan, policy_set)

    assert result.updated_policy_set is not policy_set
    assert "new content" in result.updated_policy_set.policies[0].content
    assert result.updated_policy_set.policies[0].version == 2
    assert result.written_uris == []
    assert result.metadata["dry_run"] is True
    assert result.metadata["simulated"] is True


@pytest.mark.asyncio
async def test_dry_run_policy_updater_simulates_delete_plan_items():
    policy_set = _experience_set()
    plan = _delete_plan(uri=policy_set.policies[0].uri)

    result = await DryRunPolicyUpdater().apply(plan, policy_set)

    assert result.updated_policy_set is not policy_set
    assert result.updated_policy_set.policies == []
    assert result.written_uris == []
    assert result.deleted_uris == []
    assert result.metadata["dry_run"] is True
    assert result.metadata["simulated"] is True


@pytest.mark.asyncio
async def test_memory_file_policy_updater_writes_experience_files():
    policy_set = _experience_set()
    fs = FakeVikingFS({})
    gradient = _patch_gradient(
        uri=policy_set.policies[0].uri,
        before="content",
        after='dag = workflow("new content")\nstep = tell("new content")',
        links=[],
    )
    plan = _plan_from_gradient(gradient)

    result = await MemoryFilePolicyUpdater(viking_fs=fs).apply(
        plan,
        policy_set,
        fake_request_context(),
    )

    assert result.errors == []
    assert result.written_uris == [policy_set.policies[0].uri]
    written = fs.files[policy_set.policies[0].uri]
    assert written.startswith('dag = workflow("new content")')
    assert '"memory_type": "experiences"' in written
    assert '"experience_name": "booking_duplicate_handling"' in written
    assert '"version": 2' in written


@pytest.mark.asyncio
async def test_memory_file_policy_updater_does_not_expand_lock_without_trajectory_links():
    policy_set = _experience_set()
    fs = FakeVikingFS({})
    lock_handle = object()
    gradient = _patch_gradient(
        uri=policy_set.policies[0].uri,
        before="content",
        after='dag = workflow("new content")\nstep = tell("new content")',
        links=[],
    )
    plan = _plan_from_gradient(gradient)

    result = await MemoryFilePolicyUpdater(viking_fs=fs).apply(
        plan,
        policy_set,
        fake_request_context(),
        transaction_handle=lock_handle,
    )

    assert result.errors == []
    assert result.written_uris == [policy_set.policies[0].uri]
    assert (policy_set.policies[0].uri, lock_handle) in fs.write_lock_handles
    assert fs._async_agfs.acquire_calls == []
    assert fs._async_agfs.release_calls == []


@pytest.mark.asyncio
async def test_memory_file_policy_updater_vectorizes_written_experience_files():
    policy_set = _experience_set()
    fs = FakeVikingFS({})
    vikingdb = FakeVikingDB()
    gradient = _patch_gradient(
        uri=policy_set.policies[0].uri,
        before="content",
        after='dag = workflow("new content")\nstep = tell("new content")',
        links=[],
    )
    plan = _plan_from_gradient(gradient)

    from openviking.server.identity import RequestContext, Role
    from openviking_cli.session.user_id import UserIdentifier

    result = await MemoryFilePolicyUpdater(viking_fs=fs, vikingdb=vikingdb).apply(
        plan,
        policy_set,
        RequestContext(user=UserIdentifier("default", "u"), role=Role.USER),
    )

    assert result.errors == []
    assert result.written_uris == [policy_set.policies[0].uri]
    assert len(vikingdb.embedding_messages) == 1
    embedding_msg = vikingdb.embedding_messages[0]
    assert embedding_msg.context_data["uri"] == policy_set.policies[0].uri
    assert embedding_msg.context_data["context_type"] == "memory"
    assert "new content" in embedding_msg.message


@pytest.mark.asyncio
async def test_memory_file_policy_updater_deletes_experience_files():
    policy_set = _experience_set()
    uri = policy_set.policies[0].uri
    fs = FakeVikingFS({uri: "content"})
    plan = _delete_plan(uri=uri)
    lock_handle = object()

    result = await MemoryFilePolicyUpdater(viking_fs=fs).apply(
        plan,
        policy_set,
        transaction_handle=lock_handle,
    )

    assert result.errors == []
    assert result.written_uris == []
    assert result.deleted_uris == [uri]
    assert result.updated_policy_set.policies == []
    assert uri not in fs.files
    assert fs.rm_lock_handles == [lock_handle]


@pytest.mark.asyncio
async def test_memory_file_policy_updater_detects_base_content_mismatch():
    policy_set = _experience_set()
    fs = FakeVikingFS({})
    gradient = _patch_gradient(
        uri=policy_set.policies[0].uri,
        before="stale content",
        after='dag = workflow("new content")\nstep = tell("new content")',
    )
    plan = _plan_from_gradient(gradient)

    result = await MemoryFilePolicyUpdater(viking_fs=fs).apply(plan, policy_set)

    assert result.written_uris == []
    assert result.errors == [
        "base content mismatch for booking_duplicate_handling: expected gradient before_content"
    ]
    assert policy_set.policies[0].uri not in fs.files


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
async def test_patch_merge_policy_optimizer_uses_session_skill_registry(monkeypatch, existing):
    from openviking.core.skill_loader import SkillLoader
    from openviking.session.memory.dataclass import (
        ResolvedOperation,
        ResolvedOperations,
    )
    from openviking.session.train.components.skill_policy_updater import SkillPolicyUpdater

    skill_uri = "viking://user/u/skills/code-review/SKILL.md"
    old_skill = {
        "name": "code-review",
        "description": "Old description",
        "content": "Old content.",
    }

    class SkillFS(FakeVikingFS):
        async def search(self, *args, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(to_dict=lambda: {"memories": [], "resources": [], "skills": []})

        async def read_file(self, uri, ctx=None):
            if uri not in self.files:
                raise FileNotFoundError(uri)
            return self.files[uri]

    fs = SkillFS({skill_uri: SkillLoader.to_skill_md(old_skill)} if existing else {})

    class FakeProcessor:
        async def process_skill(self, *, data, **kwargs):
            await fs.write_file(skill_uri, SkillLoader.to_skill_md(data))
            return {"root_uri": skill_uri.removesuffix("/SKILL.md")}

        async def sanitize_skill_privacy(self, skill, ctx):
            return skill

    policy_set = ExperienceSet(
        root_uri="viking://user/u/skills",
        policies=[
            Experience(
                name=old_skill["name"],
                uri=skill_uri,
                version=1,
                status="production",
                content=old_skill["content"],
                metadata={"description": old_skill["description"]},
            )
        ]
        if existing
        else [],
    )
    gradient = PatchSemanticGradient(
        before_file=None,
        after_file=MemoryFile(
            uri=skill_uri,
            content="Use this skill to review code changes.",
            memory_type=SESSION_SKILL_MEMORY_TYPE,
            extra_fields={
                "memory_type": SESSION_SKILL_MEMORY_TYPE,
                "skill_name": "code-review",
            },
        ),
        base_version=None,
        rationale="test",
        links=[],
        confidence=0.9,
        metadata={},
    )
    captured = {}

    class FakeExtractLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self):
            return (
                ResolvedOperations(
                    upsert_operations=[
                        ResolvedOperation(
                            old_memory_file_content=None,
                            memory_fields={
                                "skill_name": "code-review",
                                "description": "Review code changes",
                                "content": "Merged skill content.",
                            },
                            memory_type=SESSION_SKILL_MEMORY_TYPE,
                            uris=[skill_uri],
                        )
                    ],
                    delete_file_contents=[],
                    errors=[],
                ),
                [],
            )

    monkeypatch.setattr(
        "openviking.session.train.components.policy_optimizer.ExtractLoop",
        FakeExtractLoop,
    )

    plan = await PatchMergePolicyOptimizer(
        viking_fs=fs,
        vlm=object(),
        memory_type=SESSION_SKILL_MEMORY_TYPE,
        memory_registry=load_skill_extract_registry(),
    ).plan(
        [gradient],
        policy_set,
        PatchMergePolicyOptimizerContext(request_context=fake_request_context()),
    )

    assert captured["isolation_handler"].allowed_memory_types == {SESSION_SKILL_MEMORY_TYPE}
    assert len(plan.items) == 1
    assert plan.items[0].memory_type == SESSION_SKILL_MEMORY_TYPE
    assert plan.items[0].target_name == "code-review"
    assert plan.items[0].target_uri == skill_uri
    assert plan.items[0].after_content == "Merged skill content."

    result = await SkillPolicyUpdater(skill_processor=FakeProcessor(), viking_fs=fs).apply(
        plan, policy_set, fake_request_context()
    )
    assert result.errors == []
    assert result.written_uris == [skill_uri]
    saved = SkillLoader.parse(await fs.read_file(skill_uri))
    assert saved["description"] == "Review code changes"
    assert saved["content"] == "Merged skill content."
    assert result.updated_policy_set.policies[0].metadata["description"] == saved["description"]
    if existing:
        assert policy_set.policies[0].metadata["description"] == "Old description"


@pytest.mark.asyncio
@pytest.mark.parametrize("locked", [False, True], ids=["standalone", "outer-tree-lock"])
async def test_skill_policy_creation_preserves_lock_ownership(tmp_path, monkeypatch, locked):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from openviking.core.skill_loader import SkillLoader
    from openviking.server.identity import RequestContext, Role
    from openviking.session.train.components.skill_policy_updater import SkillPolicyUpdater
    from openviking.session.train.domain import PolicyPlanItem
    from openviking.storage.viking_fs import VikingFS
    from openviking.utils.agfs_utils import RagfsBindingConfig, create_agfs_client
    from openviking.utils.skill_processor import SkillProcessor
    from openviking_cli.session.user_id import UserIdentifier
    from openviking_cli.utils.config.agfs_config import AGFSConfig

    agfs = create_agfs_client(RagfsBindingConfig(agfs=AGFSConfig(path=str(tmp_path))))
    fs = VikingFS(agfs=agfs)
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role(Role.ROOT))
    root = "viking://user/default/skills"
    uri = f"{root}/code-review/SKILL.md"
    processor = SkillProcessor(vikingdb=AsyncMock())
    queue = SimpleNamespace(enqueue=AsyncMock())
    manager = SimpleNamespace(SEMANTIC="Semantic", get_queue=lambda *args, **kwargs: queue)
    monkeypatch.setattr("openviking.storage.queuefs.get_queue_manager", lambda: manager)
    policy_set = ExperienceSet(root_uri=root, policies=[], viking_fs=fs, request_context=ctx)
    updater = SkillPolicyUpdater(skill_processor=processor, viking_fs=fs)
    plan = PolicyUpdatePlan(
        items=[
            PolicyPlanItem(
                kind="upsert",
                memory_type=SESSION_SKILL_MEMORY_TYPE,
                target_name="code-review",
                target_uri=uri,
                before_content=None,
                after_content="Review steps.",
                metadata={"merge_memory_fields": {"description": "Review code changes"}},
            )
        ]
    )
    try:
        async with policy_set.lock() if locked else nullcontext() as lease:
            result = await updater.apply(plan, policy_set, ctx, transaction_handle=lease)
            assert result.errors == []
            assert result.written_uris == [uri]
            assert SkillLoader.parse(await fs.read_file(uri, ctx=ctx))["content"] == "Review steps."
            for filename in (".abstract.md", ".overview.md"):
                assert await fs.read_file(f"{root}/code-review/{filename}", ctx=ctx)
            if locked:
                # The callee must not release the caller's tree lock.
                with pytest.raises(LockAcquisitionError):
                    await fs._async_agfs.pathlock_acquire_exact(fs._uri_to_path(uri, ctx=ctx))
        # Queued indexing keeps its own package lease after the caller exits.
        with pytest.raises(LockAcquisitionError):
            await fs._async_agfs.pathlock_acquire_exact(fs._uri_to_path(uri, ctx=ctx))
        msg = queue.enqueue.await_args.args[0]
        worker_lease = await fs._async_agfs.pathlock_adopt(msg.lock_handoff)
        await fs._async_agfs.pathlock_release(worker_lease)
        lease = await fs._async_agfs.pathlock_acquire_tree(fs._uri_to_path(root, ctx=ctx))
        await fs._async_agfs.pathlock_release(lease)
    finally:
        from openviking.telemetry import unregister_telemetry
        from openviking.telemetry.request_wait_tracker import get_request_wait_tracker

        if queue.enqueue.await_args:
            msg = queue.enqueue.await_args.args[0]
            tracker = get_request_wait_tracker()
            tracker.mark_semantic_done(msg.telemetry_id, msg.id)
            tracker.cleanup(msg.telemetry_id)
            unregister_telemetry(msg.telemetry_id)
        agfs.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["edit", "drop", "overwrite", "noop"])
async def test_fixed_case_merge_materializes_sdk_edits_against_latest_source(method):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    policy_set = _experience_set()
    current = policy_set.policies[0]
    stale = (
        'dag = workflow("review")\n'
        "reminder = tell(\n"
        '    "Verify eligibility"\n'
        '    "; skip eligibility"\n'
        ")"
    )
    obsolete_line = '    "; skip eligibility"'
    obsolete_block = obsolete_line + "\n"
    current.content = stale + '\nkeep = tell("Preserve the current request scope")'
    current.version = 3
    expected = current.content.replace(obsolete_block, "")
    fs = FakeVikingFS(
        {
            current.uri: MemoryFileUtils.write(
                _memory_file(name=current.name, uri=current.uri, content=current.content, version=3)
            )
        }
    )
    gradient = _patch_gradient(
        before=stale,
        after=stale.replace(obsolete_block, ""),
        base_version=1,
        metadata={
            "proposal_source_sessions": [{"source_session_uri": "viking://session/s/archives/1"}],
            "experience_proposal_gate": {"passed": True},
        },
    )
    programs = {
        "edit": (
            f"prefetched_memory_1.content.edit(search={obsolete_block!r}, replace='')\nsdk.commit()"
        ),
        "drop": f"prefetched_memory_1.content.drop(text={obsolete_block!r})\nsdk.commit()",
        "overwrite": f"prefetched_memory_1.content.overwrite({expected!r})\nsdk.commit()",
        "noop": "sdk.commit()",
    }
    model = SimpleNamespace(
        model="test", get_completion_async=AsyncMock(return_value=programs[method])
    )

    plan = await PatchMergePolicyOptimizer(viking_fs=fs, vlm=model).plan(
        [gradient],
        policy_set,
        PatchMergePolicyOptimizerContext(request_context=fake_request_context()),
    )

    model.get_completion_async.assert_awaited_once()
    assert MemoryFileUtils.read(fs.files[current.uri]).content == current.content
    assert plan.read_policy_uris == [current.uri]
    if method == "noop":
        assert plan.items == []
        return
    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.after_content == expected
    assert item.before_content == current.content
    assert item.base_version == 3
    assert item.target_uri == current.uri
    assert item.links == gradient.links


@pytest.mark.asyncio
async def test_fixed_case_merge_groups_sessions_and_rebases_on_latest_policy(monkeypatch):
    from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations

    policy_set = _experience_set()
    current = policy_set.policies[0]
    current.content = 'dag = workflow("current")\ncurrent = tell("keep concurrent update")'
    current.metadata["source_sessions"] = [
        {
            "source_session_uri": "viking://session/old/archives/1",
            "passed": True,
            "feedback": ["large legacy evaluator payload"],
        }
    ]
    source_uris = [f"viking://session/s{i}/archives/1" for i in range(3)]

    def proposal(name, source):
        uri = f"{policy_set.root_uri}/{name}.md"
        return PatchSemanticGradient(
            before_file=MemoryFile(uri=uri, content="stale base"),
            after_file=MemoryFile(
                uri=uri,
                content=f"proposed {source}",
                memory_type="experiences",
                extra_fields={"experience_name": name},
            ),
            base_version=1,
            rationale="session",
            links=[StoredLink(from_uri=uri, to_uri=source, link_type="derived_from")],
            confidence=0.8,
            metadata={
                "proposal_source_sessions": [{"source_session_uri": source, "passed": True}],
                "experience_proposal_gate": {"passed": True, "enabled": True},
            },
        )

    proposals = [
        proposal(current.name, source_uris[0]),
        proposal(current.name, source_uris[1]),
        proposal("other_case", source_uris[2]),
    ]
    groups = []

    async def merge(self, *, gradients, policy_set, context):
        groups.append(gradients)
        name = gradients[0].target_name
        return ResolvedOperations(
            upsert_operations=[
                ResolvedOperation(
                    memory_type="experiences",
                    uris=[gradients[0].target_uri],
                    memory_fields={
                        "experience_name": name,
                        "content": 'dag = workflow("merged")\nkeep = tell("keep concurrent update")',
                    },
                )
            ],
            delete_file_contents=[],
            errors=[],
        )

    monkeypatch.setattr(PatchMergePolicyOptimizer, "_run_merge_extract_loop", merge)
    result = await PatchMergePolicyOptimizer().plan(
        proposals,
        policy_set,
        PatchMergePolicyOptimizerContext(request_context=fake_request_context()),
    )
    assert [len(group) for group in groups] == [2, 1]
    assert result.read_policy_uris == sorted({p.target_uri for p in proposals})
    assert len(result.items) == 2
    assert result.items[0].before_content == current.content
    assert result.items[0].base_version == current.version
    assert {link.to_uri for link in result.items[0].links} == set(source_uris[:2])
    assert {link.to_uri for link in result.items[1].links} == {source_uris[2]}
    assert {
        source["source_session_uri"] for source in result.items[0].metadata["source_sessions"]
    } == set(source_uris[:2])
    assert all("feedback" not in source for source in result.items[0].metadata["source_sessions"])


@pytest.mark.asyncio
@pytest.mark.parametrize("violation", ["rename", "delete", "cross_case", "extra"])
async def test_fixed_case_merge_rejects_target_changes(monkeypatch, violation):
    from unittest.mock import AsyncMock

    from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations

    policy_set = _experience_set()
    current = policy_set.policies[0]
    file = MemoryFile(
        uri=current.uri,
        content="new",
        memory_type="experiences",
        extra_fields={"experience_name": current.name},
    )
    gradient = PatchSemanticGradient(
        None,
        file,
        None,
        "session",
        [
            StoredLink(
                from_uri=current.uri,
                to_uri="viking://session/s/archives/1",
                link_type="derived_from",
            )
        ],
        0.8,
    )
    op = ResolvedOperation(
        memory_type="experiences",
        uris=[current.uri],
        memory_fields={"experience_name": current.name, "content": "new"},
    )
    operations = ResolvedOperations(upsert_operations=[op], delete_file_contents=[], errors=[])
    if violation == "rename":
        op.memory_fields["experience_name"] = "other"
    if violation == "delete":
        operations.delete_file_contents.append(file)
    if violation == "cross_case":
        op.uris = ["viking://user/u/memories/experiences/other.md"]
    if violation == "extra":
        operations.upsert_operations.append(op.model_copy())
    monkeypatch.setattr(
        PatchMergePolicyOptimizer, "_run_merge_extract_loop", AsyncMock(return_value=operations)
    )
    with pytest.raises(ValueError):
        await PatchMergePolicyOptimizer().plan(
            [gradient],
            policy_set,
            PatchMergePolicyOptimizerContext(request_context=fake_request_context()),
        )


@pytest.mark.asyncio
async def test_policy_updater_preserves_source_archive_and_keeps_session_metadata_transient():
    from openviking.session.train import PolicyPlanItem

    source_uri = "viking://session/s/archives/1"
    archive_uri = source_uri + "/messages.jsonl"
    source = '{"role":"user","content":"original archive"}'
    fs = FakeVikingFS({archive_uri: source})
    current = _experience_set().policies[0]
    fs.files[current.uri] = MemoryFileUtils.write(
        MemoryFile(
            uri=current.uri,
            content=current.content,
            memory_type="experiences",
            extra_fields={
                "experience_name": current.name,
                "source_sessions": [{"source_session_uri": "old-session", "passed": True}],
            },
        )
    )
    item = PolicyPlanItem(
        kind="upsert",
        memory_type="experiences",
        target_name=current.name,
        target_uri=current.uri,
        before_content=current.content,
        after_content='dag = workflow("updated")\nstep = tell("updated content")',
        links=[StoredLink(from_uri=current.uri, to_uri=source_uri, link_type="derived_from")],
        metadata={"source_sessions": [{"source_session_uri": source_uri, "passed": True}]},
    )
    result = await MemoryFilePolicyUpdater(viking_fs=fs).apply(
        PolicyUpdatePlan(items=[item]),
        _experience_set(),
        fake_request_context(),
        transaction_handle="root-lease",
    )
    assert result.errors == []
    stored = MemoryFileUtils.read(fs.files[current.uri], uri=current.uri)
    assert "source_sessions" not in stored.extra_fields
    assert item.metadata["source_sessions"][0]["passed"] is True
    assert stored.links == []
    assert item.links[0].to_uri == source_uri
    assert fs.files[archive_uri] == source
    assert source_uri not in fs.files
    assert fs._async_agfs.acquire_calls == []
