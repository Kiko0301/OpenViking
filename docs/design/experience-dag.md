# DAG experience memory

Experience templates stay at `viking://user/{user}/memories/experiences/{name}.md`.
The `content` field contains the complete DAG construction source. Python is never executed
with `exec` or `eval`; a restricted interpreter validates and builds the runtime DAG.
Existing text experiences are not converted or accepted as DAG templates.

## Case means a business domain

The existing `Case`, `cases` memory type, `case_name` and learning APIs keep their names.
A memory Case describes a business domain, not one benchmark question or narrow workflow.
Tasks within the same domain share its fixed Experience; workflow/rule differences live in
conditional Experience branches. Different business domains remain distinct.

Batch CaseSpec remains the concrete task transport for execution and scoring. Commits use the
same user-memory extraction, retrieval and merge pipeline as ordinary Sessions, not a task-ID
Case writer. Control envelopes stay outside the numbered business conversation. When Cases are
allowed, task requirements and external evaluation reports are supplied separately to the Case
extractor as rubric evidence, not as personal memories, events or task-specific domain rules.
This is a model-facing projection, not the full CaseSpec: retain domain/user request, rubric
text and reported evaluation evidence; omit task name/signature, split/task/trial metadata and
rubric labels derived from task identity. Criterion results are associated with criterion text
when available. The original envelopes, Session identifiers and experiment records remain
unchanged for tracing/scoring. This does not scrub literal strings from business dialogue or
migrate Trajectories when a Case is merged.

A Case has no `evidence` field: domain scope/context/principles are its definition; execution
facts and external scores belong to Trajectories. In a final Python extraction program, binding
an existing Case to a local variable (e.g. `matched_case = prefetched_search[0]`) selects it without
a content write; `sdk.commit()` is optional. `sdk.commit(cases=[...])` overrides these implicit
selections; `cases=[]` clears them. JSON still uses explicit `selected_case_ids`.
Only loaded, in-scope, non-deleted Case references are accepted. New or changed Cases use the
normal write path. Read-only selections bypass the write batch. Merely prefetching or reading
candidates does not select them, nor does binding an event/entity or a plain metadata dictionary.
Case merges still require explicit `case_sources`; aliases there do not select inspected candidates.
Selections remain internal references, separate from upserts and archive write diffs. Recording a
Trajectory still maintains its real provenance links/backlinks through the existing storage code.
Old evidence fields are ignored when loading a Case for learning; existing files are not migrated.

New memory proposals use the existing merge candidate lookup even in single-request batches.
Without existing candidates, a singleton can be written without a second model call; otherwise
freshly read candidates and the proposal are compared by the shared patch merger. Each type/peer
batcher keeps at most ten recent successful target URIs as additional read hints, bridging
asynchronous vector indexing without caching file contents. These hints are process-local, not
a global identity registry. Prefetched candidates alone do not select a Case; Case merges must
still associate their final Cases with the submitting executions. Candidate lookup/read/model failures
must not fall back to blindly applying new creates. Existing duplicate files are not migrated.
Automatic flushes recheck current count/time readiness after acquiring the flush lock; queued
flush tasks cannot prematurely consume a new underfilled buffer. Close still drains remaining work.

```text
Task A -> its Session, task_spec and evaluation -> Trajectory A --+
                                                              +-> shared Case -> one Experience
Task B -> its Session, task_spec and evaluation -> Trajectory B --+
```

Each Trajectory stores the supplied concrete `task_spec` (input and rubric included) as server-owned
metadata, alongside its own outcome/evaluation and source archive. It is null for Sessions without
a task envelope; their concrete facts remain in the Session/Trajectory. Persisted Case
`domain` is a one-sentence business-domain/service scope; `description` is a short
introduction with stable background, not a copy of policy/tool manuals. Both are PATCH
string fields supporting partial `edit()`/`drop()` edits or complete `overwrite` replacements;
`case_name` remains immutable. The Case instructions allow evidence-backed refinements
within the same business domain, never repurposing a Case into another domain. This
semantic boundary is expressed in the instructions, not a static domain classifier.
Display, embeddings and the Experience model's Case header use these names.
Benchmark CaseSpec `task_signature` and `input` remain unchanged execution/control data.
The shared internal Case DTO maps domain scope to `task_signature` and the description
to `input["summary"]`; those internal names are not exposed in the domain-memory header.

This is a schema rename without old-name aliases. Existing Case records and persisted
account templates must be explicitly migrated (`task_signature` → `domain`,
`input` → `description`) before reuse; the interim name `domain_scope` also becomes
`domain`. Remove the old top-level fields. The loader rejects
old-named Case records rather than silently losing their scope/description. No existing
online data is migrated automatically by this code change.

Case `rubric` is a mutable JSON list of `{id, description}` criteria
(stored in the schema's string field with `patch` merge semantics). The Case extractor maintains
stable IDs from supplied rules, requirements and feedback without storing task-specific answers.
The normal field gate validates the complete list after patch application; unchanged Cases can
still be selected without writes. Loading maps these definitions into `Rubric.criteria`.

The existing Trajectory extraction call receives the Case rubric, conversation and, if available,
the full external evaluator report. Its LLM generates a YAML list of `id`,
`passed: true/false/null`, block-scalar `evidence`, and optional `description`, without requiring
precomputed rubric results; null covers insufficient evidence or inapplicability.
Partial or empty assessments do not block factual Trajectory persistence. The Case rubric is
guidance, not an ID whitelist. Structurally valid assessments are retained in model order, including unknown/repeated IDs and model-provided descriptions. A matching
Case definition only fills a missing description; an unmatched ID neither discards the
assessment nor causes a retry or extraction failure. No judgment or ID mapping is invented.
Malformed judgments are still omitted; unparseable YAML or a non-list value becomes an empty
assessment list. These structural issues are logged without blocking the factual Trajectory.
No evaluator field-name whitelist creates the results. Retained entries are frozen in
`evaluation_criteria`; Case rubric definitions remain strict and other trajectory validations
are unchanged. Later Case edits or removals cannot reinterpret historical results. Original external outcomes/scores remain separate
and unchanged for benchmark reporting; ordinary Sessions without them retain an unknown overall
outcome. No additional model call or historical rewrite is introduced. Synthesis keeps
these executions separate and judges each against its own task and rules. Feedback collection and
fixed Case-to-Experience ownership are unchanged. Cross-Case feedback routing is not added here.
Existing task-bound Case files are not automatically merged or migrated; running services and
experiments must explicitly adopt the new code/templates before their behavior changes.

## Extraction and updates

Agent Evolution first persists one factual Trajectory for each `(Case, Session archive)`.
The extraction schema exposes four string fields: `context_rules`, `steps`, `evaluation_criteria`
and `action_scores`. `context_rules` holds relevant execution-time rule clauses; `steps` holds
ordered Markdown execution text, not an array. The model emits neither the Trajectory Record
header nor section titles. `content_template` renders the metadata, `# Session Context`, and
`# Trajectory` sections. `MemoryFile.content` / `Trajectory.content` remain the rendered text.
Trajectory identity, Case, outcome, `task_spec`, format, evaluation, execution paths and learning
state remain backend-owned metadata, not model output fields. Identity is supplied before the
normal URI/isolation resolution, not guessed by the model. Numbering, nonempty steps and a
one-to-one source-step count are not runtime requirements; there is no 256-step cap. Neither
extracted string is reparsed, renumbered or assigned role labels by position. The existing
extraction call selects rules based on
the user goal and the full supplied conversation, including requested actions that were skipped
or refused. It preserves conditions and exceptions, not unrelated persona/style instructions;
an empty context section is valid when no relevant rule is supplied. The model is instructed
not to copy training control metadata into either extracted field. Relevant rule selection is
prompt-guided; paragraphs are not required by a runtime
validator to match contiguous substrings of the source text.
Rule selection uses the existing Trajectory extraction call. The model is asked to preserve the
order, source labels and important facts of user messages, assistant messages, Experience Reminders
and tool records. Source input types remain explicit; the server owns Case name, archive
link, outcome, compact evaluation and Experience execution paths. Client-recorded reminder delivery is never
interpreted as business success. Relevant exact amounts and IDs stay in the steps; only the final
Experience generalizes facts into conditional reminders.

```text
Session archive + all associated Cases + OutcomeEvaluation
  -> Phase 1: one joint AgentTrajectoryContextProvider + ExtractLoop
  -> model emits one sdk.create_trajectories(case_name=..., context_rules=..., steps=..., ...) per Case
  -> sdk.commit() ends the single reply
  -> server validates exactly one operation per supplied Case before any write
  -> MemoryUpdater applies content, metadata and provenance links
  -> immutable Trajectory under trajectories/{case_name}/{YYYY}/{MM}/{DD}/{S|F|U}_{short_id}.md
  -> mark Trajectory experience_update.state=pending
  -> Session commit completes and returns trajectory_uris
  -> Phase 2: POST /api/v1/agent-evolution/experiences/learn
  -> claim pending Trajectories under per-record locks and group them by canonical Case
  -> for each non-success Trajectory, ANN-retrieve at most one similar processed success in the same Case
  -> ExperienceGradientEstimator: compare evidence and emit one complete DAG per Case/batch
  -> deterministic plan construction + existing locked policy apply
  -> mark each claimed Trajectory applied, no_op, or failed
```

Each extraction creates a new observation per associated Case, including repeated archives.
File-level creation guards prevent collisions from overwriting existing records. Trajectories
are not online Experience autorecall candidates. Each Trajectory retains its immutable source
Session link; compact source-session metadata is kept for attribution, not as a stored success
index. Commit diffs expose newly persisted Trajectories, including when later DAG synthesis
returns no-op. Old procedure-style Trajectories are not migrated.

Step preprocessing uses up to 16 KiB per item within a 64,000-character conversation budget with
explicit omission notices. System messages are supplied separately for rule selection, not counted
as steps. Step summaries are requested to stay within 800 characters each; this is prompt guidance,
not a runtime content check. ExtractLoop enforces exactly one Trajectory per supplied Case and
the fixed memory type and server-owned storage targets. Persistence, pending scans, historical lookup and
Experience inputs no longer decode a mandatory heading/step format. Source evidence uses bounded
text chunks without interpreting them as steps or changing the stored content. Runtime execution and external
evaluation remain server-owned metadata rather than model-generated claims. Synthesis sends the
complete combined `content` plus compact evaluation and execution metadata, so each run keeps its
own rules rather than inheriting another run's policy. Each selected Trajectory occupies its own
`user` message: compact YAML metadata followed by the original Markdown content, not a JSON-escaped
string. A separate `user` message holds Case/batch metadata, and another holds the existing
Experience as plain source text (or the read error). After the system instructions, model messages
are ordered as Case/batch metadata → existing Experience → Trajectories (success, failure, other)
→ final update request. The generic SDK binding index is still appended at the request boundary.
Existing Experience content stays in a `user` message, not the system rules.
All records still share one model request; this is not one extraction call per Trajectory.
The existing read/SDK-binding path runs before this presentation step, so
`experience.content.edit(...)`, `drop(...)`, and `overwrite(...)` refer to the actually loaded memory. No assistant/tool
prefetch pair is sent to the model on this Experience path; ordinary memory extraction keeps its
native tool messages. Instruction normalization does not rewrite quoted Trajectory evidence.
The 240,000-character comparison budget counts the rendered Case/Trajectory message text,
including metadata and headings (not the separately supplied existing Experience or system
instructions). It always retains current Trajectories. Each non-success current Trajectory
may add one ANN-ranked same-Case historical success; an all-success chunk makes no historical
query. The selected records are deduplicated by URI and admitted while space remains. Historical
failures, partial, unfinished and unknown outcomes are never used as witnesses. No eligible
ANN hit means the batch proceeds without an extra witness, not a directory-scan fallback.
This check applies to each trainer chunk. A chunk that still exceeds the budget fails rather
than silently truncating or dropping current Trajectories; count-based batching does not
imply that arbitrarily long records fit the character budget.

The comparison prompt distinguishes successful runs, failed runs and unknown/partial runs. Without
a successful run it preserves passed subcriteria and can return no-op. A grading literal alone is
not a universal rule; conflicts with user intent are retained as uncertainties. A successful run is
not proof of a whole SOP. The update focuses on evidenced differences in scope, timing or decisions.
Concrete entity IDs from a Session remain Trajectory evidence and must not appear in generated
reminders. The proposal boundary deterministically turns a candidate that copies such an ID into a
repairable extraction rejection; amounts and policy thresholds are not treated as IDs by this check.
Compact evaluator criteria are stored with each Trajectory as evidence for synthesis. Concrete
runtime results such as calculated totals remain evidence rather than literal requirements in the
Experience; the synthesized reminder should preserve the grounded scope, timing, and decision rule.

Concurrent Sessions still extract and persist Trajectories independently, but Session commit no
longer generates an Experience. The Phase 2 caller provides trigger Trajectory URIs or canonical Case names. The server
resolves the affected Cases and consumes their inboxes, including older pending records. Each Case keeps one fixed Experience target, which
may be updated by multiple consecutive chunks in the same invocation.
`pending -> processing -> applied|no_op|failed` is server-owned metadata; processing leases permit
recovery after interruption. The benchmark invokes Phase 2 once after all commit tasks in its
epoch finish and before evaluation. Online callers may invoke it after a quiet window or another
external batch boundary. Inside an invocation, up to ten Case groups execute concurrently by default
(`StreamingPolicyTrainerConfig.max_concurrent_cases`). Each admitted Case uses the existing
`StreamingPolicyTrainer.submit_gradients` count/window batching. Updates are split by
`max_gradients_per_update` (default 16) and execute serially against the latest Experience.
Model planning runs outside the policy-directory lock; fresh-version validation and apply remain
locked. Case-link/snapshot finalizers are serialized and finish each chunk before the next update.
Only that chunk's source Sessions and Trajectories are finalized; later failures preserve earlier
`applied`/`no_op` states and report already completed Experience writes. Queued Cases
remain pending until admitted, and overlapping learning invocations remain serialized. On
cancellation, active workers are joined; unfinished processing records retain their expiring
leases for recovery. Relation writes retain the same policy lock boundary. The
coordinator is process-local in this version: per-record file locks protect writes, but distributed
claim arbitration across multiple OpenViking server instances is not supported.

### Explicit training phases and progress

`OfflinePolicyOptimizationPipeline` orchestrates the two remote-training phases explicitly:

1. `SessionCommitPolicyTrainer.train_rollouts` commits each rollout batch and waits for its
   Trajectories. It does **not** invoke Experience learning.
2. After **all** loader batches in an epoch finish, `RemoteExperienceLearner.learn` receives their
   results and learns once from the deduplicated Trajectory URIs of successful commits. No URI
   means no learning request; the benchmark never falls back to scanning unrelated pending data.
3. Epoch reporting and evaluation follow the learning phase. `train_from_rollouts` uses the same
   explicit two-step sequence. Local `BatchPolicyTrainer` use without an Experience learner is
   unchanged.

Epoch reports expose `session_commit_cost_seconds` and `experience_learn_cost_seconds`, alongside
commit errors and Experience-learning results/errors. The total epoch cost still includes rollout.

With progress output enabled, each completed Session-commit batch prints one aggregate line:
`[TRAIN_COMMIT] completed=240 with_traj=192 without_traj=48 errors=0`.
`completed` counts successful non-skipped commits; `with_traj` and `without_traj` partition those
commits by whether they returned any nonempty Trajectory URI. Multiple URIs count as one producing
commit. Failed and skipped commits are excluded from both output counts; `errors` counts failed
commit records, not distinct error messages. No-output commits are not reclassified as failures.
The same `commit_summary` is stored in batch results and aggregated across commit batches in the
two-phase learner and epoch report. Existing retry/skip behavior and error lists are unchanged.

Both training phases use lenient extraction handling. When `strict_extract_errors=false`,
Session memory/Trajectory extraction failures (including `ExperienceTrainingError`) are recorded
in Trace and return an empty extraction result. If the other commit steps succeed, the commit
completes normally and retains its raw Session archive; it does not fabricate a Trajectory.
Explicit strict callers still receive the exception. Cancellation is never swallowed.

Experience-learning Case failures remain `failed` in the server-owned Trajectory state and API
result, preserving diagnostics and `retry_failed` behavior. The training adapter treats these
attempts as **skipped**, not applied: they advance processed progress without a red failure count
and do not populate training `errors`. Console summaries report `skipped`; model/parse exception
stacks are trace-only within the extraction call tree, without silencing unrelated concurrent
requests. Task-level failures are different: submission/polling errors, failed or cancelled tasks,
and malformed completed results abort client training before evaluation or the next epoch.
Their final event retains the task ID and incomplete/diagnostic metadata; no normal epoch result
is returned. Cancellation propagates, and interrupted waits never force the progress bar to 100%.

The learning API defaults to `wait=true` and returns the final result. With `wait=false`, it returns
`{"task_id": "..."}` immediately after task creation. The existing `GET /api/v1/tasks/{task_id}`
endpoint provides owner-scoped task status, final results and live `meta.progress`:

```json
{"total": 30, "completed": 12, "running": 10, "applied": 8, "no_op": 3, "failed": 1}
```

The counts refer to **Cases**, not Trajectories. They are initialized after filtering pending
records and applying `max_cases`; completion is published only after a Case's terminal Trajectory
states have been persisted. Both no-op and failed Cases count as completed work. Queued learning
requests have no known total until admitted; no artificial progress is emitted.

The benchmark submits once with `wait=false` and the API's existing `max_cases=1000` limit,
polls the task (every two seconds by default), and renders the shared progress bar from these counts.
This does not add automatic pagination for epochs above the API's Case limit. Transient polling `ReadError`s use the existing
capped backoff without resubmitting learning. The task wait is unbounded by default
(`timeout_seconds=None`), while individual HTTP requests retain their own timeout. Direct callers
can still set a positive task timeout; expiry aborts training rather than starting evaluation.
A cancelled/timed-out waiter does not cancel server work, but the client must not continue into
evaluation. The recorded task ID permits subsequent status queries. Service shutdown cancels and
joins in-process learning workers before closing storage. This is not an automatic restart/resume
scheduler.

The Phase 2 endpoint accepts trigger Trajectory URIs, canonical Case names, or both:

```http
POST /api/v1/agent-evolution/experiences/learn
Content-Type: application/json

{
  "trajectory_uris": ["viking://user/default/memories/trajectories/case/inbox/20261020_F_abcdef.md"],
  "case_names": ["case"],
  "retry_failed": false,
  "max_cases": 100,
  "lease_seconds": 900
}
```

Trajectory URIs select Case scopes, not a whitelist of files. The server reads each notification's
actual `case_name` (rather than guessing from a portable directory or benchmark task name), validates
that it belongs to that Case, and unions those Cases with `case_names`. All eligible inbox records
in those Cases participate, ordered within each Case by persisted `created_at` and URI, so new notifications do not
jump ahead of old pending work. No unrelated Case or archival directory is scanned. An archived
notification can still wake its Case's inbox through a point read of its deterministic final URI.
An explicitly unfiltered service request discovers all Case inboxes for the current user; the
benchmark does not submit such a request when an epoch has no successful Trajectory notifications.
Already learned observations are not relearned: `applied` and `no_op` records only finish any pending
archival; expired `processing` leases may be reclaimed; `failed` requires `retry_failed=true`. Historical records without the
server-owned `experience_update` state are never re-consumed as pending work. Only records marked
with the current `ordered_markdown_v3` format participate in comparison; older formats are ignored
and are not automatically migrated.

Trajectory `content` is the complete immutable model-facing evidence. At observation creation,
the content template renders a `# Trajectory Record` YAML block with `case_name`, `outcome`,
explicitly failed `evaluation_criteria` and `experience_execution`, followed by the extracted
Session Context and ordered observation body. `trajectory_uri` and `source_session_uri`
remain in internal identity/provenance records, not in the body. `action_scores` and
`action_score_sources` remain internal metadata for action feedback and are omitted
from the body. The full `task_spec` remains in metadata and extraction context,
but is omitted from the body. Metadata also retains every valid rubric assessment. Only entries
with the Boolean `passed: false` appear in the body's rubric list; true, null and missing judgments
are not displayed. Overall `evaluation.passed` and `evaluation.score` also stay in metadata;
`outcome` is the body's status summary. If nonempty diagnostic `evaluation.details` exists,
the body retains only that part under `evaluation`; otherwise it omits the whole block.
The external outcome/evaluation and fixed identity are server-owned;
criterion judgments and effects are model-produced, validated and frozen with their definitions. Metadata remains available for outcome selection,
source validation, attribution and lifecycle management; mutable `experience_update` state
is excluded from the body. Lifecycle writes and retries preserve the body without re-rendering.
Experience synthesis sends each Trajectory as one user message, retaining the Session Context
and dialogue verbatim while compacting its generated header: identity/task metadata and
Experience feedback annotations are omitted, including annotations in previously persisted
headers, and only explicitly failed criterion snapshots are presented. Full results, including true/null,
remain stored in metadata. Missing checks are not proof of success. Case context and the existing Experience
remain separate messages, and the budget counts the compact model-facing view. Formats before v3 are
not promoted or repaired at read time and are excluded from Phase 2 selection.

Trajectory source preparation omits the empty text placeholder of tool-bearing messages,
not their tool records or meaningful mixed text. Independent empty dialogue is preserved.
System rules remain in the single `source_steps` input and are used only for Session Context,
not numbered policy-receipt steps; real policy quotations and assistant explanations remain
observed dialogue. Native and VikingBot Tau2 builders no longer append synthetic `tau2-reward`
user messages. Evaluation travels through `Rollout.evaluation` and OutcomeEvaluation instead.
The internal `evaluation` includes optional `details` for original termination reasons,
non-null evaluator information and errors. Those nonredundant details remain in the body,
so removing duplicate status/score or dialogue does not hide
`max_steps` or evaluation failures. Scores, criteria, confirmations, rejected tool calls and
actual STOP events retain their existing meanings. This changes newly produced observations;
it does not rewrite existing archives, Trajectories or cached rollouts.


Updates read the old source and may apply local patches or a full replacement. The
resulting complete program begins with `dag = workflow(...)`, declares observable trigger conditions with `check(...)`
and reminder leaves with `tell(...)`, and adds relationships through object references. New
Experience programs do not encode the agent's complete action path. Node variable names stay stable
for unchanged behavior. The compiler generates internal IDs.

The interpreter permits literals, lists, dictionaries, variable assignments and the listed
constructors/methods. Imports, loops, arbitrary calls, attribute reads and computation are
rejected. Programs and serialized graphs are limited to 256 KiB; graphs have at most 256 nodes.

Validation runs after the complete program. The final graph must be nonempty, acyclic, have distinct slot names and positive
IDs, and have matching branch targets/predecessors. Single-node templates are valid.
Invalid extraction output gets one repair attempt. The optimizer uses the same construction
and validation path. Storage validates again before accepting replacements or related deletes.
The source body bypasses Markdown content templates and linkification; relations remain in
`MEMORY_FIELDS` metadata.

### JSON-string fields in the Python SDK

Case `rubric` remains a JSON-encoded `string` field. Trajectory `evaluation_criteria` and
`action_scores` use YAML-encoded string fields.
In the final Python SDK program, put valid JSON in a raw triple-single-quoted string so Python
does not consume its JSON escapes:

```python
json_text = r'''[
    {"id": "confirmation", "description": "Obtain an explicit \"yes\" before the action."}
]'''
# Pass json_text to the appropriate JSON-string field in a listed write method.
```

Use JSON `true`/`false`/`null` inside the string. Escape quotes, newlines and backslashes once
for JSON (`\"`, `\n`, `\\`), not again for Python. If the text contains literal `'''`, encode an
apostrophe as `\u0027` instead of `\'`. Ordinary SDK text arguments retain their existing quote
convention. Raw strings are already supported by the restricted Python DSL; no serializer helper,
new field type, imports, or automatic JSON repair is added. The outer JSON-mode protocol is unchanged.
Rubric IDs, evidence types, missing entries and judgments still follow their domain validation
and existing nonblocking commit behavior.

### Trajectory rubric results as YAML

Trajectory `evaluation_criteria` remains a string field, but its model-produced value is a
YAML list of `id`, `passed`, block-scalar `evidence`, and optional `description`. Generate
these judgments using the supplied Case rubric as guidance and the conversation in the existing extraction call; ordinary Sessions do
not need precomputed results or an external evaluator. For the Python SDK:

```python
assessment_text = r'''
- id: confirmation
  passed: false
  evidence: |
    The tool updated the record before the user said "yes".
'''
# Use an actual Case rubric ID and pass assessment_text to evaluation_criteria.
```

Safe YAML parsing preserves quoted and multiline evidence without nested JSON escaping.
The server attaches the exact supplied criterion descriptions and retains the existing
best-effort validation: malformed results do not block the factual Trajectory. Full snapshots
remain structured in metadata; both the Trajectory body and Experience synthesis show only
`passed: false`. This display projection does not change what the LLM assesses.
Overall external outcomes/scores remain independent; without them the overall outcome is unknown.

### Materialized content validity gate

Experience `content` uses `merge_op: patch`: local `edit`/`drop` operations and complete
`overwrite` replacements are supported. Synthesis and fixed-target merge prompts follow the generic
SDK policy: prefer the smallest unique local edit; use overwrite only for broad rewrites. Create
still requires complete source. The persisted result, not each edit snippet, must be a complete DAG.
Both JSON and Python edits use the Pi engine pinned in `memory-edit-prompt.md`:
exact lookup followed by finite Unicode/whitespace normalization, not similarity scoring.
All blocks in a field batch match the same original content; overlapping ranges or any failed
block reject the batch before writing. On normalized matches, Pi's touched-line overlay can
also normalize unselected characters on those lines; unchanged lines are retained. The final
DAG still passes the type-owned content gate. Preview and locked persistence share this engine.
If the field changed after it was read, anchors must still match exactly to protect concurrent
changes. Separate resolved operations retain the store's existing staged-view semantics.
The generic SDK contract owns the local-edit example and method selection rule. Experience
examples show a simple check/tell graph in a compact create call and a complete
ask/call/choose/tell graph as read-only field content, not an overwrite template. No-op is a
one-line instruction. Generated SDK text arguments use outer `'''...'''` strings; embedded DAG text
literals use `"""..."""`, which need no outer-layer quote escaping. Plain strings are the default; use
`f'''...'''` only for interpolation and escape literal braces. Escape literal outer delimiters,
trailing quotes that would join the closing delimiter, and backslashes that must survive decoding.
This is a generation convention, not a parser restriction: other valid quote styles remain accepted.
Do not reformat saved content or copied search/drop anchors to enforce it. The shared generation
and repair prompts demonstrate the same convention. Experience synthesis and fixed-target merge
omit only the cross-memory link example, not link rules, parser support, or validation; other
extraction contexts keep that example.
SDK editing and binding rules are defined by the output protocol, with fixed-target details in
the Experience provider. The schema owns node selection, entity-ID generalization, preservation
of unaffected graph structure, ordinary AND semantics, and explicit `any_of` joins. The provider owns trajectory evidence scope
and attribution, rather than repeating those schema rules.

The common path is **schema field merge → type-owned content gate → persistence**.
`session/memory/validation/content.py` registers type-owned validators; the Experience
implementation in `session/memory/experience/compiler.py` validates and canonicalizes full source.
ExtractLoop and MemoryUpdater dispatch the gate without knowing DAG syntax or special-casing Experiences.
This deterministic validity gate is mandatory and does not call a model.

Create-only collision checks live in `session/memory/validation/create.py`. These mandatory
checks are shared by extraction and persistence and are repeated under the write-set lease.
Experience candidate gates live in `session/train/gates/` and share the `PolicyUpdateGate`
contract. The built-in `ExperienceIdentifierGate` is temporarily disabled by default because its
identifier heuristic also rejects policy terms such as `24-hour`. Entity-ID generalization remains
required by the Experience schema prompt. Custom validation gates can use the same rejection/repair
contract. There is no built-in model-based quality replay;
mandatory create/content checks at persistence time remain in place.

Extraction previews reuse the existing operation materializer and the same gate, allowing the
normal repair opportunity on invalid content. A valid patch remains a patch in the resolved
operation: preview must not turn it into a stale full replacement. For ordinary memory, if
repair remains invalid, invalid upserts are omitted and destructive follow-up operations are
suppressed. Experience synthesis instead validates the complete materialized candidate through
its provider hook in ExtractLoop: structural/target errors and enabled gate rejections share one
repair opportunity. Concrete violations are fed back to the generating model; a second invalid
candidate raises rather than becoming `no_op`. Evaluation infrastructure failures propagate
without asking the generating model to repair them. An explicit model no-change remains valid.

For gated writes, the updater acquires/reuses the complete write-set lease, reads the latest
content, stages sequential patches to each URI, and validates every candidate before any write
or dependent delete. It then writes the already validated bytes under that lease. Unrelated
concurrent changes survive; stale/ambiguous patch anchors and invalid resulting graphs fail
without falling back to prefetched content. Exact patch application is enabled for gated types;
ordinary ungated memory types retain their existing merge/error behavior. Training proposals
also use this materializer to obtain full before/after candidates, retaining the existing base-version
and commit-time stale-plan checks. These persistence checks remain mandatory.

## Server-side execution

Use the existing search endpoint with `mode="advice"` and send the current conversation as
`state`. No server-side Session is created, loaded, or required:

```http
POST /api/v1/search/search
Content-Type: application/json

{
  "mode": "advice",
  "query": "Which rules should I review for the current request?",
  "state": [
    {"role": "user", "content": "Please check order 123."},
    {"role": "tool", "name": "query_order", "tool_call_id": "call_1", "content": "Order 123 is shipped."}
  ],
  "experience_names": []
}
```

The `list` and `context` search modes keep their existing behavior. Advice mode accepts only
`query`, `state`, `experience_names`, `limit`, `score_threshold`, and `telemetry`; `session_id`,
arbitrary target scopes, and unrelated search/assembly options are rejected rather than ignored.
Authentication and Experience access are scoped by the request context. The former Session
advice endpoint is removed, not retained as an alias. The Python SDK exposes advice through
`search(..., options={"mode": "advice", "state": ...})`; pass `limit=3` for the usual advice limit.

The server searches the current user's Experience directory for up to 10 vector candidates,
keeping the requested vector `score_threshold` (default 0.15). It evaluates Experience relevance
and the candidates' latest DAG conditions together against the current context. Relevant
Experiences are ranked by their relevance scores, with at most `limit` retained (default 3).
`matched_experience_uris` and `experiences` describe those selected templates in relevance order,
not every vector candidate. Current `instructions` come only from actions reached by their DAGs.
An explicit `experience_names` request bypasses semantic selection and evaluates only the named
DAGs' nodes, as before. Detailed per-call evaluation results remain in `experiences` for observability. There is no
persisted evaluation revision, active-candidate cache, delivery state, or server-side advice dedup.
`state` is one list of current session messages, preserving roles, order, tool names, call IDs,
arguments and results. There is no runtime `context`/`evidence` pair, generated evidence ID, or
256-record limit. The API rejects the old fields and limits compact serialized state to 131,072
characters. Both evaluation paths select recent messages within `max_state_chars` (default
32,768), including JSON overhead and escaping. One oversized latest message is clipped with an
explicit marker; roles, tool names and call IDs are not clipped. A tool call alone is not proof
of execution. Jev's existing full-request token budget and question batching remain in place.

The DAG decision provider is selectable. `jev` uses the reusable System One service:

```json
{
  "jev": {
    "api_url": "https://api.typesafe.ai/v1/systemone",
    "api_key": "...",
    "model": "jev-latest",
    "max_input_tokens": 28000,
    "timeout": 30,
    "verify_ssl": true
  },
  "agent_evolution": {
    "dag_decider": {
      "provider": "jev",
      "relevance_threshold": 0.7,
      "noul_true_threshold": 0.7,
      "noul_false_threshold": 0.3,
      "choice_confidence_threshold": 0.5
    }
  }
}
```

`vlm` uses the account's configured language model instead and does not require a `jev` block:

```json
{
  "agent_evolution": {
    "dag_decider": {
      "provider": "vlm",
      "relevance_threshold": 0.7,
      "noul_true_threshold": 0.7,
      "noul_false_threshold": 0.3,
      "choice_confidence_threshold": 0.5,
      "max_state_chars": 32768
    }
  }
}
```

Both providers implement the same `evaluate(state, questions)` interface. Semantic recall adds
one Score relevance question per candidate to the existing non-Tell node questions, submitting
both in one logical batch. The shared `state` remains the current conversation, not previous
answers; candidate descriptions, conditions and reminders belong in questions, not in the
observed facts. Relevance measures relative priority for current unresolved needs, not the
probability of a yes/no relevance statement. Its five ordered Score criteria range from unrelated
(0) through background (1), indirect/later help (2), direct secondary help (3), to the most direct
current guidance (4). Fractional scores are preserved and divided by four to expose [0, 1]
relevance scores. Candidates can be compared together; genuinely equal priorities may tie.

Relevance and node truth are independent: a relevant Experience cannot make its conditions
true, and a true condition cannot make an irrelevant Experience selected. A finite relevance
score in [0, 1] must reach `relevance_threshold` (default 0.7); selected Experiences are ordered
by score, preserving vector order on ties. Missing or malformed relevance answers do not fall
back to vector-only selection. Relevance scores and selected URIs are recorded in the decision
trace, never persisted as an Experience invalid/completed list. Every request re-evaluates them.
Changing this relevance cutoff does not change Noul node truth thresholds, Choice confidence,
or the separate vector `score_threshold`. Lowering it cannot recover a candidate below the
Top-N rank cutoff; it only admits lower-scoring candidates into that ranked set.

Node questions retain their original Noul/Choice types, and only the selected Experiences'
node answers are applied. The deterministic DAG executor, not the model, then selects the next
actions. Conditions for unselected candidates are evaluated but discarded, avoiding a second
serial model call. Named-target evaluation still asks only node questions.
Client-side delivered-advice deduplication is unchanged.

With `jev`, questions are kept in one request while the conservative estimated input is within
`max_input_tokens`; larger sets are bisected without duplicating or truncating questions. An HTTP
413, or a 422 explicitly reporting a context-length/input-token overflow, triggers the
same split. Boolean nodes become Noul questions and conditional branches become Choice questions.
Jev never generates business values: identifiers and tool-result objects stay in the session
messages. Missing, malformed or uncertain answers leave nodes pending. Transient Jev failures do not
invalidate the Experience and never fall back to slow generative slot filling when the Jev
provider is selected. With `vlm`, the same typed question contract is sent to the account model;
malformed or incomplete responses report errors for this request without banning later attempts.

The in-repository HTTP client also provides:

```python
result = await client.search(
    query="What should I verify before handling this order?",
    limit=3,
    options={
        "mode": "advice",
        "state": [{"role": "user", "content": "Please check order 123."}],
    },
)
```

The caller supplies an explicit `query` for retrieval and the current session messages as
`state` for DAG decisions. Jev receives this state directly alongside the generated `questions`. The server does not derive the query from conversation history. The server never
executes the business tools described by `CallTool`.

Memory retrieval is control-plane behavior and is never represented as a DAG node. In
particular, `call("search_advice")`, `call("search_exp")`, `call("search_experience")`, and
`call("read_experience")` are rejected by the compiler.

### Advice applicability and presentation

Each `instructions` entry returned by Experience search carries structured `conditions` for
its reached predecessor path. Each entry has `description`, `operator` (`is_true`, `is_false`, `equals`, or `not_in`),
and `values` for choice labels. Entries form one sufficient activation path: ordinary AND joins
include all required inputs, while `any_of` includes one actually satisfied input path, not every
alternative. Logical joins themselves are not model judgments or evidence facts. A condition's
original text, including any internal AND/OR, is not rewritten.
False branches negate the whole predicate; choice branches retain the selected label. An
explicit default branch lists the excluded named choices, not an unknown or missing answer.

Only ancestors of the reached advice are included, with shared ancestors listed once.
Unrelated/inactive branches are not included. Prior Tell nodes are traversed for ancestry but
never rendered as a condition that another reminder was delivered. Legacy information/tool
prerequisites retain their completion meaning. A root advice has no additional conditions.
This description is derived deterministically from the existing DAG evaluation, with no new
model call and no claim that the applicability text is independently verified ground truth.

The server API remains structured JSON. The Tau2 client's model-facing message is Markdown:

```markdown
[Experience Reminder]

### 建议 1

**适用条件**
- 用户正在询问即将出行航班及其总费用。

**建议**
核对统计范围，纳入当前有效、尚未出发的相关预订。
```

Each advice has its own section; multiple conditions are explicitly conjunctive. The client
preserves the original instruction Markdown and keeps query text, scores, Experience names,
slot names and other machine metadata out of the presentation. The marker remains unchanged
so advice is still distinguished from user/tool evidence in decisions and trajectory extraction.

Formatting and condition changes do not alter delivery identity: it remains
`(Experience URI, stable slot name, instruction text)`. Markdown is mapped back to those pending
identities only after actual context insertion. A provider with a delivery callback owns this
identity-based deduplication; the agent loop must not collapse distinct advice merely because
their display strings are identical. Providers without that callback retain text deduplication.
Updated server and Tau2 client code must be loaded together for the new response field.

### Tau2 integration

The Tau2 bridge no longer calls `ensure_session()` for advice. Its `runtime_id` (prefixed
`tau2_advice_`) is only a client-local rollout correlation key and a suffix in the local bot
conversation key. `dag_runtime.runtime_id` replaces the misleading server `session_id` field.
Local delivered-advice persistence is unchanged; normal `tau2_train_*` session commits still
create sessions for trajectory extraction. Existing empty sessions and cached rollout artifacts
are not migrated or deleted. Server, Python SDK, and Tau2 processes must reload together.

The Tau2 VikingBot executor loads no Skills and exposes no manual advice tool. Train and
test automatically recall advice for the opening user request and each real user reply,
before the next model decision. Replies are registered after their actual delivery into
model context; internal prompts, ordinary tool results, advice messages, errors, and user
simulator terminal markers do not trigger recall. Multiple queued user replies are processed
sequentially, recording each actual reminder insertion in the client session before the next recall.
A proposed batch containing a write, transfer-to-human, or done tool adds a pre-execution checkpoint.
If it yields new advice, no tool in that batch executes and the model decides again with the advice.
The discarded proposal is not appended to chat history, avoiding unmatched tool-call messages.
If there is no new advice, the batch executes normally; read-only batches skip the extra recall.

The runtime composes a bounded query from the latest user message, the preceding delivered
assistant message, and the original request. There is no query-rewriting model call or fixed
Case-name targeting. It sends one filtered, bounded copy of the current conversation as
`state` to server-side search with `mode="advice"`. It retains policy, historical advice and
ordinary tool-call/result associations, excluding only synthetic reflection prompts and private
reasoning. Plaintext assistant messages carry their known `delivered_to_user` flag. No model
summarizes or extracts another representation. Embedded image data is redacted before submission.

Only nonempty advice is inserted as a `[Experience Reminder]` message; empty results and
temporary `unavailable` responses do not block execution. Advice contains applicable review
instructions, never complete DAG programs. The evaluator is instructed not to treat historical
advice as authoritative policy, user confirmation, or proof of completed actions. Only actual insertion permits
client-side delivery recording. A unique client conversation and local reminder deduplication
remain in place per rollout; no acknowledgement is sent to the server. `advice_recalls` records each query/status/count/duration in rollout
metadata, including the user-message or before-action trigger. Injected messages retain their
tool-history positions. At actual insertion, their export copies also capture the ordered advice
`slot_name`s. The archived reminder header is annotated as `[Experience Reminder] advice_a, advice_b`,
while its full instruction body remains available to extraction. Trajectory reminder steps copy
that header rather than paraphrasing it or inventing names. Online Agent messages, captured raw
advice text, recall and deduplication are unchanged; this is an archive annotation, not a new
injection or a step-count validator. `dag_runtime` keeps evaluations and separate `delivered_advice` facts;
client delivery does not manufacture completed Ask/Call nodes. Existing retrieval thresholds and DAG rules are unchanged.

When a rollout is committed, the opening CaseSpec contains only static concrete-task input and rubric data.
The final OutcomeEvaluation control message carries two sibling fields: `evaluation` and
`experience_execution`. The latter is the final runtime snapshot keyed by Experience URI:
current evaluation state, slot values, evaluated/current nodes, actions,
completed condition nodes, action outcomes, and client-recorded `delivered_advice`. Server
node state is not a delivery receipt; `delivered_advice` identifies actual context insertions. Node arrays use stable `slot_name` values, which are the Python variable
names. These observations are passed directly to the reflection stage. They are not independently
verified business outcomes; `evaluation` remains the authoritative external result. Missing
evaluation is represented as unknown, never converted into a passing result.

Compilation, fixed-target checks, deletion restrictions, source provenance and locked reload/apply
remain mandatory. Structural validation is not evidence that a candidate will improve future runs.

Experiences do not persist `success_trajectory_refs` or automatic `derived_from` links to
Trajectories/Sessions. `source_sessions` and source links remain transient update-plan metadata
for attribution and validation. Successful comparison retrieval uses `VikingFS.find` in QUICK
vector-search mode, scoped to the requester and `trajectories/<case_name>/archive`, with L2-only results
and the existing `search_tags` filter `trajectory_outcome=success`. Trajectory archival attaches
`trajectory_outcome=success|failure|unknown` through the common MemoryUpdater vectorization path;
the value comes from the server-owned outcome, never an LLM-supplied tag. Existing untagged
vectors require explicit tagging/reindexing with tags; queries do not backfill or scan them.

Each non-success current Trajectory supplies a separate query, preferring its stored `steps`
over its rendered management header. The shared head/tail truncation helper caps the query at
2,048 locally estimated tokens; the embedding backend retains its own input guard. This is not
an exact model-tokenizer count. Queries run sequentially inside the existing learning batch
(default 16 signals), not concatenated into one query or fanned out without a limit. Each query
requests at most eight candidate hits and selects the first eligible one; results from different
queries are deduplicated by URI. Only those candidate files are read, with normalized URI scope,
Case, success outcome, content format, provenance, and `applied`/`no_op` state checked again.
Missing hits are skipped; permission/backend errors propagate. No `_comparison_key` equality
check remains: semantic similarity is a candidate-selection signal, not proof that tasks have
identical constraints, and the synthesis model must still compare the actual evidence.

A no-op never rewrites an Experience merely to save a history index. Learning snapshots eligible
work from the selected Case inboxes; it never descends into archives. Notifications determine
which Cases to wake, not which individual observations are allowed to learn. This is consumption
at existing external trigger points, not a new continuous background worker; files arriving after
the work snapshot are picked up by a later trigger. There is no additional successful-anchor regression scan
or model-based replay gate.

Trajectory storage has two lifecycle directories under `trajectories/<case_name>`:

```text
inbox/20261019_S_abcdef.md  ->  archive/2026/10/19/S_abcdef.md
```

The inbox filename starts with the UTC creation date, not the Session date or learning date.
`extract_context.get_random_id()` supplies six lowercase letters/digits, reused within one
context; `trajectory_name` retains this short ID. The server pins the naming timestamp for a
joint extraction. Each invocation creates new Trajectories, including repeated source archives.
Creation performs only point checks in inbox and at the final archive address, with create-only
validation under the exact destination-file lock (plus an overview sidecar only if configured).
Neither same-source deduplication nor a Case-directory tree lock is added.

Files retain their `experience_update` state. Pending, processing and failed observations stay
in inbox; there is no failed directory.
Learning uses the configured `max_gradients_per_update` (default 16) as the claim, synthesis,
and failure boundary, not only as an LLM input limit. Within a Case, claim one batch with its
own batch ID and lease, process and finalize it, then claim the next. A short final batch is
submitted immediately rather than waiting for the streaming timer. If a batch fails, stop that
Case for this invocation and mark only its claimed, non-terminal records failed; later records
remain pending with unchanged attempts and leases. Other Cases may continue concurrently.
A later trigger for the same Case, even if it contains only new Trajectory URIs, includes the old
pending tail without `retry_failed`; failed records still require explicit retry. Returned
`trajectory_uris`/`trajectory_count` cover claimed records, not the unclaimed tail. Benchmark events
separately record `trigger_trajectory_count` and `selection_mode=case_inbox`; the trigger count is
not the amount of work in inbox. If waiting is interrupted, the unknown learned count is null. Successful earlier batches and their snapshot provenance are retained.
All observations in a completed learning chunk are marked
`applied` or `no_op` before archival begins. Archival uses existing `VikingFS.cp` and `rm` operations
under exact source/target file locks: copy to the dated archive, enqueue embedding there, then
remove inbox. This split preserves the inbox recovery record until indexing has been submitted.
An occupied archive filename must represent the same immutable observation and terminal state;
unrelated evidence is never overwritten. Discovery of a terminal inbox record retries archival,
not learning. An explicit submission retaining an old inbox URI resolves its final archive URI
by filename, without scanning history or leaving redirect files.

Inbox writes do not enqueue embedding. The first embedding job uses the final archive URI, so
an asynchronous job cannot later recreate an index record at a removed inbox URI. Outcome tags
and account-specific embedding templates are retained. Once enqueue is recorded in archive
metadata, retrying source deletion does not enqueue again. A process crash between enqueue and
recording that marker may duplicate the same-URI index upsert; this is not an exactly-once queue
protocol. The existing window between committing an Experience and persisting terminal states
is also unchanged. No second embedding is required just to change the path or filename.

Existing files/account templates are not automatically migrated. Old layouts need an explicit
migration before being used with this inbox/archive learning lifecycle; no legacy directory scan
or persistent filename/source index is introduced.

`SessionRolloutAnalyzer.analyze_many()` sees all associated Cases and the shared source steps in
one extraction conversation. Each target carries its own rubric, evaluation, and injected actions.
The model must assign Case-relevant facts without mixing another Case's exclusive actions; shared
facts may appear in multiple trajectories when relevant. The complete output Case set and its
per-Case cardinality are checked before any write. Missing, duplicate, or unexpected Cases receive
normal bounded ExtractLoop repair feedback; there is no successful-subset write from an invalid
model reply. Zero Cases make no trajectory model call. A normal valid multi-Case reply uses one
LLM request, while tools/format/provider retries may add requests. Per-file writes retain existing
failure semantics rather than introducing an all-files transaction. The generic single-rollout
`analyze()` interface delegates to the same joint implementation with one Case. Custom analyzers
injected into `SessionCompressorV3` implement `analyze_many()`; there is no per-Case fallback.

Case and Experience names correspond one-to-one; a Trajectory's Case directory supplies its
membership. Cases persist no `links`/`backlinks` and their template has no linked-Experience
section. Trajectories retain only the `derived_from` Session link and no backlinks. Their
`experience_update` keeps execution state but not a redundant `experience_uri`; recovery and
reporting derive that URI from the canonical Case name. Experiences retain `case_name` alongside
`experience_name`, without maintaining a separate relation list.

The common serializer removes obsolete relationship fields and both ends of Case/Trajectory
relations on normal rewrites. Relation writers reject these edges before endpoint I/O; synthesis
no longer performs Case-link backfill or relation-only rewrites. Other ordinary-memory relations
remain supported. There is no bulk rewrite of historical files. Raw evidence and evaluator feedback
are not duplicated into Experience metadata, and source Session archives are never modified to
add backlinks. Snapshot messages use `OpenViking-Experience-Session-Map`; submitter memory diffs
and source diagnostics remain scoped to the originating Session and Case.

Optional skill extraction runs separately and does not generate Trajectory memories. An explicit
account-level Agent Evolution `false` disables Case/Experience learning. Tau2's no-memory mode
also disables automatic Experience recall. Candidate-validation errors follow the existing
strict/lenient commit error policy; no rejected quality-replay candidates are added to memory diffs.

## Semantics and state

- A source may fan out to many nodes with `source.then(a, b)`. Multiple predecessor edges
  to an ordinary node form an AND join: all incoming paths must be reached in this evaluation.
- `either = any_of(a, b)` creates a native `AnyOfNode` with OR inputs. It has no slot provider:
  the runtime traverses it locally when any incoming path is satisfied. False, unknown or
  unreachable checks do not satisfy the join. OR results are recomputed, never latched across requests.
  The node is excluded from JEV/LLM slot questions, advice and external completion facts.
  Nested joins are supported; empty or duplicate inputs and cycles are rejected. Branch selectors
  retain their routing semantics: join their selected path nodes rather than the selector itself.
  For example, `(a OR b) AND confirmed` is expressed as:

  ```python
  either = any_of(a, b)
  either.then(reminder)
  confirmed.then(reminder)
  ```
- Non-choice conditions and evidence-backed AskUser/CallTool nodes use booleans. TellAgent
  nodes emit advice; models cannot mark them delivered. A Tell does not act as a cross-request
  gate for subsequent advice. The server may return the same applicable advice on every call.
- IfElse and Conditional branches are evaluated from current facts, including false/unknown
  transitions. A null branch target ends only this evaluation's path, not future eligibility.
- The server reads the current template each time. It does not read/write `.experience_instances`,
  retain old candidates, pin a template per session, accept delivered-ID filters, or track receipt
  state. Session existence and user ownership checks remain enforced. Evaluation does not take
  state locks; ordinary Experience memory writes retain their existing concurrency protection.
- Only the client deduplicates advice. Its local SessionManager persists
  `metadata.delivered_experience_advice` after actual context insertion, using the Experience URI,
  stable slot name and instruction text. Numeric node IDs can change with template order and
  are not durable delivery identity. Different or updated advice remains eligible.
- Constructing a note is not delivery. A client write failure is reported and does not undo the
  already inserted note or its in-memory dedup; saving is attempted again on later delivery/close.
  Invalid persisted delivery state disables advice for that rollout rather than silently clearing it.
- Every fresh Tau2 rollout has a unique client conversation key, including across experiments.
  Reopening the same client session preserves dedup facts; a separate conversation is independent.
- The delivery ACK route, SDK methods, retry queue and `delivery_id` are removed. Deploy the
  client and server together. Existing server delivery snapshots are left untouched but unused;
  no old-server-to-client delivery-state migration is performed.
- Graph/slot validation remains enabled. Runtime decisions no longer emit synthetic evidence
  references. Trajectory evidence retains its existing format and budget. Delivery is not business
  compliance or success; training must distinguish client `delivered_advice` from server evaluation.
- Automatic conversion of legacy text and tool execution remain outside this release.

## Checks

```bash
.venv/bin/pytest tests/session/memory/test_experience_dag.py tests/service/test_experience_runtime.py -q --no-cov
```

These checks include the Python extraction protocol, repair, graph materialization, file
serialization and server runtime in one flow, using a mock VLM and storage. They also cover
invalid programs/graphs, branches, default-instance validation, stateless reevaluation, client restart recovery, isolation,
and concurrent state updates. They do not measure live-model slot-filling accuracy.


## Action 有用性反馈

Trajectory 提取在同一次模型调用中生成 `action_scores` YAML 列表。每个实际注入的
不同 action 分配一个 `A1` 等编号，输出 `id`、简短 `evidence`，可评分时给出
`score: 1`（有证据的用户目标收益）或 `score: -1`（有证据的副作用）。判断目标是用户当前的
合法请求，不是 action 自己要求做什么。仅仅遵循提醒、未采纳、不适用或整题成败，都不足以定分。
正分需要纠正决策、补充必要信息或推进任务等实际帮助；负分需要非预期成本/操作、错误变更、
任务偏移或进展阻塞等副作用证据。有帮助且有副作用时记 `-1`，证据同时说明两者，供后续保留
有效行为并修正有害部分。无明确影响或证据不足时省略 `score` 并说明原因，不使用 `0` 或 `null`。
遗漏、非法或重复评分保留为未评分，不阻断事实轨迹提取。

编号只来自运行时 `delivered_advice`，不把候选、ACK 或待执行节点当成实际注入。
`action_score_sources` 由代码绑定到 Experience URI、稳定 `slot_name` 和真实注入文本，
与不可变 Trajectory 一起保存；模型不能改写绑定。同一轨迹中相同 action 只计一次。

Experience 的系统字段 `action_feedback` 以 `slot_name` 为键累计 `useful_count`、
`not_useful_count`、`unscored_count`，并保留每类最新的有界证据和来源轨迹。
不计算内容指纹；同一 Experience 内同名 slot 的反馈跨建议文本修改持续累加，改名后独立计数。
不同 Experience 的同名 slot 不合并。每个实际注入的不同文本仍独立评分，再汇总到对应 slot。
统计项不重复保存 slot 名或建议文本；当前建议文本从 DAG 读取。
此统计格式仍为 `schema_version: 2`，不自动转换旧的指纹键统计。
此次仅收紧新评分口径，不重评旧 Trajectory、不改写累计计数。`not_useful_count` 保留原名，
历史负分可能仅表示冗余或不适用，不能直接当作副作用次数或副作用率。旧正分也不能证明新增
条款有效；更新模型必须核对证据。比较新口径的效果时应使用独立的新实验数据，不混用旧评分。
计数通过既有 `MemoryUpdater` 持久化，不重新嵌入未改变的 DAG；Experience 不保存
`observed_trajectories` 列表。反馈处理锁住当前 Trajectory 及受影响的 Experience，完成后在
Trajectory 的 `experience_update.feedback_recorded` 写入一个布尔标记，缓存复用和正常并发
调用据此跳过重复统计。计数和完成标记不是跨文件原子提交：部分写入失败或计数写入 ACK
丢失后重试，可能重复计数；不为此维护无限增长的去重账本。普通 DAG 更新保留最新反馈，
已删除的 Experience 不会被统计重建。旧列表在 Experience 下次正常写入时移除，不扫描改写
历史文件，也不自动给历史 Trajectory 补写完成标记；重放旧数据可能再次计数。

exp 更新输入中的 `# Existing Experience` 展示完整 DAG，并将只读 `action_feedback` 压缩为两组。
沿用当前按 `slot_name` 累计的全部反馈，不区分评分版本，action 文本变更后不清零。
只检查当前 DAG 仍存在的 action，以 `useful_count + not_useful_count` 为样本数，排除未评分项：

- `unchanged`（保持不变）：默认至少 10 次评分，正反馈比例 ≥ 90%。提示模型本轮不得删除、
  改名、改写这些 action，也不得改变其触发条件；不只是保留节点。这是提示词约束，不是程序硬锁。
- `revise`（需要修正）：默认至少 10 次评分，负反馈比例 ≥ 30%。核对负反馈证据后修正或删除
  有副作用的部分，保留已证实的收益。不将旧负分直接解释为副作用。

每组按对应比例降序排列，同分按 slot 名排序，最多 3 条；每条仅显示 slot 名、比例（分子/分母）
和该方向的一条现有证据（单行，最多 160 字符；缺失则不展示）。不重复 action 正文，不展示
内部去重记录或轨迹 URI。没有符合条件的 action 则省略反馈区块，未列出不等于无用。例如：

```text
### unchanged
- check_payment_scope: positive 90.0% (18/20) — 提醒后纠正了支付范围。
### revise
- repeat_completed_check: negative 40.0% (4/10) — 重复检查阻止了已获授权的操作。
```

阈值可通过 `ov.conf` 的 `memory.action_feedback` 配置：`min_samples`（默认 10）、
`unchanged_min_ratio`（0.9）、`revise_min_ratio`（0.3）、`max_per_group`（3）。
样本数和展示数量须为正整数，比例须在 `(0, 1]`；阈值重叠时优先归入 `revise`，不同时要求不变和修正。
这些配置只控制模型输入展示，不改变存储统计或运行时 action 执行。
反馈字段不声明为可编辑 schema 字段，SDK、JSON 输出及合并模型不能修改它。
评分是模型的观察性判断，不是因果证明；更新时仍需核对源步骤、适用条件和评估事实。
不据此自动重排或删除 action，不修改 benchmark 分数，不将旧 exp 级标签回填为 action 评分。
