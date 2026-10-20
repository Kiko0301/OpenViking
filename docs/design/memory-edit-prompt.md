# Memory Edit: Pi implementation

The shared text-edit engine and prompt follow `earendil-works/pi` at
`482c1381a00d8d00bc81697e4a980c593db6803e` (verified 2026-10-18), specifically
`packages/coding-agent/src/core/tools/edit.ts` and `edit-diff.ts`.
The MIT copyright and permission notice is retained in `merge_op/patch_handler.py`.
The previous RooCode-style similarity matcher and Claude-derived Edit description
are removed, not retained as fallback paths.

## Necessary memory-SDK adaptation

Pi's file editor is adapted to memory fields. OpenViking keeps its field transport:
`obj.field.edit(search=..., replace=...)` calls and
JSON `StrPatch.blocks` become the edit list; the memory URI plus field name becomes
`path` in diagnostics. All accumulated edits of one field form one batch.
Generated instructions and fixed diagnostics use the SDK parameter names
`search`/`replace`; the former parameter names are not accepted as aliases.
`drop(text=...)` is an edit with an empty replacement, not implicit whole-line
removal. Include the newline in `text` when the whole line must disappear.

The extraction loop, memory persistence/locking, creation, whole-value overwrite,
Case selection and feedback remain framework-owned. This does not introduce
Pi's filesystem tool, legacy JSON argument adapters, native-tool transport or
provider-specific constrained sampling into the Python SDK protocol.

## Matching and application

1. Preserve an initial BOM, detect line endings, and use LF internally.
2. Try exact substring matching first. If absent, apply Pi's NFKC, trailing-space,
   quote, dash and special-space normalization, then search again. There is no
   similarity threshold, line-number stripping, marker unescaping or guessed block.
3. Match every edit against the same original field, **not the result of previous
   edits in the batch**. Validate uniqueness in normalized space and reject
   overlapping/nested replacement ranges. A later edit cannot target newly added text.
4. Apply the located ranges back-to-front. If normalization was needed, use Pi's
   touched-line overlay: write normalized touched lines while preserving other lines.
   Restore the original line-ending style and initial BOM.
5. Empty edits/search, missing or ambiguous matches, overlap, and an unchanged
   final result raise errors. No successful subset is returned on batch failure.

The Python port intentionally retains Pi's touched-line normalization behavior,
including possible changes outside the selected substring on the same line. It
**does not** add a custom mapping back to original character spans. Existing
DAG/JSON content gates still reject invalid results; those gates are not a general
proof that normalization preserves every business meaning.

The memory store's stale-read guard remains: a field changed since it was read
requires exact anchors. Separate resolved operations still use the existing
staged memory view; original-snapshot matching applies to blocks within one batch.
The existing non-strict commit policy is unchanged. Ungated field failures can
still retain their old value; they do not retain successful subsets of a Pi batch.

## Prompt ownership

`extraction_output_protocol/edit_format.py` contains Pi's description and guidelines,
with only memory/SDK vocabulary adapted. Python shows them beside `edit`; JSON
shows the same guidance once beside its schema. Python's `overwrite` method says
to use it only for complete rewrites. Providers and matching-error retries refer
to this shared contract rather than duplicating tutorials.

The generic local replacement example stays beside `edit`. The Experience DAG
example is read-only field content, not an overwrite template; string-encoding
examples construct values without selecting a write method. Python literal
encoding and DAG structure rules remain separate from matching instructions.

## Validation

Tests cover original-snapshot batches, overlap and normalized ambiguity rejection,
no partial results, BOM/CRLF preservation, literal deletion, removed approximate
and line-number matching, SDK/JSON adapters, field-specific preview diagnostics,
retry feedback, stale-read protection, and invalid-DAG rejection after Pi's line
normalization. Differential checks against the pinned original TypeScript core
are separate from model-generation experiments; a passing engine test does not
guarantee that an LLM always generates a valid edit.

## Isolated Edit repair

`memory/edit_repair.py` attempts a dedicated repair for each eligible Edit failure
in either the early provider-validation path or the later content/patch-validation
path. It has no retry counter; `ExtractLoop` owns the retry limits. Each invocation
uses the same VLM, thinking setting and output limit.
The call contains only the actual original fields, their pending Edit batches,
and validation errors—not the extraction conversation, Case or Trajectory history.
An empty tools list exposes no tools while preserving response metadata in the
VLM backends; truncated/error/tool-call responses are rejected.

For missing old text, `edit_repair_candidates.py` reuses the former patch handler's
RooCode-derived normalization, `rapidfuzz.distance.Levenshtein` similarity and
single-line substring search helpers to suggest up to three source spans. This is
diagnostic ranking only: the old 80%-similar automatic replacement path stays removed.
Candidates contain verbatim source text (including mixed line endings), source line
numbers as separate metadata, match counts and up to six character differences.
The repair prompt explains that a candidate is not proof of the intended location:
copy the selected source literally and preserve surrounding content when widening
an edit. Non-unique matches request more context, not arbitrary first-match selection.

Candidate text has a shared 6,000-character budget per repair call. Search examines
at most 100,000 source characters / 2,000 lines and skips search strings over 2,048
characters; candidate spans are also capped at 2,048 characters. Limited searches are
marked, and no candidate does not mean no valid repair exists. Candidates are never
truncated into fabricated copyable text. These limits affect hints only, not the
supplied original field or the Pi execution matcher. No extra model call is added.

The model returns strict JSON with `status: repaired` and indexed replacements,
or `status: cannot_repair` with a reason. Both `search` and `replace` may change.
Only Edit blocks in failing field batches are writable. Unmentioned blocks and
read-only `drop` entries remain unchanged. The server rejects extra keys, duplicate
or invalid indices and non-string arguments; it never executes the response as SDK
code. A deep-copied operation plan preserves counters, sources, identity, links and
unrelated operations. All repaired batches are checked against their original
snapshot with the existing Pi engine and field gates; the full provider/content
checks and locked-write stale-read checks still apply.

A failed, refused or invalid dedicated repair is a pre-retry attempt, not a
replacement for the existing ExtractLoop retry. Its failure reason is returned
with the unchanged original operations and added to the normal retry feedback.
The main loop may regenerate the complete SDK draft using its existing retry
budget. That draft has not been written yet; already committed batches are not
replayed. Each regenerated draft with invalid Edits can trigger another isolated
repair, including after a previous repair was rejected by full content/provider
checks. Each repair invocation makes one model call without retrying internally;
the existing loop counters bound further draft generation. A rejected candidate
is still not accepted or persisted.

After the original retry budget is exhausted, existing terminal behavior remains:
provider rejection raises, gated candidates can be discarded, and ungated failures
keep their previous writer handling. This is not a transaction-policy change.
Create, overwrite, drop-only, semantic-provider and mixed non-Edit errors keep
their existing retry path. Multi-URI or sequential operations without one
unambiguous source snapshot are not guessed at by this repairer.

The `memory.edit_repair` span records input, output and outcome. A
`candidate_validated` outcome means only local batch/content validation passed;
it is not a claim that downstream provider gates passed or business semantics
were proved equivalent. No online success-rate improvement is claimed without a
new experiment.
