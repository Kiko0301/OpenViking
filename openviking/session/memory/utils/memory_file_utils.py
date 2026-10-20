from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, Optional

from openviking.core.namespace import is_session_uri
from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.merge_op.link_merge import is_persisted_link
from openviking.session.memory.utils.content_template import render_content_template
from openviking.session.memory.utils.link_renderer import LinkRenderer
from openviking.session.memory.utils.messages import parse_memory_file_with_fields
from openviking.session.memory.utils.template_utils import TemplateUtils
from openviking.utils.time_utils import parse_iso_datetime

# Regex patterns for MEMORY_FIELDS HTML comment
_MEMORY_FIELDS_PATTERN = re.compile(r"\n\n<!--\s*MEMORY_FIELDS\s*\n(.*?)\n-->", re.DOTALL)
_MEMORY_FIELDS_PATTERN_END = re.compile(r"<!--\s*MEMORY_FIELDS\s*\n(.*?)\n-->$", re.DOTALL)

DEFAULT_TRUNCATE_MAX_CHARS = 1000


def memory_version_from_fields(fields: Optional[Dict[str, Any]], *, default: int = 1) -> int:
    """Return a positive MEMORY_FIELDS version, falling back to ``default``."""
    try:
        version = int((fields or {}).get("version"))
    except (TypeError, ValueError):
        return default
    return version if version > 0 else default


def next_memory_version(old_file: Optional[MemoryFile]) -> int:
    """Return the next persisted MEMORY_FIELDS version for a write."""
    if old_file is None:
        return 1
    return memory_version_from_fields(old_file.extra_fields, default=1) + 1


def bump_memory_version(memory_file: MemoryFile) -> None:
    """Increment a MemoryFile's persisted MEMORY_FIELDS version in-place."""
    memory_file.extra_fields["version"] = (
        memory_version_from_fields(memory_file.extra_fields, default=1) + 1
    )


def _serialize_datetime(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _deserialize_datetime(metadata: Dict[str, Any]) -> Dict[str, Any]:
    result = metadata.copy()
    for key in ["created_at", "updated_at"]:
        if key in result and isinstance(result[key], str):
            try:
                result[key] = parse_iso_datetime(result[key])
            except (ValueError, TypeError):
                pass
    return result


def _uri_basename(uri: str) -> str:
    name = str(uri or "").rstrip("/").rsplit("/", 1)[-1]
    return name.removesuffix(".md")


def _template_link_target(source_uri: Optional[str], target_uri: str) -> str:
    if source_uri and target_uri:
        return LinkRenderer.relative_path(str(source_uri), str(target_uri)) or str(target_uri)
    return str(target_uri or "")


def _serialize_with_metadata(
    metadata: Dict[str, Any],
    content_template: str = None,
    extract_context: Any = None,
    source_uri: Optional[str] = None,
    render_links: bool = True,
    account_content_template_type: Optional[str] = None,
) -> str:
    content = metadata.pop("content", "") or ""

    if content_template and account_content_template_type:
        # Managed-template failures must not be swallowed by the legacy plain
        # content fallback: that could silently overwrite a valid Markdown body.
        content = render_content_template(
            content_template, account_content_template_type, metadata, extract_context
        )
    elif content_template:
        template_vars = metadata.copy()
        template_vars["content"] = content
        template_vars.setdefault("links", [])
        template_vars.setdefault("backlinks", [])
        template_vars["source_uri"] = source_uri or ""
        template_vars["uri_basename"] = _uri_basename
        template_vars["link_target"] = lambda target_uri: _template_link_target(
            source_uri, target_uri
        )
        # Trim template source, not interpolated evidence. Render errors must propagate:
        # silently falling back to the raw body can lose required fields and diagnostics.
        content = TemplateUtils.render(
            content_template.strip(),
            template_vars,
            extract_context,
            debug_undefined=True,
            strip=False,
        )

    clean_metadata = {k: v for k, v in metadata.items() if v is not None}

    if not clean_metadata:
        return content

    clean_metadata.pop("_uri", None)
    links = clean_metadata.get("links")
    if render_links and isinstance(links, list) and source_uri:
        content = LinkRenderer.render_links(content, str(source_uri), links)

    metadata_json = json.dumps(
        clean_metadata, indent=2, default=_serialize_datetime, ensure_ascii=False
    )
    # A "-->" inside a value would close the comment early. "\u003e" is the same
    # character to any JSON parser, so the fields still read back unchanged.
    metadata_json = metadata_json.replace("-->", "--\\u003e")

    comment = f"\n\n<!-- MEMORY_FIELDS\n{metadata_json}\n-->"

    if not content or not content.strip():
        return comment.lstrip()

    return content + comment


class MemoryFileUtils:
    """Unified read/write API for memory files.

    Encapsulates parsing + strip_links (read) and serialize + render_links (write).
    All other utilities (deserialize_content, serialize_with_metadata, etc.) are
    internal implementation details not exposed to callers.
    """

    @staticmethod
    def read(raw_content: str, uri: Optional[str] = None) -> MemoryFile:
        """Parse a memory file and return a MemoryFile with markdown links preserved."""
        parsed = parse_memory_file_with_fields(raw_content)
        parsed = _deserialize_datetime(parsed)
        return MemoryFile.from_parsed(uri=uri, parsed=parsed)

    @staticmethod
    def write(
        memory_file: MemoryFile,
        content_template: Optional[str] = None,
        extract_context: Any = None,
        render_links: bool = True,
        *,
        account_content_template_type: Optional[str] = None,
    ) -> str:
        """Serialize a MemoryFile as plain-text body plus MEMORY_FIELDS metadata."""
        metadata = memory_file.to_metadata()
        if memory_file.memory_type == "cases":
            metadata.pop("links", None)
            metadata.pop("backlinks", None)
        elif memory_file.memory_type == "trajectories":
            metadata["links"] = [
                link
                for link in memory_file.links
                if link.get("link_type") == "derived_from"
                and is_session_uri(str(link.get("to_uri") or ""))
            ]
            metadata.pop("backlinks", None)
            state = metadata.get("experience_update")
            if isinstance(state, dict):
                metadata["experience_update"] = {
                    key: value for key, value in state.items() if key != "experience_uri"
                }
        else:
            for field, endpoint in (("links", "from_uri"), ("backlinks", "to_uri")):
                if field in metadata:
                    metadata[field] = [
                        link
                        for link in metadata[field]
                        if is_persisted_link({**link, endpoint: memory_file.uri})
                    ]
        if memory_file.memory_type == "experiences":
            for key in ("source_sessions", "success_trajectory_refs"):
                metadata.pop(key, None)
            feedback = metadata.get("action_feedback")
            if isinstance(feedback, dict):
                metadata["action_feedback"] = {
                    key: value for key, value in feedback.items() if key != "observed_trajectories"
                }
            # Graph JSON is authoritative. Relations remain in MEMORY_FIELDS;
            # Markdown templates/linkification must not mutate executable data.
            content_template = None
            account_content_template_type = None
            render_links = False
        return _serialize_with_metadata(
            metadata,
            content_template=content_template,
            extract_context=extract_context,
            source_uri=memory_file.uri,
            render_links=render_links,
            account_content_template_type=account_content_template_type,
        )

    @staticmethod
    def truncate_content(content: str, max_chars: int = DEFAULT_TRUNCATE_MAX_CHARS) -> str:
        """Truncate content to max_chars while keeping complete lines."""
        if len(content) <= max_chars:
            return content
        truncated = content[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > 0:
            truncated = truncated[:last_newline]
        return truncated + f"\n... [truncated {len(content) - len(truncated)} chars]"
