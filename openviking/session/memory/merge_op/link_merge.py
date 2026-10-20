# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Link merge and dedup logic for MEMORY_FIELDS links field.

Dedup key: from_uri + to_uri + match_text
Merge rules:
- Weight conflict: take max
- link_type and description: latest write wins
"""

from __future__ import annotations

from typing import Any, Dict, List

from openviking.core.namespace import classify_uri, is_session_uri


def is_persisted_link(link: Dict[str, Any]) -> bool:
    """Case membership comes from names/directories; only Trajectory→Session is stored."""
    source, target = (str(link.get(key) or "") for key in ("from_uri", "to_uri"))
    types = []
    for uri in (source, target):
        namespace = classify_uri(uri)
        index = namespace.content_index
        types.append(
            namespace.parts[index + 1]
            if namespace.is_memory and index is not None and len(namespace.parts) > index + 1
            else None
        )
    source_type, target_type = types
    if "cases" in types or target_type == "trajectories":
        return False
    if source_type == "trajectories":
        return link.get("link_type") == "derived_from" and is_session_uri(target)
    return not (
        source_type == "experiences"
        and link.get("link_type") == "derived_from"
        and is_session_uri(target)
    )


def _dedup_key(link: Dict[str, Any]) -> str:
    """Compute dedup key for a link."""
    return f"{link.get('from_uri', '')}|{link.get('to_uri', '')}|{link.get('match_text', '')}"


def merge_links(existing_links: List[Dict], new_links: List[Dict]) -> List[Dict]:
    """
    Merge link lists with dedup and conflict resolution.

    Dedup key: from_uri + to_uri + match_text
    Weight conflict: take max
    link_type and description: latest write wins
    """
    link_map: Dict[str, Dict[str, Any]] = {}

    # Process existing links first
    for link in existing_links:
        key = _dedup_key(link)
        link_map[key] = dict(link)

    # Process new links (override existing on conflict)
    for link in new_links:
        key = _dedup_key(link)
        if key in link_map:
            existing = link_map[key]
            # Weight: take max
            existing["weight"] = max(existing.get("weight", 1.0), link.get("weight", 1.0))
            # link_type and description: latest write wins
            if "link_type" in link:
                existing["link_type"] = link["link_type"]
            if "description" in link:
                existing["description"] = link["description"]
            # created_at: keep the original
        else:
            link_map[key] = dict(link)

    return list(link_map.values())
