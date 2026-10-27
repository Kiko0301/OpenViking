"""Regression tests: the ``edit`` MCP tool must not mislabel read failures.

``FsService.read_visible`` classifies precisely: a missing file raises typed
``NotFoundError``, a transient backend fault raises AGFS errors (the FS layer
re-raises them with their own types), and the Privacy visible-read size limit
raises ``ResourceExhaustedError``. The sibling ``read`` tool forwards those
typed errors untouched, and ``download`` maps AGFS errors via
``map_exception``; ``edit`` alone re-wrapped every non-auth failure as
``NotFoundError``, telling the agent an existing file is gone.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.pyagfs.exceptions import AGFSTimeoutError
from openviking.server import mcp_endpoint
from openviking.server.mcp_endpoint import edit
from openviking_cli.exceptions import (
    NotFoundError,
    ResourceExhaustedError,
    UnavailableError,
)

URI = "viking://resources/test_edit_read_errors.md"


def _install_read_visible(monkeypatch, side_effect):
    fake_service = SimpleNamespace(
        fs=SimpleNamespace(read_visible=AsyncMock(side_effect=side_effect))
    )
    monkeypatch.setattr(mcp_endpoint, "get_service", lambda: fake_service)
    monkeypatch.setattr(mcp_endpoint, "_get_ctx", lambda: SimpleNamespace())
    monkeypatch.setattr(
        mcp_endpoint, "_resolve_mcp_workspace_uri", lambda uri, ctx: uri
    )


async def test_transient_read_error_is_not_reported_as_missing(monkeypatch):
    _install_read_visible(monkeypatch, AGFSTimeoutError("storage hiccup"))
    with pytest.raises(UnavailableError):
        await edit(uri=URI, old_string="alpha", new_string="beta")


async def test_privacy_limit_error_is_not_reported_as_missing(monkeypatch):
    _install_read_visible(
        monkeypatch, ResourceExhaustedError("File exceeds the visible-read size limit")
    )
    with pytest.raises(ResourceExhaustedError):
        await edit(uri=URI, old_string="alpha", new_string="beta")


async def test_missing_file_still_reports_not_found(monkeypatch):
    _install_read_visible(monkeypatch, NotFoundError(URI, "file"))
    with pytest.raises(NotFoundError):
        await edit(uri=URI, old_string="alpha", new_string="beta")
