# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Directory-TTL product integration surface.

The open-source edition ships only the default-off plumbing: a stable adapter
*interface* (:mod:`openviking.ttl.adapter`), a ``class_path`` loader
(:mod:`openviking.ttl.loader`) and the service wiring
(:mod:`openviking.ttl.wiring`). No TTL product logic, Redis dependency or
background worker is activated unless a commercial adapter is explicitly
configured, so the open-source server keeps TTL unsupported and returns 501 for
explicit TTL configuration.
"""
