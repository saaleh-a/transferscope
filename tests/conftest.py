"""Shared pytest configuration.

The suite is documented as running offline with no network calls, but nothing
enforced that.  A test that reached the live ClubElo API through ``soccerdata``
passed on a developer machine and crashed CI with ``Fatal Python error: Bus
error`` — a native crash inside the HTTP stack, which is far harder to diagnose
than a failed assertion.

This module makes the offline guarantee real: outbound sockets are blocked for
the whole session, so any unmocked network call fails immediately with a clear
message naming the address it tried to reach.

Set ``TRANSFERSCOPE_ALLOW_NETWORK=1`` to opt out when deliberately exercising a
live integration.
"""

from __future__ import annotations

import os
import socket

import pytest

_ALLOW_NETWORK = os.environ.get("TRANSFERSCOPE_ALLOW_NETWORK") == "1"

# Loopback stays open: Streamlit, local fixtures and some libraries connect to
# 127.0.0.1 for entirely offline reasons.
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", ""}


class NetworkCallBlocked(RuntimeError):
    """Raised when a test attempts a real outbound connection."""


def _is_local(address) -> bool:
    if isinstance(address, (tuple, list)) and address:
        return str(address[0]) in _LOCAL_HOSTS
    return False


@pytest.fixture(autouse=True, scope="session")
def _block_network():
    """Block non-loopback sockets for the duration of the test session."""
    if _ALLOW_NETWORK:
        yield
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address, *args, **kwargs):
        if _is_local(address):
            return real_connect(self, address, *args, **kwargs)
        raise NetworkCallBlocked(
            f"Test attempted a network call to {address!r}. Tests must run "
            "offline — mock the client, or set TRANSFERSCOPE_ALLOW_NETWORK=1 "
            "for a deliberate live-integration run."
        )

    def guarded_connect_ex(self, address, *args, **kwargs):
        if _is_local(address):
            return real_connect_ex(self, address, *args, **kwargs)
        raise NetworkCallBlocked(
            f"Test attempted a network call to {address!r}. Tests must run offline."
        )

    def guarded_getaddrinfo(host, *args, **kwargs):
        if str(host) in _LOCAL_HOSTS:
            return real_getaddrinfo(host, *args, **kwargs)
        raise NetworkCallBlocked(
            f"Test attempted to resolve {host!r}. Tests must run offline."
        )

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = guarded_getaddrinfo
    try:
        yield
    finally:
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex
        socket.getaddrinfo = real_getaddrinfo


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path_factory, monkeypatch):
    """Point the disk cache at a per-session temp dir.

    Without this, tests read and write the developer's real ``data/cache``,
    so results depend on what happens to be cached locally — another way a
    suite passes on one machine and fails on another.
    """
    if os.environ.get("CACHE_DIR"):
        # A test set its own cache dir; leave it alone.
        yield
        return
    cache_dir = tmp_path_factory.mktemp("ts_cache")
    monkeypatch.setenv("CACHE_DIR", str(cache_dir))
    yield
