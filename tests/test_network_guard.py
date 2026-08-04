"""Confirm the network guard blocks real calls and allows loopback."""
import socket
import unittest

import pytest

from tests.conftest import NetworkCallBlocked


class TestNetworkGuard(unittest.TestCase):
    def test_outbound_connection_is_blocked(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(NetworkCallBlocked):
            s.connect(("api.clubelo.com", 80))
        s.close()

    def test_dns_resolution_is_blocked(self):
        with pytest.raises(NetworkCallBlocked):
            socket.getaddrinfo("api.sofascore.com", 443)

    def test_loopback_is_allowed(self):
        """Local sockets must keep working — only outbound is blocked."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client.connect(("127.0.0.1", port))  # must not raise
        finally:
            client.close()
            server.close()

    def test_clubelo_client_degrades_instead_of_crashing(self):
        """The path that crashed CI must fail cleanly, not raise.

        _try_soccerdata swallows all exceptions and returns None, so a blocked
        network degrades rather than crashing. It may still return data from
        soccerdata's own on-disk cache, which is equally fine — the point is
        that it does not raise or core-dump.
        """
        from backend.data import clubelo_client

        try:
            clubelo_client._try_soccerdata("2026-08-02")
        except Exception as exc:  # pragma: no cover - the failure being guarded
            self.fail(f"_try_soccerdata raised instead of degrading: {exc!r}")


if __name__ == "__main__":
    unittest.main()
