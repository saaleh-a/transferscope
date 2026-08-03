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
        """The path that crashed CI must now fail cleanly."""
        from backend.data import clubelo_client

        # _try_soccerdata swallows all exceptions and returns None, so a
        # blocked network should degrade rather than raise.
        result = clubelo_client._try_soccerdata("2026-08-02")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
