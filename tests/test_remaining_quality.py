"""Guard autonomous phase selection and verifiable, credential-free backups."""
import copy
import errno
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_remaining_quality as coordinator


class CoordinatorTests(unittest.TestCase):
    def test_deployment_settings_are_explicit_and_peer_origin_is_validated(self):
        args = ["--rank", "0", "--workdir", "/experiments/quality",
                "--ct-workdir", "/experiments/ct", "--modern-workdir", "/experiments/modern",
                "--peer-url", "http://peer.example:29871/"]
        parsed = coordinator.parse_args(args)
        self.assertEqual(parsed.peer_url, "http://peer.example:29871")
        self.assertEqual(parsed.bind_address, "127.0.0.1")
        for url in ("file:///tmp/peer", "http://user:password@peer.example", "http://peer.example/path",
                    "http://peer.example/?token=secret", "http://peer.example:70000"):
            with self.subTest(url=url), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                coordinator.parse_args([*args[:-1], url])
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            coordinator.parse_args(["--rank", "0"])

    def test_snapshot_excludes_token_links_and_incomplete_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "peer-token").write_text("do-not-copy")
            (root / "token-link").symlink_to(root / "peer-token")
            (root / "peer-backups").mkdir()
            (root / "peer-backups/previous").write_text("do-not-recurse")
            (root / "responses.jsonl").write_bytes(b'{"complete":true}\n{"partial":')
            (root / "empty.jsonl").write_bytes(b'{"partial":')
            (root / "summary.tmp").write_text("unfinished")
            with patch.object(coordinator, "ROOT", root):
                data = coordinator.snapshot()
            manifest = coordinator.verify_snapshot(data)
            self.assertEqual(set(manifest), {"responses.jsonl", "empty.jsonl"})
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                self.assertEqual(archive.extractfile("responses.jsonl").read(), b'{"complete":true}\n')
                self.assertEqual(archive.extractfile("empty.jsonl").read(), b"")

    def test_corrupt_member_and_unlisted_member_are_rejected(self):
        for corrupt, extra in ((True, False), (False, True)):
            out = io.BytesIO()
            payload = b"original"
            members = {"evidence": b"corrupted" if corrupt else payload,
                       "SNAPSHOT-MANIFEST.json": json.dumps({"evidence": hashlib.sha256(payload).hexdigest()}).encode()}
            if extra:
                members["peer-token"] = b"secret"
            with tarfile.open(fileobj=out, mode="w:gz") as archive:
                for name, data in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            with self.assertRaises(ValueError):
                coordinator.verify_snapshot(out.getvalue())

    def test_only_fixed_experiment_names_are_accepted(self):
        for index in (-1, 3, True, "0"):
            with self.assertRaises(ValueError):
                coordinator.name(index, 0)
        for rank in (-1, 2, False, "1"):
            with self.assertRaises(ValueError):
                coordinator.name(0, rank)
        for payload in ([], {"phase": True, "status": "started"},
                        {"phase": 3, "status": "started"}, {"phase": 0, "status": "execute"}):
            with self.assertRaises(ValueError):
                coordinator.validate_peer(payload)

    def test_changed_environment_checkpoint_and_sources_are_rejected(self):
        previous = {"image": "image", "command": ["serve"], "library_sha256": "lib",
                    "package_source_sha256": {"code.py": "sha"}, "boot_sha256": "boot",
                    "exit_code": 0, "environment": {"ZIPSERV_MIXED_FP8": "off", "ZIPSERV_MIXED_BF16": "on"},
                    "mounts": [{"source": "/fixed", "destination": "/model", "access": "ro"},
                               {"source": "/audit", "destination": "/audit", "access": "ro"}]}
        with patch.object(coordinator, "read", return_value=previous):
            check = copy.deepcopy(previous)
            check["environment"].update(ZIPSERV_VALIDATE_ONLY="1", CUDA_VISIBLE_DEVICES="")
            coordinator.check_modern(check)
            check["mounts"].reverse()
            coordinator.check_modern(check)
            for key, value in (("image", "different"), ("package_source_sha256", {"code.py": "edited"}),
                               ("environment", {"ZIPSERV_MIXED_FP8": "off", "ZIPSERV_MIXED_BF16": "off"}),
                               ("mounts", [{"source": "/different", "destination": "/model", "access": "ro"}])):
                with self.assertRaises(ValueError):
                    coordinator.check_modern({**previous, key: value})

    def test_busy_tcp_port_is_waited_for_without_weakening_bind_guard(self):
        with patch.object(coordinator.socket, "socket") as socket:
            probe = socket.return_value.__enter__.return_value
            probe.bind.side_effect = OSError(errno.EADDRINUSE, "TIME_WAIT")
            self.assertFalse(coordinator.ports_idle(1, 0))
            probe.bind.assert_called_with(("127.0.0.1", 18797))
            probe.setsockopt.assert_not_called()
            probe.bind.side_effect = OSError(errno.EACCES, "Denied")
            with self.assertRaises(OSError):
                coordinator.ports_idle(1, 0)


if __name__ == "__main__":
    unittest.main()
