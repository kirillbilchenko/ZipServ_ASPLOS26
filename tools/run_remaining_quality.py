# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Bounded, private two-Spark coordinator for the three remaining quality runs.

The peer channel exchanges a fixed phase/status enum and evidence snapshots.
It cannot submit arbitrary commands. Credentials and production containers are
not changed; only the three explicitly named private experiment pairs are used.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import socket
import subprocess
import tarfile
import threading
import time
import urllib.request
from urllib.parse import urlsplit

# Populated from explicit deployment settings before any controller action.
ROOT = Path()
CT = Path()
MODERN = Path()
PHASES = ("ct-off", "ct-on", "fp8-off")
MODERN_TRIAL = 4
PEER_URL = ""
BIND = ("127.0.0.1", 29871)
DEADLINE = time.monotonic() + 6 * 3600


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def name(index, rank):
    if type(index) is not int or index not in range(3) or type(rank) is not int or rank not in (0, 1):
        raise ValueError("Only the three fixed private phases and two ranks are permitted")
    if index == 2:
        return f"zipserv-alltypes-c431talz-v6-on-{MODERN_TRIAL}-r{rank}"
    mode = "off" if index == 0 else "ct"
    return f"zipserv-ct-weight-20261024b-quality20261025a-{mode}-r{rank}"


def inspect(index, rank):
    p = subprocess.run(["docker", "inspect", name(index, rank)], capture_output=True, text=True)
    return json.loads(p.stdout)[0] if p.returncode == 0 else None


def stop(index, rank):
    info = inspect(index, rank)
    if info and info["State"]["Running"]:
        subprocess.run(["docker", "stop", "-t", "30", name(index, rank)], check=True, timeout=90)
    if info:
        phase_label = PHASES[index] + (f"-trial{MODERN_TRIAL}" if index == 2 else "")
        log = ROOT / f"{phase_label}-server-r{rank}.log"
        if not log.exists():
            with log.open("xb") as out:
                subprocess.run(["docker", "logs", name(index, rank)], stdout=out,
                               stderr=subprocess.STDOUT, check=True, timeout=60)


def wait_until(predicate, seconds, description):
    limit = min(DEADLINE, time.monotonic() + seconds)
    while time.monotonic() < limit:
        if predicate():
            return
        time.sleep(3)
    raise TimeoutError(description)


def idle_gpu():
    return not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()


def ports_idle(index, rank):
    """Match the frozen launchers' bind guards, including TCP TIME_WAIT."""
    if index == 1:
        endpoints = [("127.0.0.1", 18797 if rank == 0 else 29768)]
    else:
        endpoints = [("0.0.0.0", port) for port in (18796, 29767)] if rank == 0 else []
    for endpoint in endpoints:
        try:
            with socket.socket() as probe:
                probe.bind(endpoint)
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                return False
            raise
    return True


def command(argv, log, timeout=600):
    with log.open("ab") as out:
        subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def check_modern(receipt):
    previous = read(MODERN / "full-on-2/start-receipt.json")
    for key in ("image", "command", "library_sha256", "package_source_sha256", "boot_sha256"):
        if receipt[key] != previous[key]:
            raise ValueError("Frozen FP8-off runtime changed: " + key)
    if receipt["exit_code"] or receipt["environment"]["ZIPSERV_MIXED_FP8"] != "off":
        raise ValueError("FP8-off deployment did not pass")
    environment = dict(receipt["environment"])
    for key in ("ZIPSERV_VALIDATE_ONLY", "CUDA_VISIBLE_DEVICES"):
        environment.pop(key, None)
    if environment != previous["environment"]:
        raise ValueError("Frozen FP8-off environment changed")
    immutable_mounts = lambda row: sorted(
        (m for m in row["mounts"] if m["destination"] in ("/model", "/audit")),
        key=lambda m: m["destination"])
    if immutable_mounts(receipt) != immutable_mounts(previous):
        raise ValueError("Frozen checkpoint mounts changed")


def launch(index, rank):
    if index == 0:
        info = inspect(index, rank)
        if not info or not info["State"]["Running"]:
            raise RuntimeError("The explicitly staged CT-off server is missing")
        return
    if inspect(index, rank) is not None:
        raise RuntimeError("Preserve an existing run; do not reuse its container name")
    wait_until(idle_gpu, 120, "GPU is occupied by another workload")
    wait_until(lambda: ports_idle(index, rank), 180, "Previous private TCP port has not been released")
    log = ROOT / f"{PHASES[index]}-launch-r{rank}.log"
    if index == 1:
        command(["python3", str(CT / "tools/launch_ct_weight_control.py"), "--workdir", str(CT),
                 "--experiments-root", str(CT.parent),
                 "--mode", "ct", "--rank", str(rank), "--run", "quality20261025a"], log)
    else:
        argv = ["python3", str(ROOT / "tools/modern_deploy.py"), "--rank", str(rank),
                "--mode", "on", "--trial", str(MODERN_TRIAL), "--fp8-weights", "off"]
        command([*argv, "check"], log)
        check_modern(read(MODERN / f"full-on-{MODERN_TRIAL}/check-receipt.json"))
        command([*argv, "start"], log)
        check_modern(read(MODERN / f"full-on-{MODERN_TRIAL}/start-receipt.json"))


def receipt(index):
    return (MODERN / f"full-on-{MODERN_TRIAL}/start-receipt.json" if index == 2 else
            CT / ("quality20261025a-" + ("off" if index == 0 else "ct")) / "launch.json")


def peer_request(route, body=None):
    token = (ROOT / "peer-token").read_text().strip()
    req = urllib.request.Request(PEER_URL + route,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"X-Quality-Token": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120 if route == "/snapshot" else 10) as response:
        data = response.read()
        if route == "/snapshot":
            if hashlib.sha256(data).hexdigest() != response.headers["X-Archive-SHA256"]:
                raise ValueError("Snapshot transport hash differs")
            return data
        return json.loads(data)


def snapshot():
    members = {}
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if (not path.is_file() or path.is_symlink() or path.name == "peer-token"
                or any(p in ("peer-backups", "__pycache__") for p in rel.parts)
                or path.suffix in (".tmp", ".incomplete", ".tgz", ".pyc", ".lock")):
            continue
        data = path.read_bytes()
        if path.suffix == ".jsonl" and data and not data.endswith(b"\n"):
            data = data[:data.rfind(b"\n") + 1]
        members[rel.as_posix()] = data
    manifest = {key: hashlib.sha256(value).hexdigest() for key, value in members.items()}
    members["SNAPSHOT-MANIFEST.json"] = json.dumps(manifest, indent=2).encode()
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz", compresslevel=1) as archive:
        for key, data in members.items():
            item = tarfile.TarInfo(key)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    return out.getvalue()


def validate_peer(value):
    if (not isinstance(value, dict) or type(value.get("phase")) is not int
            or value["phase"] not in range(3) or value.get("status") not in (
                "started", "stopped", "failed", "backed_up", "finished")):
        raise ValueError("Invalid fixed peer phase/status")
    return value


def serve():
    token = (ROOT / "peer-token").read_text().strip()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if not hmac.compare_digest(self.headers.get("X-Quality-Token", ""), token):
                self.send_error(403)
                return
            if self.path == "/state":
                state = read(ROOT / "state.json")
                phase = state.get("phase", 0)
                p = ROOT / "runs" / PHASES[phase] / "summary.json"
                if p.exists():
                    state["progress"] = read(p)
                data = json.dumps(state).encode()
            elif self.path == "/snapshot":
                data = snapshot()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Archive-SHA256", hashlib.sha256(data).hexdigest())
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/peer-state" or not hmac.compare_digest(self.headers.get("X-Quality-Token", ""), token):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 16384:
                    raise ValueError("Invalid message size")
                value = validate_peer(json.loads(self.rfile.read(length)))
            except (ValueError, TypeError):
                self.send_error(400)
                return
            write(ROOT / "peer-state.json", value)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
    server = ThreadingHTTPServer(BIND, Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def verify_snapshot(data):
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        manifest = json.load(archive.extractfile("SNAPSHOT-MANIFEST.json"))
        actual = archive.getmembers()
        if (len(actual) != len(manifest) + 1
                or {m.name for m in actual} != set(manifest) | {"SNAPSHOT-MANIFEST.json"}
                or any(not m.isfile() or Path(m.name).is_absolute() or ".." in Path(m.name).parts
                       or Path(m.name).name == "peer-token" for m in actual)):
            raise ValueError("Unexpected snapshot member")
        for path, expected in manifest.items():
            if hashlib.sha256(archive.extractfile(path).read()).hexdigest() != expected:
                raise ValueError("Snapshot member mismatch: " + path)
    return manifest


def backup_peer(state):
    data = peer_request("/snapshot")
    manifest = verify_snapshot(data)
    out = ROOT / "peer-backups"
    out.mkdir(exist_ok=True)
    count = state.get("progress", {}).get("completed_questions", 0)
    path = out / f"phase{state['phase']}-{state['stage']}-{count}-{time.time_ns()}.tgz"
    path.write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "members": len(manifest),
            "phase": state["phase"], "stage": state["stage"], "completed_questions": count}


def follower():
    current = None
    last_backup = (-1, -1, "")
    last_contact = time.monotonic()
    record = None
    try:
        while time.monotonic() < DEADLINE:
            try:
                state = peer_request("/state")
                last_contact = time.monotonic()
            except OSError:
                if time.monotonic() - last_contact > 180:
                    raise TimeoutError("Leader status channel lost")
                time.sleep(3)
                continue
            index, stage = state["phase"], state["stage"]
            if stage in ("starting", "running") and current != index:
                if current is not None:
                    stop(current, 1)
                current = index
                launch(index, 1)
                write(ROOT / f"launch-{PHASES[index]}-r1.json", read(receipt(index)))
                peer_request("/peer-state", {"phase": index, "status": "started"})
                write(ROOT / "follower-state.json", {"phase": index, "status": "started"})
            count = state.get("progress", {}).get("completed_questions", 0)
            terminal = stage in ("stopping", "done", "failed")
            if (index != last_backup[0] or count >= last_backup[1] + 200
                    or terminal and stage != last_backup[2]):
                record = backup_peer(state)
                write(ROOT / "latest-peer-backup.json", record)
                last_backup = (index, count, stage)
            if stage in ("stopping", "done", "failed"):
                if current is not None:
                    stop(current, 1)
                if stage in ("done", "failed"):
                    if record is None:
                        raise RuntimeError("Final evidence backup missing")
                    peer_request("/peer-state", {"phase": index, "status": "finished", "backup": record})
                    write(ROOT / "follower-state.json", {"phase": index, "status": "finished", "backup": record})
                    return
                peer_request("/peer-state", {"phase": index, "status": "stopped"})
            time.sleep(3)
        raise TimeoutError("Quality follower deadline")
    except BaseException as error:
        write(ROOT / "follower-state.json", {"phase": current, "status": "failed", "error": repr(error)})
        try:
            peer_request("/peer-state", {"phase": current or 0, "status": "failed", "error": repr(error)})
        finally:
            if current is not None:
                stop(current, 1)
        raise


def modern_audit(stage):
    from bench_scale_cache_model import rpc
    mixed = rpc("http://127.0.0.1:18796", "zipserv_mixed_report", [True], 2)
    for row in mixed:
        if (row["status"] != "prepared" or not row["raw_owners_released"]
                or row["per_type"]["bf16"]["compressed_layers"] != 181
                or row["per_type"]["fp8"]["compressed_layers"] != 0):
            raise RuntimeError("Actual FP8-off worker representation differs")
    scales = rpc("http://127.0.0.1:18796", "zipserv_mxfp4_report", [True], 2)
    for row in scales:
        if row["mode"] != "on" or row["layers_seen"] != 42 or row["compressed_scale_groups"] != 84:
            raise RuntimeError("Actual FP4 scale coverage differs")
    write(ROOT / ("fp8-off-audit-" + stage + ".json"), {"mixed": mixed, "fp4_scales": scales})


def leader(start_phase=0):
    for previous in PHASES[:start_phase]:
        result = read(ROOT / "runs" / previous / "summary.json")
        if result["status"] != "complete" or result["completed_questions"] != 1319 or result["api_errors"]:
            raise ValueError("Cannot resume past an incomplete quality run: " + previous)
    index = start_phase
    write(ROOT / "state.json", {"phase": index, "stage": "starting", "started_epoch": time.time()})
    server = serve()
    try:
        for index, phase in enumerate(PHASES):
            if index < start_phase:
                continue
            write(ROOT / "state.json", {"phase": index, "stage": "starting", "updated_epoch": time.time()})
            launch(index, 0)
            def peer_started():
                p = ROOT / "peer-state.json"
                if not p.exists():
                    return False
                state = read(p)
                if state["status"] == "failed":
                    raise RuntimeError("Peer failed: " + state.get("error", "unknown"))
                return state["phase"] == index and state["status"] == "started"
            wait_until(peer_started, 900, "Peer did not start the matching rank")
            port = 18796 if index == 2 else 18797
            def healthy():
                info = inspect(index, 0)
                if not info or not info["State"]["Running"]:
                    raise RuntimeError("Private server exited before readiness")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                        return r.status == 200
                except OSError:
                    return False
            wait_until(healthy, 2400, "Private server readiness timeout")
            write(ROOT / f"launch-{phase}-r0.json", read(receipt(index)))
            write(ROOT / "state.json", {"phase": index, "stage": "running", "updated_epoch": time.time()})
            output = ROOT / "runs" / phase
            if index == 0:
                def completed():
                    summary = read(output / "summary.json")
                    if summary["status"] not in ("running", "complete"):
                        raise RuntimeError("CT-off client failed: " + summary.get("error", "unknown"))
                    return summary["status"] == "complete"
                wait_until(completed, 5400, "CT-off quality timeout")
            elif index == 1:
                command(["python3", str(ROOT / "tools/eval_exact_ct_gsm8k.py"), "run",
                         "--parent-protocol", str(ROOT / "parent-protocol.json"),
                         "--protocol", str(ROOT / "exact-ct-protocol.json"), "--server-receipt", str(receipt(index)),
                         "--output", str(output), "--mode", "ct"], ROOT / "ct-on-client.log", 5400)
            else:
                import eval_ct_gsm8k as common
                previous = read(MODERN / "full-on-1/gsm8k/summary.json")
                scorer = MODERN / "tools/eval_ct_gsm8k.py"
                if hashlib.sha256(scorer.read_bytes()).hexdigest() != previous["client_sha256"]:
                    raise ValueError("Do not change the original modern scoring client")
                modern_audit("before")
                command(["python3", str(scorer), "run", "--protocol", str(ROOT / "parent-protocol.json"),
                         "--output", str(output), "--arm", "fp8-off", "--server-receipt", str(receipt(index))],
                        ROOT / "fp8-off-client.log", 5400)
                modern_audit("after")
            summary = read(output / "summary.json")
            if summary["status"] != "complete" or summary["completed_questions"] != 1319 or summary["api_errors"]:
                raise RuntimeError("Quality run incomplete or has API errors: " + phase)
            print(json.dumps({"phase": phase, "summary": summary}), flush=True)
            write(ROOT / "state.json", {"phase": index, "stage": "stopping", "updated_epoch": time.time()})
            stop(index, 0)
            wait_until(lambda: (read(ROOT / "peer-state.json").get("phase") == index
                                and read(ROOT / "peer-state.json").get("status") == "stopped"),
                       180, "Peer did not release the private GPU worker")
        import eval_ct_gsm8k as common
        write(ROOT / "comparison-exact-ct.json", common.compare(ROOT / "runs/ct-off", ROOT / "runs/ct-on"))
        write(ROOT / "comparison-modern-baseline-vs-fp8-off.json",
              common.compare(MODERN.parent / "full-off-2/gsm8k", ROOT / "runs/fp8-off"))
        write(ROOT / "comparison-modern-full-vs-fp8-off.json",
              common.compare(MODERN / "full-on-1/gsm8k", ROOT / "runs/fp8-off"))
        write(ROOT / "completion.json", {"status": "complete", "private_servers_stopped": True,
                                         "completed_epoch": time.time()})
        write(ROOT / "state.json", {"phase": 2, "stage": "done", "completed_epoch": time.time()})
        wait_until(lambda: read(ROOT / "peer-state.json").get("status") == "finished", 180,
                   "Final Spark2 evidence backup was not acknowledged")
        write(ROOT / "final-backup-ack.json", read(ROOT / "peer-state.json"))
    except BaseException as error:
        write(ROOT / "state.json", {"phase": index, "stage": "failed", "error": repr(error), "epoch": time.time()})
        stop(index, 0)
        try:
            wait_until(lambda: (ROOT / "peer-state.json").exists()
                       and read(ROOT / "peer-state.json").get("status") == "finished", 180,
                       "Peer failure backup acknowledgement")
        except TimeoutError:
            pass
        raise
    finally:
        server.shutdown()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--ct-workdir", type=Path, required=True)
    p.add_argument("--modern-workdir", type=Path, required=True)
    p.add_argument("--peer-url", required=True, help="Authenticated coordinator origin on rank 0")
    p.add_argument("--bind-address", default="127.0.0.1",
                   help="Rank-0 listen address; explicitly select the private interface for a remote peer")
    p.add_argument("--bind-port", type=int, default=29871)
    p.add_argument("--rank", type=int, choices=(0, 1), required=True)
    p.add_argument("--start-phase", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--source-manifest", choices=("controller-sources.json", "controller-sources-v2.json",
                                                "controller-sources-v3.json"),
                   default="controller-sources.json")
    args = p.parse_args(argv)
    peer = urlsplit(args.peer_url)
    try:
        port = peer.port
    except ValueError:
        p.error("--peer-url has an invalid port")
    if (peer.scheme not in ("http", "https") or not peer.hostname
            or peer.username is not None or peer.password is not None
            or peer.path not in ("", "/") or peer.query or peer.fragment or port == 0):
        p.error("--peer-url must be an HTTP(S) origin without credentials, path or query")
    if not args.bind_address or not 1 <= args.bind_port <= 65535:
        p.error("Require a listen address and a port between 1 and 65535")
    args.peer_url = args.peer_url.rstrip("/")
    return args


def main():
    global ROOT, CT, MODERN, PEER_URL, BIND, DEADLINE
    args = parse_args()
    ROOT, CT, MODERN = (path.resolve() for path in
                        (args.workdir, args.ct_workdir, args.modern_workdir))
    PEER_URL = args.peer_url
    BIND = (args.bind_address, args.bind_port)
    DEADLINE = time.monotonic() + 6 * 3600
    if not CT.is_dir() or not MODERN.is_dir():
        raise RuntimeError("Require the staged CT and modern experiment directories")
    if not ROOT.is_dir() or not (ROOT / "peer-token").is_file():
        raise RuntimeError("Require the staged private experiment and peer token")
    for relative, expected in read(ROOT / args.source_manifest).items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Staged controller/client/protocol source changed: " + relative)
    with (ROOT / f"coordinator-r{args.rank}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        leader(args.start_phase) if args.rank == 0 else follower()


if __name__ == "__main__":
    main()
