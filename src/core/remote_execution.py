"""Bounded SSH retries for reads and verified idempotent operations.

An uncertain process launch is reconciled by the caller, never blindly retried.
Provider dispatches do not use this transport.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import subprocess
import tarfile
import time


@dataclass
class RemoteUnavailable(RuntimeError):
    attempts: int
    operation: str

    def __str__(self):
        return f"Remote transport unavailable after {self.attempts} attempts ({self.operation})"


class RemoteCommandError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(f"Remote command failed with exit code {code}")


class Remote:
    def __init__(self, access, on_retry=None, run=subprocess.run, sleeper=time.sleep,
                 attempts=6, delays=(1, 2, 4, 8, 15)):
        self.access = access
        self.on_retry = on_retry or (lambda event: None)
        self.run, self.sleeper = run, sleeper
        self.attempts, self.delays = attempts, delays

    def _call(self, command, *, timeout, payload=None, retry_safe=False, operation="read"):
        attempts = self.attempts if retry_safe else 1
        for index in range(attempts):
            argv = list(self.access.connection())
            argv[1:1] = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                         "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2"]
            try:
                result = self.run(argv + [command], input=payload, capture_output=True,
                                  timeout=timeout)
                if result.returncode == 0:
                    return result.stdout.decode("utf-8")
                if result.returncode != 255:
                    raise RemoteCommandError(result.returncode)
                category = "ssh_transport_exit255"
            except subprocess.TimeoutExpired:
                category = "transport_timeout"
            if index + 1 == attempts:
                raise RemoteUnavailable(attempts, operation)
            delay = self.delays[min(index, len(self.delays) - 1)]
            self.on_retry({"operation": operation, "attempt": index + 1,
                           "maximum_attempts": attempts, "category": category, "delay_seconds": delay})
            self.sleeper(delay)

    def read(self, command, timeout=40):
        return self._call(command, timeout=timeout, retry_safe=True, operation="read")

    def idempotent(self, command, timeout=45):
        """Caller guarantees re-execution has identical effect."""
        return self._call(command, timeout=timeout, retry_safe=True, operation="idempotent_write")

    def launch(self, command, timeout=45):
        """At most one dispatch; RemoteUnavailable requires status reconciliation."""
        return self._call(command, timeout=timeout, retry_safe=False, operation="launch")

    def upload(self, paths, remote_root, names):
        import shlex
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for path, name in zip(paths, names):
                archive.add(path, arcname=name, recursive=True)
        self._call(f"tar -xf - -C {shlex.quote(remote_root)}", timeout=55,
                   payload=buffer.getvalue(), retry_safe=True, operation="immutable_artifact_upload")
