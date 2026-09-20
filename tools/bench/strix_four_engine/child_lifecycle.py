"""Exclusive Linux controller ownership of adapter descendants (#3108).

Linux man-pages 6.18: PR_SET_CHILD_SUBREAPER, waitpid(-pgid, WNOHANG).
The sigaction binding is restricted to the measured glibc-2.39 x86-64 ABI:
sysdeps/unix/sysv/linux/bits/sigaction.h and bits/types/__sigset_t.h.
No inference engine or private multiprocessing API is involved.
"""
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import platform
import signal
import sys
import threading
import time

_lock = threading.Lock()
_controller = None


class Sigaction(ctypes.Structure):
    _fields_ = [("handler", ctypes.c_void_p), ("mask", ctypes.c_ulong * 16),
                ("flags", ctypes.c_int), ("restorer", ctypes.c_void_p)]


class Native:
    def __init__(self):
        if sys.platform != "linux" or platform.machine() != "x86_64":
            raise RuntimeError("child lifecycle requires Linux x86-64 glibc 2.39")
        self.lib = ctypes.CDLL(None, use_errno=True)
        version = self.lib.gnu_get_libc_version
        version.argtypes, version.restype = [], ctypes.c_char_p
        if version() != b"2.39" or ctypes.sizeof(Sigaction) != 152:
            raise RuntimeError("unsupported child lifecycle sigaction ABI (requires glibc 2.39)")
        self.lib.prctl.argtypes, self.lib.prctl.restype = [ctypes.c_int], ctypes.c_int
        self.lib.sigaction.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(Sigaction)]
        self.lib.sigaction.restype = ctypes.c_int

    def call(self, option, argument):
        result = self.lib.prctl(option, ctypes.c_ulong(argument), ctypes.c_ulong(0),
                               ctypes.c_ulong(0), ctypes.c_ulong(0))
        if result != 0:
            raise OSError(ctypes.get_errno(), "child lifecycle prctl failed")

    def state(self):
        value = ctypes.c_int(-1)
        self.call(37, ctypes.addressof(value))  # PR_GET_CHILD_SUBREAPER writes int.
        if value.value not in (0, 1):
            raise RuntimeError("invalid subreaper state")
        return value.value

    def set_state(self, value):
        self.call(36, value)  # PR_SET_CHILD_SUBREAPER, variadic unsigned-long args.

    def signal_state(self):
        action = Sigaction()
        if self.lib.sigaction(signal.SIGCHLD, None, ctypes.byref(action)) != 0:
            raise OSError(ctypes.get_errno(), "SIGCHLD sigaction query failed")
        if (signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL or action.handler or
                action.flags & 2):  # glibc SA_NOCLDWAIT: no waitable statuses.
            raise RuntimeError("competing SIGCHLD handler or automatic child reaping")
        # Linux x86-64 has 64 signal bits. libc_sigaction.c passes _NSIG/8
        # to rt_sigaction; the remaining glibc sigset_t storage is not kernel data.
        return (action.handler, action.mask[0], action.flags, action.restorer)


def exclusive_thread():
    if threading.current_thread() is not threading.main_thread() or len(list(Path("/proc/self/task").iterdir())) != 1:
        raise RuntimeError("child lifecycle requires the exclusive main OS thread")


def no_children():
    if Path(f"/proc/self/task/{os.getpid()}/children").read_text().strip():
        raise RuntimeError("controller has pre-existing children or a competing reaper")


class Controller:
    def __init__(self):
        self.native = Native()
        exclusive_thread()
        no_children()
        self.signal = self.native.signal_state()
        if self.native.state() != 0:
            raise RuntimeError("controller already has an unowned subreaper arrangement")
        self.pid, self.active, self.poisoned = os.getpid(), None, False


@contextmanager
def controller_lifecycle():
    """Declare exclusive process lifecycle ownership for a sequential controller.

    Imported drivers must explicitly enter this context before creating Adapter.
    They must not install reapers, create threads, or run concurrent lifecycles.
    The same checked context is used by the production qualification CLI.
    """
    global _controller
    if not _lock.acquire(blocking=False):
        raise RuntimeError("nested or concurrent controller lifecycle")
    try:
        owner = Controller()
        _controller = owner
        original_error = None
        try:
            yield
        except BaseException as error:
            original_error = error
            raise
        finally:
            if owner.active is not None:
                session = owner.active
                session.cleanup()
                session.release()
                if session.abandon_transport is not None:
                    session.abandon_transport()
                if original_error is not None:
                    session.annotate(original_error)
                else:
                    error = RuntimeError("controller exited with an unclosed adapter")
                    session.annotate(error)
                    raise error
    finally:
        _controller = None
        _lock.release()


class ChildLifecycle:
    def __init__(self):
        owner = _controller
        if owner is None or owner.pid != os.getpid():
            raise RuntimeError("Adapter requires an explicit controller_lifecycle context")
        exclusive_thread()
        if owner.active is not None or owner.poisoned:
            raise RuntimeError("concurrent adapter or unresolved child lifecycle failure")
        no_children()
        if owner.native.signal_state() != owner.signal:
            raise RuntimeError("controller SIGCHLD arrangement changed")
        self.owner, self.process, self.released = owner, None, False
        self.abandon_transport = None
        self.evidence = dict(adopted=[], cleanup_errors=[], restored=False)
        self.prior = owner.native.state()
        if self.prior != 0:
            raise RuntimeError("unowned subreaper state before adapter launch")
        owner.active = self
        try:
            owner.native.set_state(1)
            if owner.native.state() != 1:
                raise RuntimeError("subreaper acquisition verification failed")
        except BaseException as error:
            self.release()
            self.annotate(error)
            raise

    def annotate(self, error):
        error.lifecycle_evidence = self.evidence
        if self.evidence["cleanup_errors"]:
            error.add_note("child lifecycle: " + repr(self.evidence["cleanup_errors"]))

    def record_error(self, error):
        if len(self.evidence["cleanup_errors"]) < 16:
            self.evidence["cleanup_errors"].append(str(error)[:512])
        self.owner.poisoned = True

    def check_remaining_children(self):
        """Refuse unresolved ownership, without inferring historical ancestry."""
        def read(path):
            with Path(path).open() as stream:
                value = stream.read(65537)
            if len(value) > 65536:
                raise RuntimeError("controller child inventory bound exhausted")
            return value

        def children():
            values = read(f"/proc/self/task/{os.getpid()}/children").split()
            if len(values) > 4096 or any(not value.isdecimal() or int(value) <= 0 for value in values):
                raise RuntimeError("invalid controller child inventory")
            return sorted(int(value) for value in values)

        def identity(pid):
            raw = read(f"/proc/{pid}/stat")
            prefix, fields = raw.rsplit(")", 1)
            fields = fields.split()
            if int(prefix.split(" (", 1)[0]) != pid or int(fields[1]) != os.getpid():
                raise RuntimeError("controller child inventory identity drift")
            return dict(pid=pid, pgid=int(fields[2]), start_time=int(fields[19]))

        try:
            pids = children()
            records = [identity(pid) for pid in pids]
            if children() != pids or [identity(pid) for pid in pids] != records:
                raise RuntimeError("controller child inventory identity drift")
            if records:
                self.evidence["unresolved_children"] = records
                raise RuntimeError("unresolved controller children after owned-group drain")
        except BaseException as error:
            self.record_error(error)
            raise

    def drain(self, deadline, *, cleanup=False):
        if self.process is None or self.process.returncode is None:
            raise RuntimeError("direct parent must be waited by Popen before adopted-child drain")
        if not cleanup:
            exclusive_thread()
            if self.owner.native.signal_state() != self.owner.signal:
                raise RuntimeError("controller SIGCHLD arrangement changed")
        returned = 0
        while True:
            if time.monotonic() >= deadline or returned >= 4096:
                raise RuntimeError("adopted-child drain bound exhausted")
            try:
                pid, status = os.waitpid(-self.process.pid, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                if not cleanup:
                    raise RuntimeError("live adopted child survived shutdown")
                time.sleep(min(.01, max(0, deadline - time.monotonic())))
                continue
            returned += 1
            if len(self.evidence["adopted"]) < 8192:
                self.evidence["adopted"].append(dict(pid=pid, status=status, cleanup=cleanup))
            if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
                error = RuntimeError(f"abnormal adopted child status: pid={pid} status={status}")
                if not cleanup:
                    raise error
                self.record_error(error)

    def cleanup(self):
        """Error cleanup never changes the original acceptance result."""
        if self.process is None:
            return
        deadline = time.monotonic() + 5
        try:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except BaseException as error:
                self.record_error(error)
            self.process.wait(timeout=max(0, deadline - time.monotonic()))
            while True:
                self.drain(deadline, cleanup=True)
                try:
                    os.killpg(self.process.pid, 0)
                except ProcessLookupError:
                    self.check_remaining_children()
                    return
                if time.monotonic() >= deadline:
                    raise RuntimeError("owned-group cleanup deadline exhausted")
                time.sleep(min(.01, max(0, deadline - time.monotonic())))
        except BaseException as error:
            self.record_error(error)

    def release(self):
        if self.released:
            return
        try:
            self.owner.native.set_state(self.prior)
            if self.owner.native.state() != self.prior:
                raise RuntimeError("subreaper restoration verification failed")
            self.evidence["restored"] = True
        except BaseException as error:
            self.record_error(error)
        finally:
            self.released = True
            self.owner.active = None
