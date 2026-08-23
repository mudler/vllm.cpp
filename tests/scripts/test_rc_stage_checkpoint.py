#!/usr/bin/env python3
"""Hermetic tests for scripts/rc-stage-checkpoint.sh (#1807).

Each case builds a fake NAS checkpoint under a temp dir and stages it into a
second temp dir. The script's own stdout lines are the observable contract
(`copied N, kept M`), together with the bytes on disk.
"""
import hashlib
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rc-stage-checkpoint.sh"


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(*args):
    return subprocess.run(
        ["bash", str(SCRIPT), *[str(a) for a in args]],
        capture_output=True, text=True)


class RcStageCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="rcstage-"))
        self.src = self.tmp / "nas" / "ckpt"
        self.dst = self.tmp / "local" / "ckpt"
        (self.src / "sub").mkdir(parents=True)
        (self.src / "config.json").write_bytes(b'{"a":1}\n')
        (self.src / "model-00001-of-00002.safetensors").write_bytes(os.urandom(65536))
        (self.src / "model-00002-of-00002.safetensors").write_bytes(os.urandom(70000))
        (self.src / "sub" / "tokenizer.json").write_bytes(b"tok")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def files(self):
        return sorted(p.relative_to(self.src).as_posix()
                      for p in self.src.rglob("*") if p.is_file()
                      and p.name != "SHA256SUMS")

    def test_no_manifest_refuses_loudly(self):
        r = run(self.src, self.dst)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("REFUSED", r.stderr)
        self.assertIn("--make-manifest", r.stderr)
        self.assertFalse((self.dst / ".staged-ok").exists())

    def test_make_manifest_lists_every_file_with_sha_and_size(self):
        r = run("--make-manifest", self.src)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = (self.src / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(len(lines), 4)
        got = {}
        for line in lines:
            sha, size, rel = line.split(" ", 2)
            got[rel] = (sha, int(size))
        self.assertEqual(sorted(got), self.files())
        for rel, (sha, size) in got.items():
            p = self.src / rel
            self.assertEqual(sha, sha256(p), rel)
            self.assertEqual(size, p.stat().st_size, rel)

    def test_first_run_copies_second_run_reads_nothing(self):
        self.assertEqual(run("--make-manifest", self.src).returncode, 0)
        r1 = run(self.src, self.dst)
        self.assertEqual(r1.returncode, 0, r1.stderr)
        self.assertIn("STAGED", r1.stdout)
        self.assertIn("copied 4, kept 0", r1.stdout)
        for rel in self.files():
            self.assertEqual(sha256(self.dst / rel), sha256(self.src / rel), rel)
        marker = (self.dst / ".staged-ok").read_text().strip()
        self.assertEqual(marker, sha256(self.src / "SHA256SUMS"))
        # Second run: marker matches, sizes match, payload never opened. Prove
        # "never opened" by making the SOURCE payload unreadable: a run that
        # touched it would fail; the fast path must not notice.
        for rel in self.files():
            os.chmod(self.src / rel, 0)
        try:
            r2 = run(self.src, self.dst)
        finally:
            for rel in self.files():
                os.chmod(self.src / rel, 0o644)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertIn("ALREADY STAGED", r2.stdout)
        self.assertIn("nothing read", r2.stdout)

    def test_corrupted_local_file_is_recopied_and_others_kept(self):
        self.assertEqual(run("--make-manifest", self.src).returncode, 0)
        self.assertEqual(run(self.src, self.dst).returncode, 0)
        victim = self.dst / "model-00002-of-00002.safetensors"
        data = bytearray(victim.read_bytes())
        data[100] ^= 0xFF  # same size, different bytes
        victim.write_bytes(bytes(data))
        # Same size, so the marker fast path would accept it: the marker is the
        # contract that the bytes were verified once. A job that suspects a bad
        # local copy removes the marker, and the hash pass must then catch the
        # flipped byte.
        (self.dst / ".staged-ok").unlink()
        r = run(self.src, self.dst)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("copied 1, kept 3", r.stdout)
        self.assertEqual(sha256(victim), sha256(self.src / victim.name))

    def test_killed_copy_leftover_part_is_restaged(self):
        self.assertEqual(run("--make-manifest", self.src).returncode, 0)
        self.assertEqual(run(self.src, self.dst).returncode, 0)
        # A killed copy leaves a .part and no final file. The marker is still
        # present; the fast path must notice the absent file.
        (self.dst / "model-00001-of-00002.safetensors").unlink()
        (self.dst / "model-00001-of-00002.safetensors.part").write_bytes(b"xx")
        r = run(self.src, self.dst)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("copied 1, kept 3", r.stdout)
        self.assertFalse((self.dst / "model-00001-of-00002.safetensors.part").exists())
        for rel in self.files():
            self.assertEqual(sha256(self.dst / rel), sha256(self.src / rel), rel)
        self.assertTrue((self.dst / ".staged-ok").exists())

    def test_truncated_local_file_alone_defeats_the_fast_path(self):
        self.assertEqual(run("--make-manifest", self.src).returncode, 0)
        self.assertEqual(run(self.src, self.dst).returncode, 0)
        # ONLY a truncation, marker present, every file present: the fast
        # path's per-file SIZE check is the one thing standing between a
        # truncated shard and an "ALREADY STAGED" that reads nothing.
        t = self.dst / "sub" / "tokenizer.json"
        t.write_bytes(t.read_bytes()[:1])
        r = run(self.src, self.dst)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("ALREADY STAGED", r.stdout)
        self.assertIn("copied 1, kept 3", r.stdout)
        self.assertEqual(sha256(t), sha256(self.src / "sub" / "tokenizer.json"))
        self.assertTrue((self.dst / ".staged-ok").exists())

    def test_source_mutated_after_manifest_fails_verification_and_writes_no_marker(self):
        self.assertEqual(run("--make-manifest", self.src).returncode, 0)
        (self.src / "config.json").write_bytes(b'{"a":2}\n')  # same size, new bytes
        r = run(self.src, self.dst)
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
        self.assertIn("VERIFY FAILED", r.stderr)
        self.assertFalse((self.dst / ".staged-ok").exists())
        self.assertFalse((self.dst / "config.json").exists())

    def test_source_file_missing_is_exit_5(self):
        self.assertEqual(run("--make-manifest", self.src).returncode, 0)
        (self.src / "sub" / "tokenizer.json").unlink()
        r = run(self.src, self.dst)
        self.assertEqual(r.returncode, 5, r.stderr)
        self.assertIn("ABSENT", r.stderr)

    def test_malformed_manifest_is_exit_3(self):
        (self.src / "SHA256SUMS").write_text("nothash 12 config.json\n")
        r = run(self.src, self.dst)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("malformed", r.stderr)

    def test_unlisted_local_file_is_left_alone_and_reported(self):
        self.assertEqual(run("--make-manifest", self.src).returncode, 0)
        self.dst.mkdir(parents=True)
        stray = self.dst / "notes.txt"
        stray.write_bytes(b"mine")
        r = run(self.src, self.dst)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(stray.exists())
        self.assertIn("unlisted local files left alone", r.stdout)
        self.assertIn("notes.txt", r.stdout)

    def test_stray_names_with_space_and_glob_are_reported_verbatim(self):
        # Extends the stray-file report: a name with a space must stay one
        # line, and a name of `*` must not glob-expand against the caller's
        # cwd. Both files are left alone.
        self.assertEqual(run("--make-manifest", self.src).returncode, 0)
        self.dst.mkdir(parents=True)
        spaced = self.dst / "my notes.txt"
        spaced.write_bytes(b"mine")
        starry = self.dst / "*"
        starry.write_bytes(b"mine too")
        r = run(self.src, self.dst)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(spaced.exists())
        self.assertTrue(starry.exists())
        self.assertIn("unlisted local files left alone", r.stdout)
        reported = [l[2:] for l in r.stdout.splitlines() if l.startswith("  ")]
        self.assertEqual(sorted(reported), ["*", "my notes.txt"])
        # A glob-expanded `*` would have injected the caller's cwd listing.
        for name in os.listdir("."):
            self.assertNotIn(f"  {name}\n", r.stdout)

    def test_manifest_relpath_with_dotdot_is_refused_and_writes_nothing_outside_dst(self):
        # A hand-edited manifest naming `../escape` must be refused as
        # malformed (exit 3), not staged into DST's parent.
        payload = b"evil"
        (self.src.parent / "escape").write_bytes(payload)
        line = f"{hashlib.sha256(payload).hexdigest()} {len(payload)} ../escape\n"
        (self.src / "SHA256SUMS").write_text(line)
        r = run(self.src, self.dst)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn("../escape", r.stderr)
        self.assertFalse((self.dst.parent / "escape").exists())
        self.assertFalse((self.dst.parent / "escape.part").exists())
        self.assertFalse((self.dst / ".staged-ok").exists())

    def test_usage_errors(self):
        self.assertEqual(run().returncode, 2)
        self.assertEqual(run(self.src).returncode, 2)
        self.assertEqual(run("--make-manifest").returncode, 2)


if __name__ == "__main__":
    unittest.main()
