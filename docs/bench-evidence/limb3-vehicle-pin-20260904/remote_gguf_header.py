#!/usr/bin/env python3
# Read a REMOTE GGUF's header by HTTP RANGE request, without fetching the file.
#
# The limb-3 vehicle predicate is evaluated against an artifact's own bytes
# (#2864), and this repository's practice is to read a checkpoint manifest by
# range request rather than by download. A 16.8 GB candidate therefore has to
# answer "what architecture do you declare, and what ggml type ids do you
# store" BEFORE anyone spends the bandwidth on it.
#
# It reuses the committed reader in ../limb3-vehicle-search-20260904 rather
# than re-implementing the parse, so the two searches cannot disagree about
# what a header says. The prefix is grown until the reader stops running out
# of bytes, because a GGUF's kv block carries the whole tokenizer and its
# length is not known in advance.
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
READER = os.path.join(
    HERE, "..", "limb3-vehicle-search-20260904", "gguf_header.py")


def fetch(url, nbytes, out):
    subprocess.run(
        ["curl", "-sSL", "--fail", "-r", "0-%d" % (nbytes - 1), "-o", out, url],
        check=True)


def main(url):
    tmp = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "gguf-prefix-%d.bin" % os.getpid())
    try:
        for mb in (4, 16, 64, 192):
            fetch(url, mb * 1024 * 1024, tmp)
            p = subprocess.run(
                [sys.executable, READER, tmp],
                capture_output=True, text=True)
            if p.returncode == 0:
                print("URL                %s" % url)
                print("PREFIX_MIB         %d" % mb)
                sys.stdout.write(p.stdout.replace(tmp, "<range-prefix>"))
                return 0
            last = p.stderr
        sys.stderr.write(last)
        return 1
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
