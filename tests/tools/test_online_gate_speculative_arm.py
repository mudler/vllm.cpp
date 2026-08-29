"""The serving driver's speculative arm, read off the script on the CPU.

`scripts/dgx-online-serving.sh` owns the `1 2 4 8 16 32` concurrency grid and
interleaves its arms, which makes it the committed instrument for a c=8
comparison (#2152). Until now it contained ZERO references to `speculative`,
`dflash` or `draft`, so the DFlash2 workload could not run on it at all and was
measured beside it instead.

These assertions are read off the script's text, the same way
`test_online_gate_server_binary.py` binds the shell harness to the CMake output
name. That is what lets a bash change be gated with no GPU, no corpus and no
lease -- none of which are available to CI.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DRIVER = ROOT / "scripts" / "dgx-online-serving.sh"


class DriverSyntaxTest(unittest.TestCase):
    def test_the_driver_parses(self) -> None:
        # `bash -n` before every run is rule 3 of this script's own header: bash
        # re-reads from a byte offset, so a broken edit runs a spliced program.
        proc = subprocess.run(["bash", "-n", str(DRIVER)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


class SpeculativeArmTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = DRIVER.read_text(encoding="utf-8")

    def test_both_flags_are_accepted(self) -> None:
        self.assertIn("--draft) draft=", self.text)
        self.assertIn("--speculative-config) speculative_config=", self.text)

    def test_a_draft_without_a_config_is_REFUSED(self) -> None:
        # Half a configuration would launch one arm speculating and the other
        # not, which measures the feature rather than the implementation.
        self.assertRegex(
            self.text,
            r"-n \$\{draft\} && -z \$\{speculative_config\}[\s\S]{0,200}exit 2",
        )

    def test_a_config_without_a_draft_is_REFUSED(self) -> None:
        self.assertRegex(
            self.text,
            r"-n \$\{speculative_config\} && -z \$\{draft\}[\s\S]{0,200}exit 2",
        )

    def test_a_missing_draft_path_is_REFUSED(self) -> None:
        self.assertRegex(self.text, r"! -e \$\{draft\}[\s\S]{0,120}exit 2")

    def test_BOTH_arms_receive_the_config(self) -> None:
        # The rule `scripts/dflash2-speed-gate.sh` already enforces on its own
        # row: "the oracle arm drafts and a ratio against a plain decode
        # measures the feature, not this row". Ours appending it is not enough.
        appends = re.findall(r"server_cmd\+=\(--speculative-config", self.text)
        self.assertGreaterEqual(
            len(appends), 2,
            "both the ours arm and the oracle arm must append the speculative "
            "config; found %d append site(s)" % len(appends),
        )
        self.assertIn('${engine} != ours', self.text,
                      "the oracle-side append must be guarded to the non-ours arms")

    def test_the_non_speculative_command_is_UNCHANGED(self) -> None:
        # Every append is guarded on a non-empty config, so a run that passes
        # neither flag emits byte-identical commands to before this change --
        # which is what keeps the existing binding numbers comparable.
        for m in re.finditer(r"server_cmd\+=\(--speculative-config", self.text):
            window = self.text[max(0, m.start() - 400):m.start()]
            self.assertRegex(
                window, r"-n \$\{speculative_config\}",
                "an unguarded append would change the non-speculative command",
            )

    def test_the_flags_reach_the_recorded_command_file(self) -> None:
        # Provenance: the driver writes every argv element to
        # r<N>-server-command.txt, so appending to server_cmd BEFORE that write
        # is what makes a speculative run reproducible.
        write_at = self.text.index('printf \'%q \' "${server_cmd[@]}" >"${command_file}"')
        first_append = self.text.index("server_cmd+=(--speculative-config")
        self.assertLess(first_append, write_at,
                        "the config must be appended before the command file is written")


if __name__ == "__main__":
    unittest.main()
