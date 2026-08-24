#!/usr/bin/env python3
"""GATE-CI-SITE-HUGO-LANE (#1754) -- the docs-site lane provisions its renderer.

`tests/scripts/test_check_site.py` renders the whole Hugo site and holds every
link in the rendered benchmark index to an emitted page. That assertion is only
worth its job minutes on a runner that HAS Hugo, and the failure this file exists
to prevent has two faces that a job name cannot tell apart:

  * No binary and no guard -- `subprocess.run` raises `FileNotFoundError`, the
    suite ERRORs, and `agent-record` is red on `main` and on every branch cut
    from it (#1722, #1754, #1764).
  * No binary and a guard -- the case skips, the suite reports `OK (skipped=1)`,
    and the job goes GREEN having rendered nothing. A skip wearing a pass is the
    worse of the two, because the red at least gets looked at.

Only installing the binary distinguishes them, so the invariants below are about
the LANE, not about the suite. They are derived from the workflow rather than
written as a list of job names: whichever job runs the site suite is the job that
must provision Hugo, so a step that moves to another job stays covered.

Non-vacuity is asserted first and separately. A resolver that finds no job passes
every assertion after it, which is the mute switch this file would otherwise be.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github/workflows/ci.yml"
PAGES = ROOT / ".github/workflows/gh-pages.yml"

SITE_SUITE = "tests/scripts/test_check_site.py"
HUGO_ACTION = "peaceiris/actions-hugo"

# `${{ env.NAME }}`, with the whitespace GitHub tolerates inside the braces.
_ENV_REF = re.compile(r"^\$\{\{\s*env\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}$")


def _load(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve(value: object, *scopes: dict) -> object:
    """Resolve a `${{ env.NAME }}` reference against the given env scopes.

    The pin is spelled through a workflow- or job-level `env` in both files, so
    comparing the two `with:` values literally would compare two identical
    template strings and prove nothing about the versions behind them.
    """

    if not isinstance(value, str):
        return value
    match = _ENV_REF.match(value.strip())
    if match is None:
        return value
    name = match.group(1)
    for scope in scopes:
        if name in scope:
            return scope[name]
    raise AssertionError(f"{value} names an env key no scope defines")


def _steps(job: dict) -> list[dict]:
    return [step for step in (job.get("steps") or []) if isinstance(step, dict)]


def _runs_site_suite(step: dict) -> bool:
    return SITE_SUITE in (step.get("run") or "")


def _is_hugo_setup(step: dict) -> bool:
    return (step.get("uses") or "").startswith(HUGO_ACTION + "@")


def _jobs_running_the_site_suite(workflow: dict) -> list[tuple[str, dict]]:
    return [
        (name, job)
        for name, job in (workflow.get("jobs") or {}).items()
        if any(_runs_site_suite(step) for step in _steps(job))
    ]


class SiteLaneProvisioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ci = _load(CI)
        cls.pages = _load(PAGES)

    # ---- non-vacuity ----------------------------------------------------

    def test_the_resolver_finds_the_lane_it_is_about(self) -> None:
        """Every assertion below is vacuous if this one does not hold."""
        lanes = _jobs_running_the_site_suite(self.ci)
        self.assertEqual(
            [name for name, _ in lanes],
            ["agent-record"],
            "the docs-site suite moved, or the resolver stopped seeing it",
        )

    def test_the_published_site_still_installs_hugo_through_the_action(self) -> None:
        """The pin this file compares against has to exist to be compared."""
        setups = [
            step for step in _steps(self.pages["jobs"]["build"]) if _is_hugo_setup(step)
        ]
        self.assertEqual(len(setups), 1, f"gh-pages.yml hugo setup steps: {setups}")

    # ---- the invariant --------------------------------------------------

    def test_the_lane_installs_hugo_before_it_renders(self) -> None:
        for name, job in _jobs_running_the_site_suite(self.ci):
            steps = _steps(job)
            setup = [i for i, step in enumerate(steps) if _is_hugo_setup(step)]
            self.assertEqual(
                len(setup), 1, f"{name}: hugo setup steps at indices {setup}"
            )
            renders = [i for i, step in enumerate(steps) if _runs_site_suite(step)]
            self.assertTrue(renders, f"{name}: resolver disagrees with itself")
            self.assertLess(
                setup[0],
                min(renders),
                f"{name}: hugo is installed after the suite that needs it",
            )

    def test_the_lane_asks_for_the_extended_build(self) -> None:
        """`gh-pages` renders `extended`. A lane on plain Hugo renders a
        different site from the published one, so its verdict is about a site
        nobody visits."""
        seen = 0
        for name, job in _jobs_running_the_site_suite(self.ci):
            for step in _steps(job):
                if _is_hugo_setup(step):
                    seen += 1
                    self.assertIs(
                        (step.get("with") or {}).get("extended"),
                        True,
                        f"{name}: hugo setup is not the extended build",
                    )
        self.assertTrue(seen, "no hugo setup step to judge; this case was vacuous")

    def test_the_lane_and_the_published_site_pin_the_same_hugo(self) -> None:
        published = None
        for step in _steps(self.pages["jobs"]["build"]):
            if _is_hugo_setup(step):
                published = _resolve(
                    (step.get("with") or {}).get("hugo-version"),
                    self.pages.get("env") or {},
                )
        self.assertIsInstance(published, str, "gh-pages.yml pins no hugo version")
        self.assertRegex(
            str(published),
            r"^\d+\.\d+\.\d+$",
            "a floating hugo pin renders a different site every run",
        )

        seen = 0
        for name, job in _jobs_running_the_site_suite(self.ci):
            for step in _steps(job):
                if not _is_hugo_setup(step):
                    continue
                seen += 1
                lane = _resolve(
                    (step.get("with") or {}).get("hugo-version"),
                    job.get("env") or {},
                    self.ci.get("env") or {},
                )
                self.assertEqual(
                    lane,
                    published,
                    f"{name} renders on Hugo {lane}; the site publishes on "
                    f"{published}. website/hugo.toml uses `excludeFiles`, "
                    "deprecated from 0.153, so the two pins moving apart is a "
                    "difference in what gets published, not only in a version.",
                )
        self.assertTrue(seen, "no hugo setup step to compare; this case was vacuous")

    def test_the_lane_proves_the_binary_resolved_before_running_the_suite(self) -> None:
        """A setup step that stopped producing a binary must fail LEGIBLY.

        Without this line the suite reports the absence itself, as either a red
        it did not cause or a skip nobody reads.
        """
        for name, job in _jobs_running_the_site_suite(self.ci):
            for step in _steps(job):
                if not _runs_site_suite(step):
                    continue
                body = step["run"].splitlines()
                probe = [i for i, line in enumerate(body) if line.strip() == "hugo version"]
                suite = [i for i, line in enumerate(body) if SITE_SUITE in line]
                self.assertTrue(
                    probe, f"{name}: the step runs the site suite without probing hugo"
                )
                self.assertLess(min(probe), min(suite), f"{name}: probe runs too late")


if __name__ == "__main__":
    unittest.main()
