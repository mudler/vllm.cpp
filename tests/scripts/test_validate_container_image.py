#!/usr/bin/env python3
"""Mutation suite for the hub-reach audit in scripts/validate-container-image.py.

ENG-HF-MODEL-DOWNLOAD W5 (#1280). The audit exists because every CHEAPER check
passes on an image that cannot fetch anything. A `readelf` on the binary finds
libssl whether or not `ca-certificates` survived the layer. A symbol check finds
`SSL_connect` whether or not `VLLM_CPP_HF_DOWNLOAD` resolved OFF, because the
server links OpenSSL for other reasons on some lanes. Only an answer FROM THE
HUB separates the states, and this suite pins that the classifier actually
separates them rather than accepting anything non-zero.

The classifier is pure string work on purpose, so every branch is reachable with
no docker daemon and no network.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts/validate-container-image.py"


def load_module():
    spec = importlib.util.spec_from_file_location("validate_container_image", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


class HubReachClassifierTests(unittest.TestCase):
    def test_hub_404_is_the_verified_state(self):
        status, detail = MODULE.classify_hub_reach(
            1,
            "vllm.cpp: HuggingFace answered HTTP 404 for "
            "'/api/models/does-not-exist/nope/refs'. Repository "
            "'does-not-exist/nope' or the revision it was asked for does not exist.",
        )
        self.assertEqual(status, "ok", detail)

    def test_authorization_answer_also_proves_the_session_completed(self):
        status, _ = MODULE.classify_hub_reach(
            1,
            "vllm.cpp: HuggingFace refused repository 'does-not-exist/nope' with "
            "HTTP 401. The repository is private or gated.",
        )
        self.assertEqual(status, "ok")

    def test_the_build_option_message_is_a_FAILURE_not_a_pass(self):
        """THE CASE THE AUDIT EXISTS FOR.

        A build where the TLS option resolved OFF still exits non-zero for an
        unknown repository, so a check that only asked "did it fail?" would go
        green on an image that cannot fetch anything.
        """
        status, detail = MODULE.classify_hub_reach(
            1,
            "vllm.cpp: this build cannot speak HTTPS, so it cannot reach "
            "https://huggingface.co/. Rebuild with -DVLLM_CPP_HF_DOWNLOAD=ON and "
            "one of -DVLLM_CPP_OPENSSL=ON or -DVLLM_CPP_BUILD_BORINGSSL=ON.",
        )
        self.assertEqual(status, "fail", detail)
        self.assertIn("cannot speak HTTPS", detail)

    def test_a_no_tls_build_is_not_rescued_by_a_404_elsewhere_in_the_log(self):
        """Order matters: the build-option message WINS over a stray 404 line.

        A server prints many lines. If the no-TLS marker check ran after the
        404 check, an image whose logs happened to contain that phrase would be
        classified verified while carrying no TLS at all.
        """
        status, _ = MODULE.classify_hub_reach(
            1,
            "some earlier line mentioning HuggingFace answered HTTP 404\n"
            "vllm.cpp: this build cannot speak HTTPS, so it cannot reach "
            "https://huggingface.co/.",
        )
        self.assertEqual(status, "fail")

    def test_a_transport_failure_is_UNVERIFIED_not_verified(self):
        """No egress is not evidence about the build, in either direction."""
        status, detail = MODULE.classify_hub_reach(
            1,
            "vllm.cpp: cannot reach https://huggingface.co/ for repository "
            "'does-not-exist/nope': Connection",
        )
        self.assertEqual(status, "unverified", detail)

    def test_an_unrecognised_failure_is_UNVERIFIED_not_verified(self):
        status, _ = MODULE.classify_hub_reach(1, "Segmentation fault")
        self.assertEqual(status, "unverified")

    def test_exit_zero_for_a_repository_that_does_not_exist_is_a_FAILURE(self):
        """The resolver must have RUN. A zero exit means it did not refuse."""
        status, detail = MODULE.classify_hub_reach(0, "serving on 0.0.0.0:8000")
        self.assertEqual(status, "fail", detail)

    def test_the_probed_repository_is_a_well_formed_org_slash_repo(self):
        """`IsValidHfRepoId` requires exactly one '/' and no leading special.

        A malformed identifier would be refused by the GRAMMAR before any socket
        opened, and the audit would then measure the parser rather than TLS.
        """
        repo = MODULE.HUB_REACH_MODEL
        self.assertEqual(repo.count("/"), 1)
        org, name = repo.split("/")
        self.assertTrue(org and name)
        for part in (org, name):
            self.assertTrue(part[0].isalnum() and part[-1].isalnum(), part)


class HubReachWiringTests(unittest.TestCase):
    def test_the_audit_runs_by_default(self):
        """Opt-in would let a lane ship the feature disabled and validate green."""
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"--skip-hub-reach"', source)
        self.assertIn("if not args.skip_hub_reach:", source)

    def test_a_failing_audit_becomes_a_validation_error(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('if hub_reach_status == "fail":', source)
        self.assertIn('errors.append(f"hub reach: {hub_reach_detail}")', source)

    def test_the_runtime_image_installs_the_tls_library_and_a_trust_store(self):
        """Both halves. libssl3 alone cannot verify a certificate."""
        dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
        runtime = dockerfile.split("FROM ${UBUNTU_BASE} AS runtime-base", 1)[1]
        runtime = runtime.split("\nFROM ", 1)[0]
        self.assertIn("libssl3", runtime)
        self.assertIn("ca-certificates", runtime)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=2).result.wasSuccessful() else 1)
