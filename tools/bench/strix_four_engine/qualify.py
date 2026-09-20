"""Matched Qwen3-4B qualification. All runtime artifacts are supplied, never installed."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import statistics
import subprocess
import sys
import tarfile
import time

from tools.bench.strix_four_engine.audit import CONVERTER_INVENTORY, publish_result
from tools.bench.strix_four_engine.child_lifecycle import ChildLifecycle, controller_lifecycle

PROMPTS = ["The capital city of France is", "The three primary colors are",
           "Water boils at a temperature of", "The Pythagorean theorem states that",
           "In 1969, humans first walked on", "A prime number is a natural number"]
MODEL = "1cfa9a7208912126459214e8b04321603b3df60c"
PINS = {"vllm.cpp": "6e3cbfb940be89e28d1d71c264fd8c3a4e44afeb",
        "vLLM": "e126687a9a828d513c01a07cd69f025f27d63280",
        "patched SGLang": "f63458b5beaceabbd9d749b9fc956370e1b649e6",
        "llama.cpp": "10bf611e533d81f739128304991c5e133c6aebd8"}
RESOLVED = dict(backend="rocm", model_dtype="bfloat16", kv_dtype="bfloat16",
                context_per_sequence=2048, slots=4, prefix_caching=False)


def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key: " + key)
        value[key] = item
    return value


def parse(data):
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_binding(value):
    path = Path(value["path"])
    if not path.is_absolute() or not path.is_file() or sha(path) != value["sha256"]:
        raise ValueError("file binding mismatch: " + str(path))
    return dict(path=str(path), resolved=str(path.resolve()), sha256=value["sha256"])


def model_binding(value):
    if value["revision"] != MODEL:
        raise ValueError("model revision mismatch")
    root = Path(value["directory"])
    if not root.is_absolute() or not root.is_dir() or not value["files"]:
        raise ValueError("model directory/files required")
    identities = []
    for name, digest in value["files"].items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("model path escapes directory")
        identities.append(file_binding(dict(path=str(path), sha256=digest)))
    for required in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        if required not in value["files"]:
            raise ValueError("missing model file: " + required)
    index = parse((root / "model.safetensors.index.json").read_text())["weight_map"]
    if not set(index.values()) <= value["files"].keys():
        raise ValueError("unbound model shard")
    return identities


def validate_manifest(manifest):
    if os.environ.get("RC_DEVICE") != "strix:gpu0" or not os.environ.get("RC_JOB_ID"):
        raise ValueError("qualification requires a Strix lease")
    if manifest["schema"] != 1 or not isinstance(manifest["run_id"], str) or not manifest["run_id"]:
        raise ValueError("invalid manifest identity")
    identities = model_binding(manifest["model"])
    identities.append(file_binding(manifest["gguf"]))
    identities.append(file_binding(manifest["audit"]))
    audit = parse(Path(manifest["audit"]["path"]).read_text())
    if (audit["result"] != "PASS" or audit["tensor_count"] != 398 or
            audit["model"]["revision"] != MODEL or audit["model"]["files"] != manifest["model"]["files"] or
            audit["gguf_sha256"] != manifest["gguf"]["sha256"] or
            audit["converter"]["revision"] != PINS["llama.cpp"] or
            audit["converter"]["inventory_sha256"] != CONVERTER_INVENTORY):
        raise ValueError("conversion audit does not bind model")
    if set(manifest["engines"]) != set(PINS):
        raise ValueError("exactly four engines required")
    for name, engine in manifest["engines"].items():
        if engine["source_revision"] != PINS[name]:
            raise ValueError("engine source revision mismatch: " + name)
        identities.append(file_binding(engine["source"]))
        with tarfile.open(engine["source"]["path"]) as source:
            if source.pax_headers.get("comment") != PINS[name]:
                raise ValueError("source archive revision mismatch")
        if name == "patched SGLang":
            identities.append(file_binding(engine["patch"]))
            if engine["patch"]["sha256"] != "7dde3188aaefac2890de7658c90cd65477b92fccd3bdb30bd7a17fe5365397e9":
                raise ValueError("SGLang compatibility patch mismatch")
        identities.append(file_binding(engine["binary"]))
        env = engine["environment"]
        if not isinstance(env["variables"], dict) or not env["files"]:
            raise ValueError("bound environment required")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in env["variables"].items()):
            raise ValueError("environment values must be strings")
        if env["variables"].get("VLLM_ALLOW_INSECURE_SERIALIZATION") not in (None, "0"):
            raise ValueError("insecure serialization refused")
        for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            if os.environ.get(key) != env["variables"].get(key):
                raise ValueError("lease device visibility mismatch")
        bound = [file_binding(item) for item in env["files"]]
        identities.extend(bound)
        command = engine["command"]
        if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
            raise ValueError("adapter command must be argv")
        if command[0] != engine["binary"]["path"]:
            raise ValueError("command executable not bound")
        known = {x["path"] for x in bound} | {engine["binary"]["path"]}
        if any(Path(x).is_absolute() and Path(x).is_file() and x not in known for x in command):
            raise ValueError("unbound command file")
    for key, low, high in (("timeout_seconds", 1, 7200), ("log_bytes", 1024, 536870912)):
        if type(manifest[key]) is not int or not low <= manifest[key] <= high:
            raise ValueError("invalid resource limit: " + key)
    return identities


class Adapter:
    """Bounded JSON-lines transport; stdout is protocol, stderr is diagnostic."""
    def __init__(self, record, output, limit, timeout):
        self.output, self.limit, self.timeout = output, limit, timeout
        self.count, self.buffer, self.received, self.rss = 0, b"", 0, 0
        self.closed, self.stderr, self.selector = False, None, None
        self.lifecycle = ChildLifecycle()
        self.lifecycle.abandon_transport = self._abandon_transport
        self.lifecycle_evidence = self.lifecycle.evidence
        try:
            self.stderr = (output / "stderr.log").open("xb")
            self.process = subprocess.Popen(record["command"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, env=record["environment"]["variables"],
                                        start_new_session=True)
            self.lifecycle.process = self.process
            self.selector = selectors.DefaultSelector()
            for stream in (self.process.stdout, self.process.stderr):
                os.set_blocking(stream.fileno(), False)
                self.selector.register(stream, selectors.EVENT_READ)
        except BaseException as error:
            self.lifecycle.cleanup()
            self.lifecycle.release()
            self._close_transport()
            self.lifecycle.annotate(error)
            raise

    def _abandon_transport(self):
        self.closed = True
        self._close_transport()

    def _close_transport(self):
        resources = [self.selector, self.stderr]
        if self.lifecycle.process is not None:
            resources.extend((self.process.stdin, self.process.stdout, self.process.stderr))
        for resource in resources:
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    self.lifecycle.record_error(error)

    def sample_rss(self):
        total = 0
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if int(stat[2]) == self.process.pid:
                    total += int(stat[21]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                continue
        self.rss = max(self.rss, total)

    def exchange(self, command):
        self.count += 1
        command = dict(command, schema=1, id=self.count)
        self.process.stdin.write((json.dumps(command) + "\n").encode())
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            self.sample_rss()
            if b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                result = parse(line)
                if type(result.get("schema")) is not int or result["schema"] != 1 or type(result.get("id")) is not int or result["id"] != self.count:
                    raise ValueError("duplicate, missing, or out-of-order adapter result")
                if result.get("status") != "ok":
                    raise ValueError("adapter failure: " + str(result.get("error")))
                return result
            for key, _ in self.selector.select(min(0.05, max(0, deadline - time.monotonic()))):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    self.selector.unregister(key.fileobj)
                    continue
                self.received += len(chunk)
                if self.received > self.limit:
                    raise ValueError("adapter log limit exceeded")
                if key.fileobj is self.process.stdout:
                    self.buffer += chunk
                else:
                    self.stderr.write(chunk)
                    self.stderr.flush()
            if self.process.poll() is not None and not self.buffer:
                raise ValueError("adapter exited before result: " + str(self.process.returncode))
        raise TimeoutError("adapter command timeout")

    def close(self):
        if self.closed:
            return
        error = None
        try:
            if self.process.poll() is not None:
                raise ValueError("adapter exited before shutdown protocol completion")
            self.exchange(dict(command="shutdown"))
            self.process.wait(timeout=10)
            if self.process.returncode != 0 or self.buffer.strip():
                raise ValueError("adapter teardown failed or surplus result")
            self.lifecycle.drain(time.monotonic() + 1)
            try:
                os.killpg(self.process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError("adapter descendants survived shutdown")
            self.lifecycle.check_remaining_children()
        except BaseException as failure:
            error = failure
            self.lifecycle.cleanup()
        finally:
            self.lifecycle.release()
            self._close_transport()
            self.closed = True
        if error is not None:
            self.lifecycle.annotate(error)
            raise error
        if self.lifecycle_evidence["cleanup_errors"]:
            raise RuntimeError("adapter lifecycle restoration failed: " + repr(self.lifecycle_evidence))


def tokens(value):
    if not isinstance(value, list) or not value or any(type(x) is not int or not 0 <= x < 151936 for x in value):
        raise ValueError("invalid token IDs")
    return value


def validate_run(result, prompts, concurrency, expected=None):
    requests = result["requests"]
    if len(requests) != 6 or any(type(r["index"]) is not int for r in requests) or [r["index"] for r in requests] != list(range(6)):
        raise ValueError("request IDs missing, duplicated, or reordered")
    start, stop = result["started"], result["completed"]
    if not all(type(x) in (int, float) and math.isfinite(x) for x in (start, stop)) or stop <= start:
        raise ValueError("invalid corpus timing")
    events, outputs = [], []
    for index, request in enumerate(requests):
        if tokens(request["prompt_ids"]) != prompts[index] or request["status"] != "ok":
            raise ValueError("consumed input mismatch or failed request")
        output = tokens(request["output_ids"])
        if len(output) != 128 or request["finish_reason"] != "length":
            raise ValueError("generation stopped or truncated")
        dispatch, completion = request["dispatched"], request["completed"]
        if not all(type(x) in (int, float) and math.isfinite(x) for x in (dispatch, completion)) or not start <= dispatch < completion <= stop:
            raise ValueError("invalid request timing boundary")
        events.extend(((dispatch, 1), (completion, -1)))
        outputs.append(output)
    active = 0
    for _, change in sorted(events):
        active += change
        if active > concurrency:
            raise ValueError("concurrency overrun")
    if expected is not None and outputs != expected:
        raise ValueError("token-exact or repeatability mismatch")
    return outputs, 768 / (stop - start)


def runtime_evidence(engine, name, manifest, workload):
    binding = engine.get("runtime_evidence")
    if binding is None:
        return "PENDING"
    file_binding(binding)
    evidence = parse(Path(binding["path"]).read_text())
    required = dict(schema=1, verified=True, engine=name, source_revision=PINS[name],
                    binary_sha256=engine["binary"]["sha256"], model_revision=MODEL,
                    model_files=manifest["model"]["files"], workload_sha256=workload,
                    environment_sha256=canonical(engine["environment"]), resolved=RESOLVED)
    prior_run = evidence.get("evidence_run_id")
    if (any(evidence.get(k) != v for k, v in required.items()) or
            not isinstance(prior_run, str) or not prior_run.strip() or prior_run == manifest["run_id"]):
        raise ValueError("unverified or mismatched runtime evidence: " + name)
    if name == "llama.cpp" and evidence.get("gguf_sha256") != manifest["gguf"]["sha256"]:
        raise ValueError("runtime GGUF binding mismatch")
    if not evidence.get("artifacts") or not any(item.get("kind") == "trace" for item in evidence["artifacts"]):
        raise ValueError("runtime evidence artifacts required")
    for artifact in evidence["artifacts"]:
        if artifact["kind"] not in ("trace", "log"):
            raise ValueError("unknown runtime evidence kind")
        file_binding(artifact)
    return "PASS"


def execute(manifest, output):
    before = validate_manifest(manifest)
    results = {name: dict(status="PENDING", runs=[]) for name in PINS}
    prompts = None
    expected = {}
    workload = None
    # All engines qualify before any measured corpus. Engines are never resident together.
    for phase in ("qualification", "measurement"):
        if phase == "measurement" and any(r["status"] != "PASS" for r in results.values()):
            break
        for name in ("vLLM", "vllm.cpp", "patched SGLang", "llama.cpp"):
            record = manifest["engines"][name]
            directory = output / (name.replace(" ", "-") + "-" + phase)
            directory.mkdir()
            adapter = None
            try:
                adapter = Adapter(record, directory, manifest["log_bytes"], manifest["timeout_seconds"])
                config = adapter.exchange(dict(command="configure", engine=name, model=manifest["model"]["directory"],
                                               gguf=manifest["gguf"]["path"], prompts=PROMPTS, prompt_ids=prompts))
                received = [tokens(p) for p in config["prompt_ids"]]
                if len(received) != 6 or (prompts is not None and received != prompts):
                    raise ValueError("canonical tokenizer mismatch")
                if prompts is None:
                    prompts = received
                    workload = canonical(dict(prompts=PROMPTS, prompt_ids=prompts, tokens=128, concurrency=[1, 4],
                                              settings=RESOLVED))
                if config["requested"] != RESOLVED:
                    raise ValueError("requested settings mismatch")
                results[name]["configuration"] = config
                run_errors = []
                for concurrency in (1, 4):
                    repetitions = range(2) if phase == "qualification" else range(4)
                    for repetition in repetitions:
                        run_phase = phase if phase == "qualification" else ("warmup" if repetition == 0 else "measured")
                        run = adapter.exchange(dict(command="run", phase=run_phase, concurrency=concurrency,
                                                    repetition=repetition))
                        retained = dict(phase=run_phase, concurrency=concurrency, repetition=repetition, raw=run)
                        results[name]["runs"].append(retained)
                        try:
                            oracle = expected.get(concurrency)
                            ids, rate = validate_run(run, prompts, concurrency, oracle)
                            if name == "vLLM" and phase == "qualification" and repetition == 0:
                                expected[concurrency] = ids
                            if name != "vLLM" and oracle is None:
                                raise ValueError("pinned vLLM reference unavailable")
                            retained["tokens_per_second"] = rate
                        except ValueError as error:
                            retained["error"] = str(error)
                            run_errors.append(str(error))
                proof_status = runtime_evidence(record, name, manifest, workload)
                results[name]["status"] = "FAIL" if run_errors else proof_status
                if run_errors:
                    results[name]["errors"] = run_errors
            except Exception as error:
                results[name]["status"] = "FAIL"
                results[name]["error"] = str(error)
                if hasattr(error, "lifecycle_evidence"):
                    results[name]["lifecycle"] = error.lifecycle_evidence
            finally:
                if adapter is not None:
                    results[name]["sampled_process_group_rss_bytes"] = adapter.rss
                    try:
                        adapter.close()
                    except Exception as error:
                        results[name]["status"] = "FAIL"
                        results[name]["teardown_error"] = str(error)
                    results[name]["lifecycle"] = adapter.lifecycle_evidence
                publish_result(results[name], directory / "result.json")
    try:
        if before != validate_manifest(manifest):
            raise ValueError("artifact identity changed during qualification")
    except Exception as error:
        for result in results.values():
            result["status"] = "FAIL"
            result["binding_error"] = str(error)
    for name, record in manifest["engines"].items():
        if workload is not None:
            try:
                runtime_evidence(record, name, manifest, workload)
            except Exception as error:
                results[name]["status"] = "FAIL"
                results[name]["evidence_error"] = str(error)
    accepted = all(r["status"] == "PASS" for r in results.values())
    if accepted:
        for result in results.values():
            result["throughput"] = {}
            for concurrency in (1, 4):
                values = [r["tokens_per_second"] for r in result["runs"]
                          if r["phase"] == "measured" and r["concurrency"] == concurrency]
                if len(values) != 3:
                    raise ValueError("missing measured repetitions")
                result["throughput"][str(concurrency)] = dict(values=values, median=statistics.median(values),
                                                             minimum=min(values), maximum=max(values))
        for result in results.values():
            for c, rate in result["throughput"].items():
                rate["ratio_to_vllm"] = rate["median"] / results["vLLM"]["throughput"][c]["median"]
    return dict(schema=1, run_id=manifest["run_id"], lease=dict(device=os.environ["RC_DEVICE"], job_id=os.environ["RC_JOB_ID"]),
                status="PASS" if accepted else "NOT_ACCEPTED", engines=results,
                prompt_ids=prompts, workload_sha256=workload, bindings=before, manifest=manifest,
                memory_method="/proc process-group RSS sampled at transport polls, at most 50 ms between polls",
                gpu_memory="unavailable", ttft="unavailable", inter_token_latency="unavailable",
                acceptance="operator-attested bound runtime traces plus token-exact qualification; idle reproduction remains required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    created = False
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("output already exists")
        args.output.mkdir(parents=True)
        created = True
        with controller_lifecycle():
            result = execute(parse(args.manifest.read_text()), args.output)
        publish_result(result, args.output / "result.json")
        return 0 if result["status"] == "PASS" else 1
    except Exception as error:
        if created:
            try:
                publish_result(dict(status="FAIL", error=str(error)), args.output / "result.json")
            except Exception as output_error:
                print("QUALIFICATION_OUTPUT_FAIL: " + str(output_error), file=sys.stderr)
        print("QUALIFICATION_FAIL: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
