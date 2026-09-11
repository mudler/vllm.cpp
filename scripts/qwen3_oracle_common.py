"""Shared capture contracts. Metadata verification does not prove GPU execution.

vLLM e126687a9a828d513c01a07cd69f025f27d63280:
entrypoints/llm.py:194-204 defines revision, seed, and production defaults.
config/cache.py:119 defines auto as the model dtype. The resolved engine
configuration is observed separately from the requested selector.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile


class CaptureError(RuntimeError):
    """A capture cannot establish its declared input or output contract."""


def require(condition, detail, classification="ARTIFACT_MISMATCH"):
    if not condition:
        raise CaptureError(f"{classification}: {detail}")


def add_options(parser):
    # None preserves the original constructor contract for legacy callers.
    # vLLM resolves omitted selectors to auto and an omitted seed to zero.
    parser.add_argument("--kv-cache-dtype", choices=("auto", "bfloat16", "fp8_e4m3"))
    parser.add_argument("--execution-mode", choices=("production", "eager"))
    parser.add_argument("--enforce-eager", action="store_true",
                        help="alias for --execution-mode eager (diagnostic)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-revision")
    parser.add_argument("--vllm-revision")
    parser.add_argument("--provenance-out")
    parser.add_argument("--vllm-wheel")
    parser.add_argument("--runtime-manifest")


def finish_args(parser, argv):
    args = parser.parse_args(argv)
    if args.enforce_eager and args.execution_mode == "production":
        parser.error("--enforce-eager conflicts with --execution-mode production")
    args.execution_mode = args.execution_mode or ("eager" if args.enforce_eager else "production")
    args.enforce_eager = args.execution_mode == "eager"
    for name in ("runs", "max_tokens", "topk"):
        if hasattr(args, name) and getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def llm_kwargs(args, gpu_mem):
    kwargs = dict(model=args.model, dtype="bfloat16", enforce_eager=args.enforce_eager,
                  gpu_memory_utilization=gpu_mem)
    if args.kv_cache_dtype is not None:
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.model_revision is not None:
        kwargs.update(revision=args.model_revision, tokenizer_revision=args.model_revision)
    return kwargs


def mode_narration(args):
    mode = "eager diagnostic" if args.enforce_eager else "production"
    return f"oracle execution: {mode}; enforce_eager={args.enforce_eager}"


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    return {"sha256": sha256(path), "size": Path(path).stat().st_size}


def file_identity(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino


def check_prompts(script_path, prompts):
    root = Path(script_path).resolve().parents[1]
    sources = {}
    for name in ("qwen3-oracle-capture.py", "qwen3-neartie-gap.py"):
        path = root / "scripts" / name
        tree = ast.parse(path.read_text())
        declarations = [node.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "PROMPTS"
                                for target in node.targets)]
        require(len(declarations) == 1 and ast.literal_eval(declarations[0]) == prompts,
                f"{name} differs from the prompt battery", "PROMPTS_MISMATCH")
        sources[str(path.relative_to(root))] = file_record(path)
    path = root / "tests/parity/test_qwen35_paged_engine.cpp"
    match = re.search(r"Prompts\(\)\s*\{.*?\bp\s*=\s*\{(.*?)\};", path.read_text(), re.S)
    strings = re.findall(r'"(?:[^"\\]|\\.)*"', match.group(1)) if match else []
    require([json.loads(value) for value in strings] == prompts and len(prompts) == 16,
            "Qwen3.5 C++ Prompts differs from the prompt battery", "PROMPTS_MISMATCH")
    sources[str(path.relative_to(root))] = file_record(path)
    sources["scripts/qwen3_oracle_common.py"] = file_record(Path(__file__))
    return {"prompts_sha256": hashlib.sha256(json_bytes(prompts)).hexdigest(), "scripts": sources}


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise CaptureError(f"ARTIFACT_MISMATCH: cannot read {path}: {error}") from error
    require(isinstance(value, dict), f"{path} must contain an object")
    return value


def identity(config):
    def get(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    result = {"model_type": get(config, "model_type"),
              "architectures": get(config, "architectures"),
              "text_model_type": get(get(config, "text_config", {}), "model_type")}
    require(all(result[key] is None or isinstance(result[key], str)
                for key in ("model_type", "text_model_type")), "model type identity is malformed")
    architectures = result["architectures"]
    require(architectures is None or isinstance(architectures, list)
            and all(isinstance(name, str) for name in architectures), "model architectures are malformed")
    return result


def is_qwen35(value):
    names = [value.get("model_type"), value.get("text_model_type"),
             *(value.get("architectures") or [])]
    return any("qwen35" in re.sub(r"[^a-z0-9]", "", str(name).lower()) for name in names)


def full_revision(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def git_blob_sha1(path):
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identity(args):
    root = Path(args.model).absolute()
    config = read_json(root / "config.json") if (root / "config.json").is_file() else {}
    observed = identity(config)
    strict = is_qwen35(observed)
    missing = []
    if not config:
        missing.append("local model config")
    if not args.model_revision:
        missing.append("model revision")
    else:
        require(full_revision(args.model_revision), "model revision must be a full commit SHA")
    records = {}
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(part.startswith(".") for part in relative.parts) or not path.is_file():
                continue
            record = file_record(path)
            records[str(relative)] = record
            if not args.model_revision:
                continue
            metadata = root / ".cache/huggingface/download" / (str(relative) + ".metadata")
            revision, etag = None, None
            if metadata.is_file():
                lines = metadata.read_text().splitlines()
                require(len(lines) >= 2, f"incomplete download metadata for {relative}")
                revision, etag = lines[:2]
            elif root.parent.name == "snapshots" and path.is_symlink():
                revision, etag = root.name, path.resolve().name
            if revision is None:
                missing.append(f"artifact revision: {relative}")
                continue
            require(revision == args.model_revision, f"model revision differs for {relative}")
            if re.fullmatch(r"[0-9a-f]{64}", etag or ""):
                require(record["sha256"] == etag, f"model hash differs for {relative}")
            elif re.fullmatch(r"[0-9a-f]{40}", etag or ""):
                require(git_blob_sha1(path) == etag, f"model Git-blob hash differs for {relative}")
            else:
                require(False, f"unsupported artifact identity for {relative}")
            record["revision"] = revision
    if not records:
        missing.append("model artifacts")
    result = {"path": str(root), "requested_revision": args.model_revision,
              "identity": observed, "files": records, "missing": missing}
    if strict:
        require(not missing, "strict model identity is incomplete: " + ", ".join(missing))
        require(observed["model_type"] and observed["architectures"], "strict model identity is incomplete")
        require("text_config" not in config or observed["text_model_type"], "strict text model identity is incomplete")
    return result


def distribution_metadata_files(distribution):
    """Record the filesystem metadata read by Python's selected distribution."""
    # CPython importlib.metadata: Distribution.version reads metadata, whose
    # read_text calls try METADATA, PKG-INFO, then an old egg-info file.
    # PathDistribution joins those names to _path, not to the package root or
    # a guessed version directory. RECORD need not exist.
    root = getattr(distribution, "_path", None)
    if not isinstance(distribution, importlib.metadata.PathDistribution) or not isinstance(root, Path):
        return {}
    records = {}
    for name in ("METADATA", "PKG-INFO", ""):
        path = root / name
        consumed = distribution.read_text(name)
        if consumed is not None:
            records[str(path.absolute())] = file_record(path)
        if consumed:
            break
    return records


def runtime_identity(module, args, strict):
    missing = []
    requested = args.vllm_revision
    if requested:
        require(full_revision(requested), "vLLM revision must be a full commit SHA")
    else:
        missing.append("requested vLLM revision")
    filename = getattr(module, "__file__", None)
    package = Path(filename).resolve().parent if filename else None
    files = {}
    if package and package.is_dir():
        files = {"vllm/" + str(path.relative_to(package)): file_record(path)
                 for path in sorted(package.rglob("*")) if path.is_file()
                 and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    if not files:
        missing.append("imported package bytes")
    version = getattr(module, "__version__", None)
    metadata_files = {}
    revision, verification, source_root = None, None, None
    if package:
        def git(*argv):
            try:
                return subprocess.run(["git", "-C", str(package), *argv], capture_output=True, text=True,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0"))
            except FileNotFoundError:
                return subprocess.CompletedProcess(argv, 127, "", "git is unavailable")
        probe = git("ls-files", "--error-unmatch", Path(filename).name)
        if probe.returncode == 0:
            head = git("rev-parse", "HEAD")
            require(head.returncode == 0, "cannot read imported source revision")
            revision, verification = head.stdout.strip(), "clean_git_source"
            source_root = git("rev-parse", "--show-toplevel").stdout.strip()
            require(git("diff", "--quiet", "HEAD", "--", source_root).returncode == 0,
                    "imported vLLM source has tracked changes")
        try:
            distribution = importlib.metadata.distribution("vllm")
        except importlib.metadata.PackageNotFoundError:
            distribution = None
        if distribution:
            require(str(distribution.version) == str(version), "imported and installed vLLM versions differ")
            metadata_files = distribution_metadata_files(distribution)
    prefix_match = re.search(r"(?:\+|\.)g([0-9a-f]{7,40})(?:[.+-]|$)", str(version))
    prefix = prefix_match.group(1) if prefix_match else None
    if revision and prefix:
        require(revision.startswith(prefix), "installed version disagrees with imported Git source")
    if revision is None and prefix:
        revision, verification = prefix, "installed_version_vcs_prefix"
    if revision is None:
        missing.append("observed source or installed VCS revision")
    elif requested:
        require(requested.startswith(revision), "observed vLLM revision differs from requested revision")
    wheel = {"path": args.vllm_wheel, "sha256": None}
    if args.vllm_wheel:
        wheel["sha256"] = sha256(args.vllm_wheel)
        try:
            archive = zipfile.ZipFile(args.vllm_wheel)
        except zipfile.BadZipFile as error:
            raise CaptureError("ARTIFACT_MISMATCH: vLLM wheel is not a valid archive") from error
        with archive:
            members = [entry for entry in archive.infolist()
                       if entry.filename.startswith("vllm/") and not entry.is_dir()
                       and "__pycache__" not in entry.filename.split("/")
                       and not entry.filename.endswith(".pyc")]
            require(len({entry.filename for entry in members}) == len(members), "duplicate wheel package member")
            require({entry.filename for entry in members} == set(files), "wheel and imported package file sets differ")
            for entry in members:
                digest = hashlib.sha256()
                with archive.open(entry) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                require(digest.hexdigest() == files[entry.filename]["sha256"],
                        f"wheel and imported package bytes differ: {entry.filename}")
    else:
        missing.append("vLLM wheel")
    image = {"digest": None, "source": "launcher_attestation",
             "verification": "The Python process did not independently measure its container image."}
    manifest_hash = None
    if args.runtime_manifest:
        launcher = read_json(args.runtime_manifest)
        require(full_revision(launcher.get("vllm_revision")), "launcher vLLM revision is missing or invalid")
        require(launcher["vllm_revision"] == requested, "launcher and requested vLLM revisions differ")
        require(launcher.get("wheel_sha256") == wheel["sha256"] and wheel["sha256"] is not None,
                "launcher and measured wheel hashes differ")
        require(re.fullmatch(r"sha256:[0-9a-f]{64}", str(launcher.get("image_digest"))),
                "launcher image identity must be an immutable sha256 digest")
        image["digest"] = launcher["image_digest"]
        manifest_hash = sha256(args.runtime_manifest)
    else:
        missing.append("launcher image attestation")
    if strict:
        require(not missing, "strict runtime identity is incomplete: " + ", ".join(missing))
    return {"version": version, "revision": revision, "requested_revision": requested,
            "revision_verification": verification,
            "revision_limit": "An installed +g suffix verifies only the recorded VCS prefix; the full requested SHA is separate.",
            "source_root": source_root, "package_files": files, "distribution_metadata": metadata_files,
            "wheel": wheel, "image": image,
            "launcher_manifest_sha256": manifest_hash, "missing": missing}


def resolved_context(args, llm, model, runtime, prompt_record, batch_size):
    config = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
    model_config = getattr(config, "model_config", None)
    resolved_identity = identity(getattr(model_config, "hf_config", None))
    strict = is_qwen35(model["identity"]) or is_qwen35(resolved_identity)
    if strict:
        require(is_qwen35(model["identity"]) and not model["missing"], "strict model artifacts are not verified")
        require(model["identity"] == resolved_identity, "artifact and runtime model identities differ")
        require(not runtime["missing"], "strict runtime identity is incomplete")
        require(args.runs >= 10, "strict capture requires at least ten deterministic repeats")
    dtype = str(getattr(model_config, "dtype", "unknown")).removeprefix("torch.")
    reported_cache = getattr(getattr(config, "cache_config", None), "cache_dtype", None)
    resolved_cache = dtype if reported_cache == "auto" else reported_cache
    requested_cache = args.kv_cache_dtype or "auto"
    expected_cache = "bfloat16" if requested_cache == "auto" else requested_cache
    if strict:
        require(dtype == "bfloat16" and resolved_cache == expected_cache, "requested and resolved cache dtypes differ")
        require(getattr(model_config, "enforce_eager", None) == args.enforce_eager, "runtime execution mode differs")
        require(getattr(model_config, "seed", None) == (args.seed or 0), "runtime seed differs")
        require(getattr(model_config, "revision", None) == args.model_revision
                and getattr(model_config, "tokenizer_revision", None) == args.model_revision,
                "runtime model or tokenizer revision differs")
    return {"schema_version": 1, "regime": "qwen3_5_strict" if strict else "legacy_distributional",
            "provenance_status": "incomplete" if model["missing"] or runtime["missing"] else "complete",
            "arguments": vars(args), "command": [sys.executable, *sys.argv],
            "model": model, "runtime": runtime, "resolved_model_identity": resolved_identity,
            "cache": {"requested": requested_cache, "reported_selector": reported_cache,
                      "resolved": resolved_cache, "physical_dtype": None,
                      "verification": "Resolved configuration only; physical cache storage is not measured."},
            "execution_mode": args.execution_mode,
            "purpose": "diagnostic" if args.enforce_eager else "production_capture",
            "batching": {"batch_size": batch_size, "concurrency": batch_size},
            "sampling": {"temperature": 0.0, "max_tokens": args.max_tokens, "seed": args.seed or 0},
            "repetitions": args.runs, **prompt_record}


def strict_inputs(args, model):
    if is_qwen35(model["identity"]):
        require(args.vllm_revision and args.vllm_wheel and args.runtime_manifest,
                "strict capture requires vLLM revision, wheel, and runtime manifest")
        require(args.runs >= 10, "strict capture requires at least ten deterministic repeats")


def confirm_inputs(args, module, context, script_path, prompts):
    require(model_identity(args) == context["model"], "model artifacts changed during capture")
    require(runtime_identity(module, args, context["regime"] == "qwen3_5_strict") == context["runtime"],
            "runtime identity changed during capture")
    require(check_prompts(script_path, prompts) == {key: context[key] for key in ("prompts_sha256", "scripts")},
            "capture scripts or prompts changed during capture")


def sampling_record(params):
    """Serialize the supplied sampling object without claiming engine resolution."""
    def encode(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, dict):
            return {str(key): encode(item) for key, item in value.items()}
        if isinstance(value, (set, frozenset)):
            return [encode(item) for item in sorted(value, key=repr)]
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        return str(value)
    names = getattr(params, "__struct_fields__", None)
    values = vars(params) if names is None else {name: getattr(params, name) for name in names}
    return {name: encode(value) for name, value in values.items()}


def record_sampling(context, params, key="sampling"):
    context[key + "_normalized"] = sampling_record(params)
    context[key + "_resolved"] = None
    # Pinned vLLM v1/engine/input_processor.py:356,364-369 resolves a clone.
    # Reading the supplied object after generate() cannot observe that request.
    context[key + "_resolution"] = {
        "status": "unobserved",
        "limit": "vLLM resolves a cloned request using generation config and tokenizer; "
                 "the engine-resolved values are not observed.",
    }
    context[key] = {name: getattr(params, name) for name in ("temperature", "max_tokens", "seed")}


def capture_input_paths(args, module, context, script_path):
    """Name the files used to verify this capture before publication."""
    model_root = Path(context["model"]["path"])
    paths = {model_root / name for name in context["model"]["files"]}
    paths.update(model_root / ".cache/huggingface/download" / (name + ".metadata")
                 for name in context["model"]["files"])
    filename = getattr(module, "__file__", None)
    if filename:
        package_root = Path(filename).resolve().parent.parent
        paths.update(package_root / name for name in context["runtime"]["package_files"])
    paths.update(Path(path) for path in context["runtime"]["distribution_metadata"])
    project_root = Path(script_path).resolve().parents[1]
    paths.update(project_root / name for name in context["scripts"])
    paths.update(Path(path) for path in (args.vllm_wheel, args.runtime_manifest) if path)
    return paths


def publish(directory, payloads, provenance, manifest_name, external=None, *, protected_inputs=()):
    """Publish validated results; legacy callers retain their overwrite contract."""
    directory = Path(directory).resolve()
    provenance["outputs"] = {name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                             for name, data in sorted(payloads.items())}
    provenance["output_sha256"] = hashlib.sha256(json_bytes(provenance["outputs"])).hexdigest()
    manifest = json_bytes(provenance)
    targets = {directory / name: data for name, data in payloads.items()}
    targets[directory / manifest_name] = manifest
    if external:
        path = Path(external).absolute()
        # Preserve the final name so an alias to another manifest stays a
        # distinct target. An explicit default name still publishes once.
        path = path.parent.resolve() / path.name
        require(path not in targets or path == directory / manifest_name, "provenance path overlaps an output")
        targets[path] = manifest
    legacy = provenance["regime"] == "legacy_distributional"
    # stat follows symbolic links and identifies hardlinks to the same input.
    protected = {file_identity(path) for path in protected_inputs if Path(path).exists()}
    backups = {}
    resolved_targets, output_identities = set(), set()
    for target in targets:
        resolved = target.resolve()
        require(resolved not in resolved_targets, f"publication paths alias an output: {target}")
        resolved_targets.add(resolved)
        if target.exists():
            inode = file_identity(target)
            require(inode not in output_identities, f"publication paths alias an output: {target}")
            output_identities.add(inode)
        require(not target.exists() or file_identity(target) not in protected,
                f"publication path overlaps an input: {target}")
        require(legacy or not target.exists(), f"refusing to overwrite {target}")
        if target.exists():
            backups[target] = target.read_bytes()
    created = []
    try:
        for target, data in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb" if legacy else "xb") as stream:
                created.append(target)
                stream.write(data)
    except BaseException:
        for target in created:
            if target in backups:
                target.write_bytes(backups[target])
            else:
                target.unlink(missing_ok=True)
        raise
