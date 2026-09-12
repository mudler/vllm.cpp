#!/usr/bin/env python3
"""Check the DeepSeek-V4 Vision config, index and shard-1 header manifests.

A PORT. This checker comes from a parallel implementation of row
`MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`, preserved at
`row/MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm-CODEX-LINE` (`3f3860851`).
It ties the tensor map this tree derives to the artifact
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at
`86f746b36186f0e567729a5c06a8c918caba82a9` without downloading any weight
payload: the manifests are built from `config.json`, the safetensors index, and
two HTTP RANGE requests covering shard 1's header.

WHAT CHANGED IN THE PORT, AND WHY IT HAD TO. The ported version ALWAYS went to
the network -- every invocation fetched three URLs before it could say anything.
A record gate in this repository has to run offline: CI runs no checker that
makes a network call, and a checker that cannot answer without huggingface.co is
one that fails on a disconnected machine and, worse, whose verdict depends on a
third party's uptime rather than on the tree. So the default mode here VERIFIES
THE COMMITTED FIXTURES AGAINST EACH OTHER AND AGAINST THE DERIVED NAME MAP, with
no network at all, and `--refresh` is the network path that rebuilds them.

WHAT THE OFFLINE MODE CAN AND CANNOT PROVE. It proves that the committed
manifests describe the tensor map this tree derives from the committed
`config.json`: every count, every classification, every vision shape, the
payload byte total and the two content hashes are recomputed rather than read.
It also re-hashes the committed `config.json` bytes and holds them to the
`config_sha256` the index manifest recorded, which ties the two files together.
It CANNOT prove that those bytes are still what the Hugging Face revision
serves: `index_sha256` and `header_sha256` are digests of remote bytes this
repository does not mirror, and only `--refresh` re-reads them.

THE WEIGHT PAYLOAD HAS NEVER BEEN READ, in either mode. The released checkpoint
is 156.287 GiB across 48 shards. Every gate over this artifact in this tree is a
synthetic fixture built to the header this script pins. See `## Owed` in
`.agents/specs/deepseek-v4-flash-vision.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import urllib.request
from pathlib import Path
from typing import Any

REPO = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
REVISION = "86f746b36186f0e567729a5c06a8c918caba82a9"
SHARD = "model-00001-of-00048.safetensors"
ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests/parity/goldens/deepseek_v4_vision"
CONFIG_PATH = FIXTURE_DIR / "config.json"
INDEX_MANIFEST_PATH = FIXTURE_DIR / "index_manifest.json"
HEADER_MANIFEST_PATH = FIXTURE_DIR / "shard1_header_manifest.json"


def url(name: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true"


def get(name: str, byte_range: tuple[int, int] | None = None) -> tuple[bytes, Any]:
    headers = {"User-Agent": "vllm.cpp-deepseek-v4-vision-manifest/1"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url(name), headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
        status = response.status
        content_range = response.headers.get("Content-Range")
    if byte_range is not None:
        if status != 206:
            raise SystemExit(f"{name}: range request returned HTTP {status}, expected 206")
        expected = byte_range[1] - byte_range[0] + 1
        if len(payload) != expected:
            raise SystemExit(f"{name}: range returned {len(payload)} bytes, expected {expected}")
        if content_range is None:
            raise SystemExit(f"{name}: range response has no Content-Range")
    elif status != 200:
        raise SystemExit(f"{name}: request returned HTTP {status}, expected 200")
    return payload, content_range


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fnv1a_lines(lines: list[str]) -> str:
    value = 1469598103934665603
    for line in lines:
        for byte in (line + "\n").encode("utf-8"):
            value ^= byte
            value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return str(value)


def classify(name: str) -> str:
    """Which of the three families a checkpoint tensor belongs to.

    The `vision` set here is EXACTLY the set the loader reads: `vision.*`,
    `aligner.*` and the four learned sentinels. It is the same partition
    `tests/vllm/models/test_deepseek_v4_mm_loader.cpp` applies, and the two are
    compared against the same committed manifest, so a change to one that the
    other does not make turns the suite red.
    """
    if (
        name.startswith("vision.")
        or name.startswith("aligner.")
        or name in {"image_start", "image_end", "image_newline", "image_pad"}
    ):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    return "language"


def official_names(config: dict[str, Any]) -> list[str]:
    """Every tensor name the released checkpoint carries, derived from config."""
    layers = int(config["num_hidden_layers"])
    experts = int(config["n_routed_experts"])
    hash_layers = int(config["num_hash_layers"])
    mtp_layers = int(config["num_nextn_predict_layers"])
    vision_layers = int(config["vision_n_layers"])
    ratios = [int(value) for value in config["compress_ratios"]]
    if len(ratios) < layers:
        raise SystemExit("config: compress_ratios is shorter than num_hidden_layers")

    names = [
        "embed.weight",
        "norm.weight",
        "head.weight",
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
        "vision.patch_embed.proj.weight",
        "vision.patch_embed.proj.bias",
        "vision.norm.weight",
        "aligner.w1.weight",
        "aligner.w1.bias",
        "aligner.w2.weight",
        "aligner.w2.bias",
        "image_start",
        "image_end",
        "image_newline",
        "image_pad",
    ]
    for layer in range(vision_layers):
        prefix = f"vision.blocks.{layer}."
        names.extend(
            prefix + suffix
            for suffix in (
                "norm1.weight",
                "attn.wqkv.weight",
                "attn.wqkv.bias",
                "attn.wo.weight",
                "attn.wo.bias",
                "norm2.weight",
                "mlp.w1.weight",
                "mlp.w2.weight",
            )
        )

    def add_block(prefix: str) -> None:
        attn = prefix + "attn."
        ffn = prefix + "ffn."
        names.extend(
            prefix + suffix
            for suffix in (
                "attn_norm.weight",
                "ffn_norm.weight",
                "hc_attn_base",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_ffn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
            )
        )
        for stem in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            names.extend((attn + stem + ".weight", attn + stem + ".scale"))
        names.extend(attn + suffix for suffix in ("q_norm.weight", "kv_norm.weight", "attn_sink"))
        names.extend(ffn + suffix for suffix in ("gate.weight", "gate.bias", "gate.bias_vl"))
        for stem in ("w1", "w2", "w3"):
            base = ffn + "shared_experts." + stem
            names.extend((base + ".weight", base + ".scale"))
        for expert in range(experts):
            for stem in ("w1", "w2", "w3"):
                base = f"{ffn}experts.{expert}.{stem}"
                names.extend((base + ".weight", base + ".scale"))

    for layer in range(layers):
        prefix = f"layers.{layer}."
        add_block(prefix)
        attn = prefix + "attn."
        if ratios[layer] != 0:
            names.extend(
                attn + "compressor." + suffix
                for suffix in ("ape", "norm.weight", "wgate.weight", "wkv.weight")
            )
        if ratios[layer] == 4:
            names.extend(
                attn + "indexer.compressor." + suffix
                for suffix in ("ape", "norm.weight", "wgate.weight", "wkv.weight")
            )
            names.extend(
                (
                    attn + "indexer.weights_proj.weight",
                    attn + "indexer.wq_b.weight",
                    attn + "indexer.wq_b.scale",
                )
            )
        if layer < hash_layers:
            names.append(prefix + "ffn.gate.tid2eid")

    for layer in range(mtp_layers):
        prefix = f"mtp.{layer}."
        add_block(prefix)
        if layer == 0:
            names.extend(
                (prefix + "main_norm.weight", prefix + "main_proj.weight", prefix + "main_proj.scale")
            )
        if layer + 1 == mtp_layers:
            names.extend(
                prefix + suffix
                for suffix in (
                    "confidence_head.proj.weight",
                    "hc_head_base",
                    "hc_head_fn",
                    "hc_head_scale",
                    "markov_head.markov_w1.weight",
                    "markov_head.markov_w2.weight",
                    "norm.weight",
                )
            )

    if len(names) != len(set(names)):
        raise SystemExit("derived checkpoint name map contains duplicates")
    return sorted(names)


def vision_shape(name: str, config: dict[str, Any]) -> list[int]:
    """The shape the released checkpoint stores for one vision tensor.

    These are the shapes `deepseek_v4_vision_weights.cpp` requires, written once
    here and once there. Both are held to the committed header manifest, which
    is what stops the two descriptions from drifting apart in silence.
    """
    hidden = int(config["hidden_size"])
    vision = int(config["vision_dim"])
    intermediate = int(config["vision_inter_dim"])
    patch = int(config["vision_patch_size"])
    downsample = int(config["vision_downsample_ratio"])
    if name == "vision.patch_embed.proj.weight":
        return [vision, 3 * patch * patch]
    if (
        name in {"vision.patch_embed.proj.bias", "vision.norm.weight"}
        or name.endswith("norm1.weight")
        or name.endswith("norm2.weight")
        or name.endswith("attn.wo.bias")
    ):
        return [vision]
    if name.endswith("attn.wqkv.weight"):
        return [3 * vision, vision]
    if name.endswith("attn.wqkv.bias"):
        return [3 * vision]
    if name.endswith("attn.wo.weight"):
        return [vision, vision]
    if name.endswith("mlp.w1.weight"):
        return [2 * intermediate, vision]
    if name.endswith("mlp.w2.weight"):
        return [vision, intermediate]
    if name == "aligner.w1.weight":
        return [hidden, vision * downsample * downsample]
    if name == "aligner.w2.weight":
        return [hidden, hidden]
    if name.startswith("aligner.") or name.startswith("image_"):
        return [hidden]
    raise SystemExit(f"no released vision shape rule for {name}")


def summary(names: list[str]) -> dict[str, Any]:
    return {"count": len(names), "fnv1a64": fnv1a_lines(names), "sha256": sha256(("\n".join(names) + "\n").encode())}


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def vision_record(name: str, dtype: str, shape: list[int]) -> str:
    return f"{name}\t{dtype}\t{','.join(str(value) for value in shape)}"


# ── the OFFLINE verification, which is what CI and a preflight run ────────────


def check_index_manifest(config_bytes: bytes, config: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """Recompute every derived field of the index manifest. Returns failures."""
    failures: list[str] = []
    names = official_names(config)

    def expect(label: str, actual: Any, wanted: Any) -> None:
        if actual != wanted:
            failures.append(f"index manifest: {label} is {actual!r}, derived {wanted!r}")

    expect("repo", manifest.get("repo"), REPO)
    expect("revision", manifest.get("revision"), REVISION)
    expect("tensor_count", manifest.get("tensor_count"), len(names))
    # The committed config.json IS the bytes this digest was taken over, so the
    # two files cannot be updated independently without this going red.
    expect("config_sha256", manifest.get("config_sha256"), sha256(config_bytes))
    for key, value in summary(names).items():
        expect(f"all_names.{key}", manifest.get("all_names", {}).get(key), value)

    classes: dict[str, list[str]] = {"language": [], "mtp": [], "vision": []}
    for name in names:
        classes[classify(name)].append(name)
    if sum(len(group) for group in classes.values()) != len(names):
        failures.append("index manifest: the classifier did not account for every tensor once")
    for category, group in classes.items():
        recorded = manifest.get("classifications", {}).get(category, {})
        for key, value in summary(group).items():
            expect(f"classifications.{category}.{key}", recorded.get(key), value)
    return failures


def check_header_manifest(config: dict[str, Any], index: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """Recompute every derived field of the shard-1 header manifest."""
    failures: list[str] = []
    names = official_names(config)
    vision_names = sorted(name for name in names if classify(name) == "vision")
    tensors = manifest.get("tensors", {})

    def expect(label: str, actual: Any, wanted: Any) -> None:
        if actual != wanted:
            failures.append(f"header manifest: {label} is {actual!r}, derived {wanted!r}")

    expect("repo", manifest.get("repo"), REPO)
    expect("revision", manifest.get("revision"), REVISION)
    expect("shard", manifest.get("shard"), SHARD)
    expect("vision_tensor_count", manifest.get("vision_tensor_count"), len(vision_names))
    expect("header_tensor_count", manifest.get("header_tensor_count"), len(tensors))

    # Shard 1 holds the whole vision group plus the token embedding, and nothing
    # else. That is a claim about the artifact's layout, so it is stated here
    # rather than left implied by the counts.
    expected_header = sorted(vision_names + ["embed.weight"])
    if sorted(tensors) != expected_header:
        missing = sorted(set(expected_header) - set(tensors))
        unexplained = sorted(set(tensors) - set(expected_header))
        failures.append(
            f"header manifest: tensor set differs, {len(missing)} missing "
            f"{missing[:1]}, {len(unexplained)} unexplained {unexplained[:1]}"
        )
        return failures

    payload_bytes = 0
    records: list[str] = []
    for name in sorted(tensors):
        entry = tensors[name]
        dtype = entry.get("dtype")
        shape = [int(value) for value in entry.get("shape", [])]
        if dtype != "BF16":
            failures.append(f"header manifest: {name} has dtype {dtype!r}, expected BF16")
            continue
        if name == "embed.weight":
            wanted = [int(config["vocab_size"]), int(config["hidden_size"])]
            if shape != wanted:
                failures.append(f"header manifest: embed.weight is {shape}, config derives {wanted}")
            continue
        wanted = vision_shape(name, config)
        if shape != wanted:
            failures.append(f"header manifest: {name} is {shape}, derived {wanted}")
            continue
        size = 2
        for dimension in shape:
            size *= dimension
        payload_bytes += size
        records.append(vision_record(name, dtype, shape))

    expect("vision_payload_bytes", manifest.get("vision_payload_bytes"), payload_bytes)
    expect("vision_records_fnv1a64", manifest.get("vision_records_fnv1a64"), fnv1a_lines(records))
    expect(
        "vision_records_sha256",
        manifest.get("vision_records_sha256"),
        sha256(("\n".join(records) + "\n").encode()),
    )

    # The two manifests have to agree about how many vision tensors exist.
    recorded = index.get("classifications", {}).get("vision", {}).get("count")
    expect("vision count against the index manifest", manifest.get("vision_tensor_count"), recorded)
    return failures


def verify_offline() -> int:
    missing = [path for path in (CONFIG_PATH, INDEX_MANIFEST_PATH, HEADER_MANIFEST_PATH) if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing fixture {path.relative_to(ROOT)}")
        print(f"run {Path(__file__).name} --refresh to build them from the pinned revision")
        return 1

    config_bytes = CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes)
    index = json.loads(INDEX_MANIFEST_PATH.read_bytes())
    header = json.loads(HEADER_MANIFEST_PATH.read_bytes())

    failures = check_index_manifest(config_bytes, config, index)
    failures += check_header_manifest(config, index, header)
    for failure in failures:
        print(failure)
    if failures:
        print(
            f"{len(failures)} manifest disagreement(s). Either the derivation changed and the "
            f"fixtures are stale (rerun {Path(__file__).name} --refresh against the pinned "
            "revision), or the derivation is wrong."
        )
        return 1
    vision = index["classifications"]["vision"]["count"]
    print(
        f"ok {REPO}@{REVISION[:12]}: {index['tensor_count']} tensors over "
        f"{index['shard_count']} shards, {vision} of them vision, "
        f"{header['vision_payload_bytes']} vision payload bytes; no network, no weight bytes read"
    )
    return 0


# ── the NETWORK path, which rebuilds the fixtures ─────────────────────────────


def build_manifests() -> tuple[bytes, bytes, bytes]:
    config_bytes, _ = get("config.json")
    index_bytes, _ = get("model.safetensors.index.json")
    config = json.loads(config_bytes)
    index = json.loads(index_bytes)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise SystemExit("index: weight_map is not an object")

    derived = official_names(config)
    present = sorted(weight_map)
    missing = sorted(set(derived) - set(present))
    unexplained = sorted(set(present) - set(derived))
    if missing or unexplained:
        raise SystemExit(
            f"index map differs: {len(missing)} missing, {len(unexplained)} unexplained; "
            f"first missing={missing[:1]}, first unexplained={unexplained[:1]}"
        )

    shards = sorted(set(weight_map.values()))
    classes = {
        category: [name for name in present if classify(name) == category]
        for category in ("language", "mtp", "vision")
    }
    if sum(len(names) for names in classes.values()) != len(present):
        raise SystemExit("index classifier did not account for every tensor exactly once")
    index_manifest = {
        "repo": REPO,
        "revision": REVISION,
        "config_sha256": sha256(config_bytes),
        "index_file": "model.safetensors.index.json",
        "index_sha256": sha256(index_bytes),
        "total_size": int(index["metadata"]["total_size"]),
        "tensor_count": len(present),
        "shard_count": len(shards),
        "all_names": summary(present),
        "classifications": {category: summary(names) for category, names in classes.items()},
    }

    prefix, content_range = get(SHARD, (0, 7))
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length <= 0 or header_length > 64 * 1024 * 1024:
        raise SystemExit(f"{SHARD}: implausible safetensors header length {header_length}")
    header_bytes, second_range = get(SHARD, (8, 7 + header_length))
    header = json.loads(header_bytes)
    header.pop("__metadata__", None)
    total_match = re.fullmatch(r"bytes \d+-\d+/(\d+)", str(content_range))
    if total_match is None:
        raise SystemExit(f"{SHARD}: malformed Content-Range {content_range!r}")
    file_size = int(total_match.group(1))
    if second_range != f"bytes 8-{7 + header_length}/{file_size}":
        raise SystemExit(f"{SHARD}: inconsistent header Content-Range {second_range!r}")

    header_names = sorted(header)
    expected_header = sorted(classes["vision"] + ["embed.weight"])
    if header_names != expected_header:
        missing = sorted(set(expected_header) - set(header_names))
        unexplained = sorted(set(header_names) - set(expected_header))
        raise SystemExit(
            f"{SHARD} header differs: {len(missing)} missing, {len(unexplained)} unexplained; "
            f"first missing={missing[:1]}, first unexplained={unexplained[:1]}"
        )

    tensors: dict[str, Any] = {}
    vision_payload_bytes = 0
    records: list[str] = []
    for name in header_names:
        info = header[name]
        dtype = str(info["dtype"])
        shape = [int(value) for value in info["shape"]]
        offsets = [int(value) for value in info["data_offsets"]]
        if len(offsets) != 2 or offsets[0] < 0 or offsets[1] < offsets[0]:
            raise SystemExit(f"{name}: invalid data_offsets {offsets}")
        expected_bytes = 2
        for dimension in shape:
            if dimension <= 0:
                raise SystemExit(f"{name}: non-positive shape {shape}")
            expected_bytes *= dimension
        if dtype != "BF16" or offsets[1] - offsets[0] != expected_bytes:
            raise SystemExit(
                f"{name}: expected contiguous BF16, got dtype={dtype}, shape={shape}, offsets={offsets}"
            )
        if name != "embed.weight":
            expected_shape = vision_shape(name, config)
            if shape != expected_shape:
                raise SystemExit(f"{name}: header shape {shape}, expected {expected_shape}")
            vision_payload_bytes += expected_bytes
            records.append(vision_record(name, dtype, shape))
        elif shape != [int(config["vocab_size"]), int(config["hidden_size"])]:
            raise SystemExit(f"embed.weight: header shape {shape} disagrees with config")
        if weight_map.get(name) != SHARD:
            raise SystemExit(f"index maps {name} to {weight_map.get(name)!r}, expected {SHARD}")
        tensors[name] = {"dtype": dtype, "shape": shape}

    header_manifest = {
        "repo": REPO,
        "revision": REVISION,
        "shard": SHARD,
        "shard_file_size": file_size,
        "header_length": header_length,
        "header_sha256": sha256(header_bytes),
        "header_tensor_count": len(header_names),
        "vision_tensor_count": len(classes["vision"]),
        "vision_payload_bytes": vision_payload_bytes,
        "vision_records_fnv1a64": fnv1a_lines(records),
        "vision_records_sha256": sha256(("\n".join(records) + "\n").encode()),
        "tensors": tensors,
    }
    return config_bytes, json_bytes(index_manifest), json_bytes(header_manifest)


def refresh() -> int:
    config, index_manifest, header_manifest = build_manifests()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for path, payload in (
        (CONFIG_PATH, config),
        (INDEX_MANIFEST_PATH, index_manifest),
        (HEADER_MANIFEST_PATH, header_manifest),
    ):
        changed = not path.exists() or path.read_bytes() != payload
        path.write_bytes(payload)
        print(f"{'wrote' if changed else 'unchanged'} {path.relative_to(ROOT)}")
    # The rebuilt fixtures go straight back through the offline checks, so a
    # refresh that produced something the derivation disagrees with is a failure
    # here rather than on somebody else's machine later.
    return verify_offline()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-read the pinned revision over the network and rewrite the three fixtures",
    )
    args = parser.parse_args()
    return refresh() if args.refresh else verify_offline()


if __name__ == "__main__":
    raise SystemExit(main())
