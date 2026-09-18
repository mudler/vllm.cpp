#!/usr/bin/env python3
"""Validate W7 release bytes after safe extraction, never from a build tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import release_manifest  # noqa: E402


PRIMARY_CUDA_SMS = release_manifest.PRIMARY_CUDA_SMS
AOT_SMS = tuple(sm for sm in PRIMARY_CUDA_SMS if release_manifest.AOT_AVAILABILITY[sm])
REQUIRED_METADATA = {
    "VERSION",
    "THIRD_PARTY_NOTICES",
    "release-manifest.json",
    "sbom.spdx.json",
}
VERSION_FIELDS = {
    "version",
    "commit",
    "artifact_id",
    "backend",
    "host_os",
    "host_arch",
    "host_abi",
    "source_clean",
    "c_abi_version",
}
FORBIDDEN_SUFFIXES = {
    ".a", ".c", ".cc", ".cmake", ".cpp", ".cu", ".h", ".hpp", ".o", ".obj", ".py", ".pyc"
}
SECRET_NAMES = {
    ".env", "credentials", "id_dsa", "id_ed25519", "id_rsa", "known_hosts"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=release_manifest._reject_duplicate_keys)
    except (OSError, json.JSONDecodeError, release_manifest.ManifestError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def validate_checksum(archive: Path, checksum_path: Path) -> tuple[str, list[str]]:
    actual = sha256(archive)
    errors: list[str] = []
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return actual, [f"checksum sidecar cannot be read: {exc}"]
    expected_line = f"{actual}  {archive.name}"
    if lines != [expected_line]:
        errors.append("checksum sidecar must contain the exact final archive SHA256 and filename")
    return actual, errors


def validate_provenance(
    path: Path, archive: Path, digest: str, manifest: dict[str, Any] | None = None
) -> list[str]:
    try:
        statement = load_json(path, "provenance sidecar")
    except ValueError as exc:
        return [str(exc)]
    errors: list[str] = []
    if statement.get("_type") != "https://in-toto.io/Statement/v1":
        errors.append("provenance _type must be the in-toto Statement v1 URI")
    if statement.get("predicateType") != "https://slsa.dev/provenance/v1":
        errors.append("provenance predicateType must be SLSA provenance v1")
    subject = statement.get("subject")
    expected_subject = [{"name": archive.name, "digest": {"sha256": digest}}]
    if subject != expected_subject:
        errors.append("provenance subject must name and digest the exact final archive")
    if manifest is not None:
        parameters = (
            statement.get("predicate", {})
            .get("buildDefinition", {})
            .get("externalParameters", {})
        )
        expected = {
            "artifact_id": manifest.get("artifact", {}).get("id"),
            "source_commit": manifest.get("build", {}).get("source_commit"),
        }
        if parameters != expected:
            errors.append("provenance build parameters disagree with the release manifest")
    return errors


def safe_extract(archive: Path, destination: Path, archive_format: str) -> list[str]:
    suffix = ".zip" if archive_format == "zip" else ".tar.gz"
    if archive_format not in {"tar.gz", "zip"} or not archive.name.endswith(suffix):
        return ["archive name does not match the explicit archive format"]
    if archive_format == "zip":
        return safe_extract_zip(archive, destination)
    errors: list[str] = []
    try:
        with tarfile.open(archive, "r:*") as bundle:
            members = bundle.getmembers()
            for member in members:
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or member.name in {"", "."}:
                    errors.append(f"unsafe archive path: {member.name!r}")
                    continue
                if member.islnk():
                    errors.append(f"hard links are not permitted in release archives: {member.name}")
                    continue
                if member.issym():
                    target = PurePosixPath(member.linkname)
                    if target.is_absolute() or ".." in target.parts or name.parts[0] != "lib":
                        errors.append(f"unsafe archive symlink: {member.name} -> {member.linkname}")
                        continue
                if not (member.isdir() or member.isfile() or member.issym()):
                    errors.append(f"unsupported archive member type: {member.name}")
            if errors:
                return errors
            bundle.extractall(destination, filter="data")
    except (OSError, tarfile.TarError) as exc:
        return [f"archive cannot be extracted: {exc}"]
    return errors


def unsafe_zip_name(name: str) -> bool:
    if not name or name == "." or "\\" in name or re.match(r"^[A-Za-z]:", name):
        return True
    path = PurePosixPath(name)
    parts = name.rstrip("/").split("/")
    if path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        return True
    reserved = re.compile(r"(?i)^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$")
    return any(
        ":" in part or part.endswith((".", " ")) or reserved.fullmatch(part)
        for part in parts
    )


def zip_windows_key(name: str) -> str:
    return "/".join(part.casefold() for part in name.rstrip("/").split("/"))


def safe_extract_zip(archive: Path, destination: Path) -> list[str]:
    errors: list[str] = []
    try:
        with zipfile.ZipFile(archive) as bundle:
            seen: set[str] = set()
            for info in bundle.infolist():
                if unsafe_zip_name(info.filename):
                    errors.append(f"unsafe archive path: {info.filename!r}")
                    continue
                normalized = zip_windows_key(info.filename)
                if normalized in seen:
                    errors.append(f"duplicate archive path: {info.filename!r}")
                    continue
                seen.add(normalized)
                unix_mode = info.external_attr >> 16
                member_type = stat.S_IFMT(unix_mode)
                if (stat.S_ISLNK(unix_mode) or info.external_attr & 0x400 or
                        member_type not in {0, stat.S_IFREG, stat.S_IFDIR}):
                    errors.append(f"archive links/reparse points are forbidden: {info.filename}")
                    continue
                is_directory = info.is_dir()
                if member_type and (is_directory != stat.S_ISDIR(unix_mode)):
                    errors.append(
                        f"archive member type disagrees with its path marker: {info.filename}"
                    )
            if errors:
                return errors
            destination.mkdir(parents=True, exist_ok=True)
            for info in bundle.infolist():
                relative = PurePosixPath(info.filename.rstrip("/"))
                target = destination.joinpath(*relative.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(info) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    mode = info.external_attr >> 16
                    if mode:
                        target.chmod(stat.S_IMODE(mode))
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        return [f"archive cannot be extracted: {exc}"]
    return errors


def allowed_file(relative: str) -> bool:
    if relative in REQUIRED_METADATA or relative in {"bin/vllm-server", "bin/vllm-server.exe"}:
        return True
    return relative.startswith("lib/") or relative.startswith("share/licenses/")


def scan_file(path: Path, needles: list[bytes]) -> tuple[set[bytes], bool]:
    """Scan large release binaries without reading them wholly into memory."""
    found: set[bytes] = set()
    aws_key = False
    overlap = max([24, *(len(needle) for needle in needles)], default=24) - 1
    tail = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            data = tail + chunk
            found.update(needle for needle in needles if needle and needle in data)
            aws_key = aws_key or re.search(rb"AKIA[0-9A-Z]{16}", data) is not None
            tail = data[-overlap:] if overlap else b""
    return found, aws_key


def validate_contents(root: Path, forbidden_paths: list[str]) -> list[str]:
    errors: list[str] = []
    files = {path.relative_to(root).as_posix(): path for path in root.rglob("*") if not path.is_dir()}
    missing = sorted(REQUIRED_METADATA - files.keys())
    if missing:
        errors.append(f"archive is missing required files: {missing}")
    servers = {"bin/vllm-server", "bin/vllm-server.exe"} & files.keys()
    if len(servers) != 1:
        errors.append("archive must contain exactly one platform server executable")
    for relative, path in sorted(files.items()):
        if not allowed_file(relative):
            errors.append(f"archive contains undeclared path: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"archive contains source/object file: {relative}")
        if path.name.lower() in SECRET_NAMES or path.suffix.lower() in {".key", ".pem"}:
            errors.append(f"archive contains credential-like path: {relative}")
        if path.is_symlink():
            continue
        encoded_paths = [os.fsencode(forbidden) for forbidden in forbidden_paths if forbidden]
        found_paths, aws_key = scan_file(path, encoded_paths)
        for forbidden in forbidden_paths:
            if forbidden and os.fsencode(forbidden) in found_paths:
                errors.append(f"archive file {relative} embeds forbidden build path {forbidden!r}")
        if aws_key:
            errors.append(f"archive file {relative} embeds an AWS access-key-shaped credential")
    server = root / "bin/vllm-server"
    if server.exists() and not os.access(server, os.X_OK):
        errors.append("bin/vllm-server is not executable")
    license_files = [path for path in files if path.startswith("share/licenses/")]
    if not license_files:
        errors.append("archive has no license files under share/licenses/")
    return errors


def validate_windows_layout(root: Path) -> list[str]:
    errors: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root).as_posix().casefold().startswith("lib/"):
            errors.append(f"Windows release archive contains forbidden library payload: {path.relative_to(root)}")
    return errors


def parse_version(path: Path) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return values, [f"VERSION cannot be read: {exc}"]
    for number, line in enumerate(lines, 1):
        if not line or "=" not in line:
            errors.append(f"VERSION line {number} must be non-empty key=value")
            continue
        key, value = line.split("=", 1)
        if key in values:
            errors.append(f"VERSION duplicates field {key!r}")
        values[key] = value
    if set(values) != VERSION_FIELDS:
        errors.append(f"VERSION fields {sorted(values)} != required {sorted(VERSION_FIELDS)}")
    if values.get("source_clean") not in {"true", "false"}:
        errors.append("VERSION source_clean must be true or false")
    if not values.get("c_abi_version", "").isdigit():
        errors.append("VERSION c_abi_version must be a decimal integer")
    return values, errors


def validate_version(values: dict[str, str], manifest: dict[str, Any]) -> list[str]:
    artifact = manifest.get("artifact", {})
    build = manifest.get("build", {})
    backend = manifest.get("backend", {})
    host = manifest.get("host", {})
    expected = {
        "version": artifact.get("version"),
        "commit": build.get("source_commit"),
        "artifact_id": artifact.get("id"),
        "backend": backend.get("name"),
        "host_os": host.get("os"),
        "host_arch": host.get("arch"),
        "host_abi": host.get("abi"),
        "source_clean": "true" if build.get("source_clean") is True else "false",
        "c_abi_version": str(artifact.get("c_abi_version", "")),
    }
    return [
        f"VERSION {key}={values.get(key)!r} disagrees with manifest value {value!r}"
        for key, value in expected.items()
        if values.get(key) != value
    ]


def validate_sbom(path: Path, root: Path, manifest: dict[str, Any]) -> list[str]:
    try:
        sbom = load_json(path, "SPDX SBOM")
    except ValueError as exc:
        return [str(exc)]
    errors: list[str] = []
    if sbom.get("spdxVersion") != "SPDX-2.3":
        errors.append("SBOM spdxVersion must be SPDX-2.3")
    if sbom.get("SPDXID") != "SPDXRef-DOCUMENT":
        errors.append("SBOM document SPDXID must be SPDXRef-DOCUMENT")
    if sbom.get("dataLicense") != "CC0-1.0":
        errors.append("SBOM dataLicense must be CC0-1.0")
    if sbom.get("name") != manifest.get("artifact", {}).get("id"):
        errors.append("SBOM name disagrees with manifest artifact id")
    files = sbom.get("files")
    if not isinstance(files, list):
        return errors + ["SBOM files must be an array"]
    by_name = {item.get("fileName"): item for item in files if isinstance(item, dict)}
    server_name = "vllm-server.exe" if manifest.get("host", {}).get("os") == "windows" else "vllm-server"
    shipped = [root / "bin" / server_name]
    lib_dir = root / "lib"
    if lib_dir.is_dir():
        shipped.extend(path for path in lib_dir.rglob("*") if path.is_file() and not path.is_symlink())
    for shipped_path in shipped:
        relative = "./" + shipped_path.relative_to(root).as_posix()
        item = by_name.get(relative)
        if not isinstance(item, dict):
            errors.append(f"SBOM must inventory {relative}")
            continue
        recorded = {
            checksum.get("checksumValue")
            for checksum in item.get("checksums", [])
            if isinstance(checksum, dict) and checksum.get("algorithm") == "SHA256"
        }
        if recorded != {sha256(shipped_path)}:
            errors.append(f"SBOM SHA256 does not match extracted bytes for {relative}")
    return errors


def validate_archive_name(
    archive: Path, manifest: dict[str, Any], archive_format: str
) -> list[str]:
    artifact = manifest.get("artifact", {})
    expected = f"vllm.cpp-{artifact.get('version')}-{artifact.get('id')}.{archive_format}"
    if archive.name != expected:
        return [f"canonical archive name must be {expected!r}"]
    return []


def parse_elf_needed(dynamic_output: str) -> list[str]:
    return re.findall(r"\(NEEDED\).*?\[([^]]+)\]", dynamic_output)


def parse_elf_rpaths(dynamic_output: str) -> list[str]:
    values = re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^]]*)\]", dynamic_output)
    return [entry for value in values for entry in value.split(":") if entry]


def parse_elf_interpreter(program_output: str) -> str:
    match = re.search(r"Requesting program interpreter:\s*([^]]+)\]", program_output)
    return match.group(1) if match else ""


def validate_linux_dynamic(
    manifest: dict[str, Any],
    needed: list[str],
    rpaths: list[str],
    interpreter: str,
    forbidden_paths: list[str],
) -> list[str]:
    errors: list[str] = []
    declared = {
        item.get("name")
        for item in manifest.get("dependencies", [])
        if (
            isinstance(item, dict)
            and item.get("linkage") == "dynamic"
            and not str(item.get("name", "")).endswith(".metallib")
        )
    }
    literal_static = manifest.get("artifact", {}).get("static_boundary") == "literal-static"
    if literal_static and (needed or rpaths or interpreter):
        errors.append("literal-static artifact has an ELF interpreter, dependency, or RPATH")
    for dependency in needed:
        if dependency not in declared:
            errors.append(f"undeclared ELF dependency: {dependency}")
    for dependency in sorted(declared - set(needed)):
        errors.append(f"declared dynamic dependency is not linked: {dependency}")
    for entry in rpaths:
        if entry.startswith("/"):
            errors.append(f"absolute ELF RPATH/RUNPATH is forbidden: {entry}")
        if any(path and path in entry for path in forbidden_paths):
            errors.append(f"ELF RPATH/RUNPATH contains a forbidden build path: {entry}")
        if not (entry.startswith("$ORIGIN") or entry.startswith("${ORIGIN}")):
            errors.append(f"ELF RPATH/RUNPATH must be relative to the extracted bundle: {entry}")
    host = manifest.get("host", {})
    if not literal_static and host.get("abi") == "glibc" and interpreter and "ld-linux" not in interpreter:
        errors.append(f"glibc artifact has unexpected ELF interpreter: {interpreter}")
    if not literal_static and host.get("abi") == "musl" and "ld-musl" not in interpreter:
        errors.append(f"musl artifact has unexpected ELF interpreter: {interpreter!r}")
    return errors


def macho_dependency_name(install_name: str) -> str:
    for component in install_name.split("/"):
        if component.endswith(".framework"):
            return component
    return Path(install_name).name


def validate_macho_dynamic(
    manifest: dict[str, Any],
    dependencies: list[str],
    rpaths: list[str],
    forbidden_paths: list[str],
) -> list[str]:
    errors: list[str] = []
    declared_dynamic = {
        item.get("name")
        for item in manifest.get("dependencies", [])
        if (
            isinstance(item, dict)
            and item.get("linkage") == "dynamic"
            and not str(item.get("name", "")).endswith(".metallib")
        )
    }
    declared_external = {
        item.get("name")
        for item in manifest.get("dependencies", [])
        if isinstance(item, dict) and item.get("linkage") == "external"
    }
    actual_dynamic: set[str] = set()
    actual_external: set[str] = set()
    for install_name in dependencies:
        name = macho_dependency_name(install_name)
        if name.endswith(".framework"):
            actual_external.add(name)
        else:
            actual_dynamic.add(name)
        allowed = install_name.startswith(("/usr/lib/", "/System/Library/", "@rpath/", "@loader_path/"))
        if not allowed:
            errors.append(f"forbidden Mach-O install name: {install_name}")
        if any(path and path in install_name for path in forbidden_paths):
            errors.append(f"Mach-O install name contains forbidden build path: {install_name}")
    for name in sorted(actual_dynamic - declared_dynamic):
        errors.append(f"undeclared Mach-O dependency: {name}")
    for name in sorted(declared_dynamic - actual_dynamic):
        errors.append(f"declared dynamic dependency is not linked: {name}")
    for name in sorted(actual_external - declared_external):
        errors.append(f"undeclared Mach-O framework dependency: {name}")
    for name in sorted(declared_external - actual_external):
        errors.append(f"declared external framework is not linked: {name}")
    for entry in rpaths:
        if not entry.startswith("@loader_path/"):
            errors.append(f"Mach-O RPATH must be relative to the extracted bundle: {entry}")
        if any(path and path in entry for path in forbidden_paths):
            errors.append(f"Mach-O RPATH contains a forbidden build path: {entry}")
    return errors


def parse_otool_dependencies(output: str) -> list[str]:
    lines = output.splitlines()[1:]
    return [line.strip().split(" (", 1)[0] for line in lines if line.strip()]


def parse_otool_rpaths(output: str) -> list[str]:
    lines = output.splitlines()
    rpaths: list[str] = []
    for index, line in enumerate(lines):
        if line.strip() == "cmd LC_RPATH":
            for candidate in lines[index + 1:index + 5]:
                match = re.match(r"\s*path\s+(\S+)\s+\(offset", candidate)
                if match:
                    rpaths.append(match.group(1))
                    break
    return rpaths


def validate_macho_install_id(install_id: str, forbidden_paths: list[str]) -> list[str]:
    if not install_id.startswith(("@rpath/", "@loader_path/")):
        return [f"bundled Mach-O install ID must be relative: {install_id}"]
    if any(path and path in install_id for path in forbidden_paths):
        return [f"bundled Mach-O install ID contains a forbidden build path: {install_id}"]
    return []


WINDOWS_SYSTEM_DLLS = {
    "ADVAPI32.DLL", "BCRYPT.DLL", "COMDLG32.DLL", "CRYPT32.DLL", "DNSAPI.DLL",
    "GDI32.DLL", "IPHLPAPI.DLL", "KERNEL32.DLL", "NORMALIZ.DLL", "NTDLL.DLL",
    "OLE32.DLL", "OLEAUT32.DLL", "PSAPI.DLL", "RPCRT4.DLL", "SECUR32.DLL",
    "SETUPAPI.DLL", "SHELL32.DLL", "SHLWAPI.DLL", "USER32.DLL", "VERSION.DLL",
    "WINMM.DLL", "WS2_32.DLL",
}


def validate_pe_audit(
    manifest: dict[str, Any],
    machine: str,
    imports: list[str],
    debug_paths: list[str],
    forbidden_paths: list[str],
) -> list[str]:
    errors: list[str] = []
    if machine.upper() not in {"8664", "AMD64", "X64"}:
        errors.append(f"PE executable must target AMD64, got {machine!r}")
    actual = {name.upper() for name in imports}
    declared = {
        str(item.get("name", "")).upper()
        for item in manifest.get("dependencies", [])
        if isinstance(item, dict) and item.get("linkage") == "dynamic"
    }
    for name in sorted(actual - declared):
        errors.append(f"undeclared PE import: {name}")
    for name in sorted(declared - actual):
        errors.append(f"declared dynamic dependency is not imported: {name}")
    for name in sorted(actual):
        if name.startswith("API-MS-WIN-CRT-"):
            errors.append(f"dynamic CRT import violates the /MT static-CRT contract: {name}")
            continue
        if name.startswith(("API-MS-WIN-", "EXT-MS-WIN-")):
            continue
        if name.startswith(("MSYS-", "MINGW")) or name in {"LIBGCC_S_SEH-1.DLL", "LIBSTDC++-6.DLL"}:
            errors.append(f"MinGW/MSYS runtime import is forbidden: {name}")
        if re.search(r"(?:VCRUNTIME|MSVCP|MSVCR|UCRTBASE|CONCRT).*D\.DLL$", name):
            errors.append(f"debug CRT import is forbidden: {name}")
        elif re.search(r"(?:VCRUNTIME|MSVCP|MSVCR|UCRTBASE|CONCRT).*\.DLL$", name):
            errors.append(f"dynamic CRT import violates the /MT static-CRT contract: {name}")
        elif name not in WINDOWS_SYSTEM_DLLS and name not in declared:
            errors.append(f"non-system PE import is forbidden: {name}")
    for value in debug_paths:
        if re.match(r"^[A-Za-z]:[\\/]", value):
            errors.append(f"PE debug path contains a developer drive path: {value}")
        if any(path and path.lower() in value.lower() for path in forbidden_paths):
            errors.append(f"PE debug/build path contains a forbidden build path: {value}")
    return errors


def parse_dumpbin_machine(output: str) -> str:
    match = re.search(r"\b(8664|14C)\s+machine\b", output, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"machine\s*\(([^)]+)\)", output, re.IGNORECASE)
    return match.group(1) if match else ""


def parse_dumpbin_imports(output: str) -> list[str]:
    return sorted(set(re.findall(r"(?im)^\s*([A-Za-z0-9_.+-]+\.dll)\s*$", output)))


def parse_pe_debug_paths(output: str) -> list[str]:
    return sorted(set(re.findall(r"(?i)[A-Za-z]:[\\/][^\r\n\x00]*?\.pdb", output)))


def validate_cuda_inventory(
    manifest: dict[str, Any], images: list[str], symbols: list[str]
) -> list[str]:
    if manifest.get("backend", {}).get("name") != "cuda":
        return []
    errors: list[str] = []
    image_sms = {match.group(1) for image in images if (match := re.search(r"sm_([0-9]+a?)", image))}
    for sm in PRIMARY_CUDA_SMS:
        if sm not in image_sms:
            errors.append(f"CUDA archive is missing sm_{sm} device code")
    for sm in sorted(image_sms - set(PRIMARY_CUDA_SMS)):
        errors.append(f"CUDA archive contains undeclared sm_{sm} device code")
    symbol_text = "\n".join(symbols)
    for sm in AOT_SMS:
        if f"vt_aot_sm_{sm}_" not in symbol_text:
            errors.append(f"CUDA archive is missing exact AOT namespace for sm_{sm}")
    for sm in set(PRIMARY_CUDA_SMS) - set(AOT_SMS):
        if f"vt_aot_sm_{sm}_" in symbol_text:
            errors.append(f"CUDA archive fabricates unavailable AOT namespace for sm_{sm}")
    return errors


def run(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout + result.stderr


def ldd_reports_static(output: str) -> bool:
    normalized = output.lower()
    return any(
        marker in normalized
        for marker in (
            "not a dynamic executable",
            "not a valid dynamic program",
            "statically linked",
        )
    )


def inspect_linux(
    server: Path,
    manifest: dict[str, Any],
    forbidden_paths: list[str],
    skip_version_smoke: bool,
) -> list[str]:
    errors: list[str] = []
    for tool in ("file", "readelf", "ldd"):
        if shutil.which(tool) is None:
            errors.append(f"required Linux archive inspector is unavailable: {tool}")
    if errors:
        return errors
    file_rc, file_output = run(["file", "-b", str(server)])
    if file_rc != 0 or "ELF" not in file_output:
        errors.append(f"server is not an inspectable ELF executable: {file_output.strip()}")
    expected_machine = "x86-64" if manifest.get("host", {}).get("arch") == "x86_64" else "aarch64"
    if expected_machine.lower() not in file_output.lower():
        errors.append(f"ELF host architecture does not match manifest: {file_output.strip()}")
    dynamic_rc, dynamic = run(["readelf", "-dW", str(server)])
    program_rc, program = run(["readelf", "-lW", str(server)])
    if dynamic_rc != 0 or program_rc != 0:
        errors.append("readelf could not inspect the extracted server")
    else:
        errors.extend(
            validate_linux_dynamic(
                manifest,
                parse_elf_needed(dynamic),
                parse_elf_rpaths(dynamic),
                parse_elf_interpreter(program),
                forbidden_paths,
            )
        )
    ldd_rc, ldd_output = run(["ldd", str(server)])
    literal_static = manifest.get("artifact", {}).get("static_boundary") == "literal-static"
    static_ldd = ldd_reports_static(ldd_output)
    if literal_static:
        if not static_ldd:
            errors.append(f"literal-static server has a dynamic ldd result: {ldd_output.strip()}")
    elif ldd_rc != 0 or "not found" in ldd_output:
        errors.append(f"ldd found a missing dependency: {ldd_output.strip()}")
    for forbidden in forbidden_paths:
        if forbidden and forbidden in ldd_output:
            errors.append(f"ldd resolves a dependency through forbidden path {forbidden!r}")
    help_rc, help_output = run([str(server), "--help"])
    if help_rc != 0 or "usage" not in help_output.lower():
        errors.append("extracted vllm-server --help smoke failed")
    if not skip_version_smoke:
        version_rc, version_output = run([str(server), "--version"])
        expected_version = manifest.get("artifact", {}).get("version", "")
        version_record = parse_version(server.parent.parent / "VERSION")[0]
        expected_abi = version_record.get("c_abi_version", "")
        if (
            version_rc != 0
            or f"vllm.cpp {expected_version}" not in version_output
            or f"c-abi={expected_abi}" not in version_output
        ):
            errors.append("extracted vllm-server --version disagrees with VERSION/manifest")
    if manifest.get("backend", {}).get("name") == "cuda":
        if shutil.which("cuobjdump") is None or shutil.which("nm") is None:
            errors.append("CUDA archive validation requires cuobjdump and nm")
        else:
            cuda_rc, cuda_output = run(["cuobjdump", "--list-elf", str(server)])
            nm_rc, nm_output = run(["nm", "-a", str(server)])
            if cuda_rc != 0 or nm_rc != 0:
                errors.append("CUDA archive inspectors could not read the extracted server")
            else:
                errors.extend(validate_cuda_inventory(manifest, cuda_output.splitlines(), nm_output.splitlines()))
    return errors


def inspect_macos(
    root: Path,
    server: Path,
    manifest: dict[str, Any],
    forbidden_paths: list[str],
    skip_version_smoke: bool,
) -> list[str]:
    errors: list[str] = []
    for tool in ("file", "otool"):
        if shutil.which(tool) is None:
            errors.append(f"required macOS archive inspector is unavailable: {tool}")
    if errors:
        return errors
    file_rc, file_output = run(["file", "-b", str(server)])
    if file_rc != 0 or "Mach-O" not in file_output or "arm64" not in file_output:
        errors.append(f"server is not a Mach-O arm64 executable: {file_output.strip()}")
    binaries = [server]
    if manifest.get("backend", {}).get("name") == "mlx":
        binaries.append(root / "lib/libmlx.dylib")
    dependencies: set[str] = set()
    rpaths: set[str] = set()
    for binary in binaries:
        binary_rc, binary_output = run(["file", "-b", str(binary)])
        deps_rc, deps_output = run(["otool", "-L", str(binary)])
        load_rc, load_output = run(["otool", "-l", str(binary)])
        if binary_rc != 0 or "Mach-O" not in binary_output or "arm64" not in binary_output:
            errors.append(f"bundled executable/library is not Mach-O arm64: {binary}")
        if deps_rc != 0 or load_rc != 0:
            errors.append(f"otool could not inspect extracted Mach-O file: {binary}")
            continue
        dependencies.update(parse_otool_dependencies(deps_output))
        rpaths.update(parse_otool_rpaths(load_output))
        if binary.suffix == ".dylib":
            id_rc, id_output = run(["otool", "-D", str(binary)])
            install_ids = [line.strip() for line in id_output.splitlines()[1:] if line.strip()]
            if id_rc != 0 or len(install_ids) != 1:
                errors.append(f"bundled dylib has no unique install ID: {binary}")
            else:
                errors.extend(validate_macho_install_id(install_ids[0], forbidden_paths))
    errors.extend(
        validate_macho_dynamic(
            manifest,
            sorted(dependencies),
            sorted(rpaths),
            forbidden_paths,
        )
    )
    if manifest.get("backend", {}).get("name") == "mlx":
        for relative in ("lib/libmlx.dylib", "lib/mlx.metallib"):
            if not (root / relative).is_file():
                errors.append(f"MLX archive is missing bundled {relative}")
    help_rc, help_output = run([str(server), "--help"])
    if help_rc != 0 or "usage" not in help_output.lower():
        errors.append("extracted vllm-server --help smoke failed")
    if not skip_version_smoke:
        version_rc, version_output = run([str(server), "--version"])
        expected_version = manifest.get("artifact", {}).get("version", "")
        version_record = parse_version(root / "VERSION")[0]
        expected_abi = version_record.get("c_abi_version", "")
        if (
            version_rc != 0
            or f"vllm.cpp {expected_version}" not in version_output
            or f"c-abi={expected_abi}" not in version_output
        ):
            errors.append("extracted vllm-server --version disagrees with VERSION/manifest")
    return errors


def inspect_windows(
    root: Path,
    server: Path,
    manifest: dict[str, Any],
    forbidden_paths: list[str],
    skip_version_smoke: bool,
) -> list[str]:
    dumpbin = shutil.which("dumpbin") or shutil.which("dumpbin.exe")
    if dumpbin is None:
        return ["required Windows PE inspector is unavailable: dumpbin"]
    headers_rc, headers = run([dumpbin, "/nologo", "/headers", str(server)])
    imports_rc, imports = run([dumpbin, "/nologo", "/dependents", str(server)])
    raw_rc, raw = run([dumpbin, "/nologo", "/rawdata", str(server)])
    if headers_rc != 0 or imports_rc != 0 or raw_rc != 0:
        return ["dumpbin could not inspect the extracted Windows server"]
    errors = validate_pe_audit(
        manifest,
        parse_dumpbin_machine(headers),
        parse_dumpbin_imports(imports),
        parse_pe_debug_paths(headers + "\n" + raw),
        forbidden_paths,
    )
    bundled_dlls = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.dll"))
    if bundled_dlls:
        errors.append(f"Windows archive contains unexpected bundled DLLs: {bundled_dlls}")
    help_rc, help_output = run([str(server), "--help"])
    if help_rc != 0 or "usage" not in help_output.lower():
        errors.append("extracted Windows vllm-server.exe --help smoke failed")
    if not skip_version_smoke:
        version_rc, version_output = run([str(server), "--version"])
        expected_version = manifest.get("artifact", {}).get("version", "")
        expected_abi = parse_version(root / "VERSION")[0].get("c_abi_version", "")
        if (
            version_rc != 0
            or f"vllm.cpp {expected_version}" not in version_output
            or f"c-abi={expected_abi}" not in version_output
        ):
            errors.append("extracted Windows vllm-server.exe --version disagrees with VERSION/manifest")
    return errors


def validate_release(args: argparse.Namespace) -> list[str]:
    archive = args.archive.resolve()
    digest, errors = validate_checksum(archive, args.checksum.resolve())
    errors.extend(validate_provenance(args.provenance.resolve(), archive, digest))
    if errors:
        return errors
    with tempfile.TemporaryDirectory(prefix="vllm-release-validate-") as temporary:
        extracted = Path(temporary) / "extracted"
        extracted.mkdir()
        errors.extend(safe_extract(archive, extracted, args.archive_format))
        if errors:
            return errors
        forbidden = [str(Path(value).resolve()) for value in args.forbid_path]
        errors.extend(validate_contents(extracted, forbidden))
        manifest_path = extracted / "release-manifest.json"
        if not manifest_path.is_file():
            return errors
        try:
            manifest = load_json(manifest_path, "release manifest")
            schema = release_manifest.load_schema(args.schema.resolve())
        except (ValueError, OSError, release_manifest.ManifestError) as exc:
            return errors + [str(exc)]
        errors.extend(release_manifest.validate_manifest(manifest, schema, args.repo_root.resolve()))
        if manifest.get("host", {}).get("os") == "windows":
            errors.extend(validate_windows_layout(extracted))
        errors.extend(validate_archive_name(archive, manifest, args.archive_format))
        values, version_errors = parse_version(extracted / "VERSION")
        errors.extend(version_errors)
        errors.extend(validate_version(values, manifest))
        errors.extend(validate_sbom(extracted / "sbom.spdx.json", extracted, manifest))
        errors.extend(validate_provenance(args.provenance.resolve(), archive, digest, manifest))
        if manifest.get("host", {}).get("os") == "linux":
            errors.extend(
                inspect_linux(
                    extracted / "bin/vllm-server",
                    manifest,
                    forbidden,
                    args.skip_version_smoke,
                )
            )
        elif manifest.get("host", {}).get("os") == "macos":
            errors.extend(
                inspect_macos(
                    extracted,
                    extracted / "bin/vllm-server",
                    manifest,
                    forbidden,
                    args.skip_version_smoke,
                )
            )
        elif manifest.get("host", {}).get("os") == "windows":
            errors.extend(
                inspect_windows(
                    extracted,
                    extracted / "bin/vllm-server.exe",
                    manifest,
                    forbidden,
                    args.skip_version_smoke,
                )
            )
        else:
            errors.append("release archive has an unsupported host OS")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--schema", type=Path, default=SCRIPT_DIR.parent / "release/manifest-v1.schema.json")
    parser.add_argument("--archive-format", choices=("tar.gz", "zip"), required=True)
    parser.add_argument("--forbid-path", action="append", default=[])
    parser.add_argument("--skip-version-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_release(args)
    if errors:
        print("release archive validation FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("release archive validation: extracted bytes, dependencies, metadata, and supply chain OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
