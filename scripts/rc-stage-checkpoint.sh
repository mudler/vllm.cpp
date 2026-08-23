#!/usr/bin/env bash
# rc-stage-checkpoint.sh -- copy a checkpoint directory from the NAS to local
# disk ONCE, verify it by manifest, and do nothing on the next run (#1807).
#
# Inside an `rc` lease `/workspace` is the NAS over CIFS. A gate that opens a
# 22 GB or 67 GB checkpoint from there pays CIFS bandwidth on every read, so a
# job copies the checkpoint to local disk first. This script makes that copy
# idempotent and verified:
#
#   * completeness is defined by `SRC/SHA256SUMS` (one `<sha256> <size> <relpath>`
#     line per file), written once by `--make-manifest` from any host that sees
#     the NAS; a directory with no manifest is refused, because `ls` cannot tell
#     a complete copy from one a killed job left behind;
#   * a run that finds `DST/.staged-ok` holding the manifest's own sha256, and
#     every listed file present at its listed size, exits 0 reading only the
#     manifest -- the payload is never re-read, over CIFS or locally;
#   * otherwise every file is hashed locally (local disk, not CIFS), copied
#     through `<relpath>.part` + rename when missing or different, hashed again
#     after the copy, and the marker is written only when every file verified;
#   * a file the manifest does not list is left alone and reported.
#
# Usage:
#   scripts/rc-stage-checkpoint.sh --make-manifest SRC
#   scripts/rc-stage-checkpoint.sh SRC DST
#
# Exit codes: 0 staged (or already staged); 2 usage; 3 manifest absent or
# malformed; 4 a file failed verification after the copy; 5 a source file the
# manifest lists is absent.
set -uo pipefail

usage() {
  echo "usage: $0 --make-manifest SRC | $0 SRC DST" >&2
  exit 2
}

MANIFEST_NAME=SHA256SUMS
MARKER_NAME=.staged-ok

sha_of_file() { sha256sum -- "$1" | cut -d' ' -f1; }
size_of_file() { stat -c %s -- "$1"; }

make_manifest() {
  local src=$1
  [ -d "$src" ] || { echo "rc-stage-checkpoint: SRC '$src' is not a directory" >&2; exit 2; }
  local tmp="$src/$MANIFEST_NAME.part"
  : > "$tmp" || { echo "rc-stage-checkpoint: cannot write $tmp" >&2; exit 2; }
  local n=0
  # Sorted, relative, excluding the manifest itself and the marker. `-print0`
  # keeps a space in a path intact; none is expected, but a silent truncation
  # would be worse than the loop.
  while IFS= read -r -d '' f; do
    local rel=${f#"$src"/}
    case "$rel" in
      "$MANIFEST_NAME"|"$MANIFEST_NAME.part"|"$MARKER_NAME"|*.part) continue ;;
    esac
    printf '%s %s %s\n' "$(sha_of_file "$f")" "$(size_of_file "$f")" "$rel" >> "$tmp"
    n=$((n+1))
  done < <(find "$src" -type f -print0 | sort -z)
  [ "$n" -gt 0 ] || { rm -f "$tmp"; echo "rc-stage-checkpoint: no files under $src" >&2; exit 3; }
  mv -f "$tmp" "$src/$MANIFEST_NAME"
  echo "rc-stage-checkpoint: wrote $src/$MANIFEST_NAME ($n files, manifest sha256 $(sha_of_file "$src/$MANIFEST_NAME"))"
}

stage() {
  local src=$1 dst=$2
  local manifest="$src/$MANIFEST_NAME"
  [ -f "$manifest" ] || {
    echo "rc-stage-checkpoint: REFUSED -- no $MANIFEST_NAME beside $src." >&2
    echo "  A directory without a manifest cannot be told complete. Run:" >&2
    echo "  $0 --make-manifest $src    (once, from any host that sees the NAS)" >&2
    exit 3
  }
  mkdir -p "$dst" || { echo "rc-stage-checkpoint: cannot create $dst" >&2; exit 2; }
  local msha; msha=$(sha_of_file "$manifest")

  # Parse once; refuse a malformed line rather than staging half a checkpoint.
  local -a shas sizes rels
  local line want_sha want_size rel lineno=0
  while IFS= read -r line || [ -n "$line" ]; do
    lineno=$((lineno+1))
    [ -z "$line" ] && continue
    want_sha=${line%% *}; line=${line#* }
    want_size=${line%% *}; rel=${line#* }
    if ! [[ "$want_sha" =~ ^[0-9a-f]{64}$ ]] || ! [[ "$want_size" =~ ^[0-9]+$ ]] || [ -z "$rel" ] || [ "$rel" = "$line" ]; then
      echo "rc-stage-checkpoint: malformed manifest line $lineno in $manifest" >&2
      exit 3
    fi
    # A relpath that is absolute or climbs through `..` would write outside
    # DST; `--make-manifest` never emits one, so it is a hand-edit. Refuse it.
    case "$rel" in
      /*|..|../*|*/..|*/../*)
        echo "rc-stage-checkpoint: malformed manifest line $lineno in $manifest -- relpath '$rel' escapes DST" >&2
        exit 3 ;;
    esac
    shas+=("$want_sha"); sizes+=("$want_size"); rels+=("$rel")
  done < "$manifest"
  [ "${#rels[@]}" -gt 0 ] || { echo "rc-stage-checkpoint: empty manifest $manifest" >&2; exit 3; }

  # Fast path: marker names this manifest and every file is present at size.
  if [ -f "$dst/$MARKER_NAME" ] && [ "$(cat "$dst/$MARKER_NAME")" = "$msha" ]; then
    local ok=1 i
    for i in "${!rels[@]}"; do
      local f="$dst/${rels[$i]}"
      if [ ! -f "$f" ] || [ "$(size_of_file "$f")" != "${sizes[$i]}" ]; then ok=0; break; fi
    done
    if [ "$ok" = 1 ]; then
      echo "rc-stage-checkpoint: ALREADY STAGED $dst (${#rels[@]} files, manifest $msha); nothing read"
      return 0
    fi
    echo "rc-stage-checkpoint: marker present but a file is missing or resized; re-verifying"
  fi
  rm -f "$dst/$MARKER_NAME"

  local copied=0 kept=0 failed=0 i
  for i in "${!rels[@]}"; do
    rel=${rels[$i]}; want_sha=${shas[$i]}; want_size=${sizes[$i]}
    local s="$src/$rel" d="$dst/$rel"
    if [ ! -f "$s" ]; then
      echo "rc-stage-checkpoint: source file listed in manifest is ABSENT: $s" >&2
      exit 5
    fi
    if [ -f "$d" ] && [ "$(size_of_file "$d")" = "$want_size" ] && [ "$(sha_of_file "$d")" = "$want_sha" ]; then
      kept=$((kept+1)); continue
    fi
    mkdir -p "$(dirname "$d")"
    rm -f "$d" "$d.part"
    # `cp` over CIFS reads the source once; the write lands on local disk.
    if ! cp -- "$s" "$d.part"; then
      echo "rc-stage-checkpoint: copy failed: $s" >&2; failed=$((failed+1)); rm -f "$d.part"; continue
    fi
    if [ "$(size_of_file "$d.part")" != "$want_size" ] || [ "$(sha_of_file "$d.part")" != "$want_sha" ]; then
      echo "rc-stage-checkpoint: VERIFY FAILED after copy: $rel (size $(size_of_file "$d.part") want $want_size)" >&2
      rm -f "$d.part"; failed=$((failed+1)); continue
    fi
    mv -f "$d.part" "$d"
    copied=$((copied+1))
    echo "rc-stage-checkpoint: copied+verified $rel ($want_size bytes)"
  done

  # Report, never delete, what the manifest does not know about.
  local extra
  extra=$(cd "$dst" && find . -type f ! -name "$MARKER_NAME" ! -name '*.part' -printf '%P\n' | sort | comm -23 - <(printf '%s\n' "${rels[@]}" | sort))
  # Quoted through sed: an unquoted $extra would word-split a name with a
  # space and glob-expand a name like `*` against the caller's cwd.
  [ -n "$extra" ] && echo "rc-stage-checkpoint: NOTE unlisted local files left alone:" && printf '%s\n' "$extra" | sed 's/^/  /'

  if [ "$failed" -ne 0 ]; then
    echo "rc-stage-checkpoint: NOT STAGED -- $failed file(s) failed verification (copied $copied, kept $kept)" >&2
    exit 4
  fi
  printf '%s\n' "$msha" > "$dst/$MARKER_NAME"
  echo "rc-stage-checkpoint: STAGED $dst (copied $copied, kept $kept, manifest $msha)"
}

case "${1:-}" in
  --make-manifest) [ $# -eq 2 ] || usage; make_manifest "$2" ;;
  -h|--help|"") usage ;;
  *) [ $# -eq 2 ] || usage; stage "$1" "$2" ;;
esac
