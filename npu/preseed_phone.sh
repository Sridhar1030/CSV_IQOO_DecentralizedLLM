#!/usr/bin/env bash
# npu/preseed_phone.sh
#
# Pre-seed every layer shard (+ config.json) for the current dist/ build onto a
# phone's dev.dllm.node app-private storage, ahead of the hub ever assigning it
# a range. Node.kt's fetch() skips any file that already exists on disk with
# non-zero length, so once a file is here the app never downloads it again --
# whatever range the hub hands out later, the phone already has every shard it
# could possibly need.
#
# adb cannot write into an app-private directory directly, so each file goes:
#   adb push <file> /data/local/tmp/<name>
#   adb shell run-as dev.dllm.node cp /data/local/tmp/<name> files/shards/<name>.part
#   adb shell run-as dev.dllm.node mv files/shards/<name>.part files/shards/<name>
#   adb shell rm /data/local/tmp/<name>
# one file at a time, so the /data/local/tmp staging area never holds more than
# one shard (~57 MB) at once, never the full ~1.4 GB set.
#
# Usage:
#   npu/preseed_phone.sh [-s SERIAL]... [-d SRC_DIR] [-k tflite|npz] [-h]
#
#   -s SERIAL   Target this adb serial. Repeatable. Default: every device
#               `adb devices` currently lists.
#   -d SRC_DIR  Directory holding config.json + layer_NN.<kind> files.
#               Default: <repo-root>/dist
#   -k KIND     Shard kind to push: "tflite" (NPU engine, default) or "npz"
#               (CPU engine).
#
# Safe to re-run: a file already on the phone with the same size as the source
# is skipped, so a repeat run is a fast no-op. Never deletes or touches shards
# of the other kind, or anything else already in files/shards.
set -o pipefail

PKG="dev.dllm.node"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

serials=()
src_dir=""
kind="tflite"

usage() {
  cat <<EOF
Usage: $(basename "$0") [-s SERIAL]... [-d SRC_DIR] [-k tflite|npz] [-h]

  -s SERIAL   Target this adb serial (repeatable). Default: every device
              \`adb devices\` lists.
  -d SRC_DIR  Directory with config.json + layer_NN.<kind> files.
              Default: $REPO_ROOT/dist
  -k KIND     "tflite" (default, NPU engine) or "npz" (CPU engine).
  -h          Show this help.

Examples:
  $(basename "$0")                                    # every attached phone, tflite, dist/
  $(basename "$0") -s 10BFAT1U06000Z9                  # just one phone
  $(basename "$0") -s 10BFAT1U06000Z9 -s 10BFAU15A9000XR
  $(basename "$0") -k npz -d /path/to/other_shards
EOF
}

while getopts ":s:d:k:h" opt; do
  case "$opt" in
    s) serials+=("$OPTARG") ;;
    d) src_dir="$OPTARG" ;;
    k) kind="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
    :) echo "Option -$OPTARG requires an argument" >&2; usage; exit 2 ;;
  esac
done

if [ "$kind" != "tflite" ] && [ "$kind" != "npz" ]; then
  echo "ERROR: -k must be 'tflite' or 'npz', got '$kind'" >&2
  exit 2
fi

if [ -z "$src_dir" ]; then
  src_dir="$REPO_ROOT/dist"
fi
if [ ! -d "$src_dir" ]; then
  echo "ERROR: source dir '$src_dir' does not exist." >&2
  exit 1
fi
src_dir="$(cd "$src_dir" && pwd)"

# ---- local file helpers (macOS/BSD stat, with a GNU fallback for portability) ----
local_size() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null; }
human_size() { awk -v b="${1:-0}" 'BEGIN{printf "%.1f MB", b/1048576}'; }

# ---- validate the source dir has what we need, once, before touching any device ----
config_src="$src_dir/config.json"
if [ ! -f "$config_src" ]; then
  echo "ERROR: $config_src not found -- is -d pointing at a shard export?" >&2
  exit 1
fi

layer_files=()
for f in "$src_dir"/layer_[0-9][0-9]."$kind"; do
  [ -e "$f" ] && layer_files+=("$f")
done
if [ ${#layer_files[@]} -eq 0 ]; then
  echo "ERROR: no layer_NN.$kind files found in $src_dir" >&2
  echo "       (looked for $src_dir/layer_00.$kind, layer_01.$kind, ...)" >&2
  exit 1
fi
layer_files=($(printf '%s\n' "${layer_files[@]}" | sort))

total_src_bytes=$(local_size "$config_src")
for f in "${layer_files[@]}"; do
  total_src_bytes=$((total_src_bytes + $(local_size "$f")))
done

echo "preseed_phone.sh"
echo "  source dir : $src_dir"
echo "  shard kind : $kind"
echo "  files      : config.json + ${#layer_files[@]} layer_NN.$kind files ($(human_size "$total_src_bytes") total)"

# ---- figure out which devices to target ----
ADB_DEVICES_OUTPUT="$(adb devices)"

state_of() {
  local serial="$1" line
  line="$(printf '%s\n' "$ADB_DEVICES_OUTPUT" | grep -E "^${serial}[[:space:]]")"
  [ -n "$line" ] || return 1
  printf '%s\n' "$line" | awk '{ $1=""; sub(/^[ \t]+/,""); print }'
}

if [ ${#serials[@]} -eq 0 ]; then
  while read -r serial rest; do
    [ -n "$serial" ] || continue
    [ "$serial" = "List" ] && continue
    case "$serial" in \**) continue ;; esac
    serials+=("$serial")
  done <<< "$ADB_DEVICES_OUTPUT"
fi

if [ ${#serials[@]} -eq 0 ]; then
  echo "ERROR: no devices found by \`adb devices\`. Plug a phone in and enable USB debugging." >&2
  exit 1
fi

echo "  targets    : ${serials[*]}"
echo

# ---- push one file, skipping it if the phone already has it at the right size ----
cur_serial=""
cur_stage=""
cleanup_stage() {
  if [ -n "$cur_stage" ] && [ -n "$cur_serial" ]; then
    adb -s "$cur_serial" shell rm -f "$cur_stage" >/dev/null 2>&1
  fi
}
trap cleanup_stage EXIT INT TERM

push_one() {
  local serial="$1" local_path="$2" remote_name="$3"
  local lsize rsize cp_out mv_out new_rsize remote_path stage_path tmp_path

  lsize="$(local_size "$local_path")"
  if [ -z "$lsize" ]; then
    echo "  ERROR: cannot stat local file $local_path" >&2
    return 1
  fi

  remote_path="files/shards/$remote_name"
  rsize="$(adb -s "$serial" shell run-as "$PKG" stat -c%s "$remote_path" 2>/dev/null | tr -d '\r\n')"
  if [ -n "$rsize" ] && [ "$rsize" -eq "$lsize" ] 2>/dev/null; then
    printf '  [skip] %-24s already present (%s)\n' "$remote_name" "$(human_size "$lsize")"
    already_count=$((already_count + 1))
    already_bytes=$((already_bytes + lsize))
    return 0
  fi

  printf '  [push] %-24s %s\n' "$remote_name" "$(human_size "$lsize")"

  tmp_path="/data/local/tmp/preseed.$remote_name"
  cur_serial="$serial"; cur_stage="$tmp_path"
  adb -s "$serial" shell rm -f "$tmp_path" >/dev/null 2>&1

  if ! adb -s "$serial" push "$local_path" "$tmp_path"; then
    echo "  ERROR: adb push failed for $remote_name" >&2
    adb -s "$serial" shell rm -f "$tmp_path" >/dev/null 2>&1
    cur_stage=""
    return 1
  fi

  stage_path="${remote_path}.part"
  if ! cp_out="$(adb -s "$serial" shell run-as "$PKG" cp "$tmp_path" "$stage_path" 2>&1)"; then
    echo "  ERROR: run-as cp failed for $remote_name: $cp_out" >&2
    adb -s "$serial" shell rm -f "$tmp_path" >/dev/null 2>&1
    cur_stage=""
    return 1
  fi

  adb -s "$serial" shell rm -f "$tmp_path" >/dev/null 2>&1
  cur_stage=""

  if ! mv_out="$(adb -s "$serial" shell run-as "$PKG" mv "$stage_path" "$remote_path" 2>&1)"; then
    echo "  ERROR: could not finalize $remote_name on device: $mv_out" >&2
    return 1
  fi

  new_rsize="$(adb -s "$serial" shell run-as "$PKG" stat -c%s "$remote_path" 2>/dev/null | tr -d '\r\n')"
  if [ "$new_rsize" != "$lsize" ]; then
    echo "  ERROR: $remote_name size mismatch after push (expected $lsize, got ${new_rsize:-nothing})" >&2
    return 1
  fi

  pushed_count=$((pushed_count + 1))
  pushed_bytes=$((pushed_bytes + lsize))
  return 0
}

free_space() {
  adb -s "$1" shell df -h /data 2>/dev/null | tail -n +2 | awk '{print $4" free of "$2" ("$5" used)"}'
}

overall_rc=0
grand_already_count=0; grand_pushed_count=0; grand_pushed_bytes=0
grand_start_ts=$(date +%s)

for serial in "${serials[@]}"; do
  echo "=== $serial ==="

  state="$(state_of "$serial")"
  if [ -z "$state" ]; then
    echo "ERROR [$serial]: not seen by \`adb devices\` -- is it plugged in and did you accept the USB-debugging prompt?" >&2
    overall_rc=1
    echo
    continue
  fi
  if [ "$state" != "device" ]; then
    echo "ERROR [$serial]: adb reports state '$state' (expected 'device') -- if 'unauthorized', accept the prompt on the phone; if 'offline', reconnect the cable." >&2
    overall_rc=1
    echo
    continue
  fi

  if ! pkg_out="$(adb -s "$serial" shell pm list packages "$PKG" 2>&1)"; then
    echo "ERROR [$serial]: could not query installed packages: $pkg_out" >&2
    overall_rc=1
    echo
    continue
  fi
  if ! printf '%s\n' "$pkg_out" | grep -q "^package:${PKG}$"; then
    echo "ERROR [$serial]: $PKG is not installed on this device." >&2
    overall_rc=1
    echo
    continue
  fi

  if ! runas_out="$(adb -s "$serial" shell run-as "$PKG" true 2>&1)"; then
    echo "ERROR [$serial]: \`run-as $PKG\` failed (${runas_out:-no output}) -- is this a debug build?" >&2
    overall_rc=1
    echo
    continue
  fi

  if ! mkdir_out="$(adb -s "$serial" shell run-as "$PKG" mkdir -p files/shards 2>&1)"; then
    echo "ERROR [$serial]: could not create files/shards: $mkdir_out" >&2
    overall_rc=1
    echo
    continue
  fi

  already_count=0; already_bytes=0; pushed_count=0; pushed_bytes=0; dev_failed=0
  dev_start_ts=$(date +%s)

  echo "  free space before: $(free_space "$serial")"

  push_one "$serial" "$config_src" "config.json" || dev_failed=1
  for f in "${layer_files[@]}"; do
    push_one "$serial" "$f" "$(basename "$f")" || dev_failed=1
  done

  dev_elapsed=$(( $(date +%s) - dev_start_ts ))

  echo "  --- $serial summary ---"
  echo "  already present : $already_count files"
  echo "  pushed          : $pushed_count files ($(human_size "$pushed_bytes"))"
  echo "  elapsed         : ${dev_elapsed}s"
  echo "  free space after: $(free_space "$serial")"
  if [ "$dev_failed" -ne 0 ]; then
    echo "  ** one or more files FAILED to preseed correctly on $serial -- see errors above **" >&2
    overall_rc=1
  fi
  echo

  grand_already_count=$((grand_already_count + already_count))
  grand_pushed_count=$((grand_pushed_count + pushed_count))
  grand_pushed_bytes=$((grand_pushed_bytes + pushed_bytes))
done

if [ ${#serials[@]} -gt 1 ]; then
  grand_elapsed=$(( $(date +%s) - grand_start_ts ))
  echo "=== grand total across ${#serials[@]} device(s) ==="
  echo "  already present : $grand_already_count files"
  echo "  pushed          : $grand_pushed_count files ($(human_size "$grand_pushed_bytes"))"
  echo "  elapsed         : ${grand_elapsed}s"
fi

exit $overall_rc
