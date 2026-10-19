#!/usr/bin/env bash
# Pack the kit + built datasets for the RTX box:  bash scripts/make_zip.sh [data dir]
set -euo pipefail
KIT=$(cd "$(dirname "$0")/.." && pwd); D=${1:-/scratch.global/radke149/r1x4_runs/data}
cd "$KIT" && diff -ruN -x __pycache__ -x outputs verl_orig_ampo_ro verl > PATCHES.diff || true
STAGE=$(mktemp -d /scratch.global/radke149/exports/.stage_XXXX); mkdir -p "$STAGE/$(basename "$KIT")/data"
rsync -a --exclude __pycache__ --exclude outputs --exclude data/src --exclude '*.egg-info' "$KIT/" "$STAGE/$(basename "$KIT")/"
cp "$D"/*.parquet "$D"/*.manifest.json "$STAGE/$(basename "$KIT")/data/" 2>/dev/null || true
cp -r "$D/ampo" "$STAGE/$(basename "$KIT")/data/"; [ -d "$D/mu" ] && cp -r "$D/mu" "$STAGE/$(basename "$KIT")/data/"
Z=/scratch.global/radke149/exports/$(basename "$KIT")_$(date +%Y-%m-%d_%H%M).zip
(cd "$STAGE" && zip -qr -1 "$Z" "$(basename "$KIT")") && chmod -R u+w "$STAGE" && rm -rf "$STAGE"
ls -la "$Z"; unzip -l "$Z" | tail -1
