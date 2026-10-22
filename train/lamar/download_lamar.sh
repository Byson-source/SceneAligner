#!/usr/bin/env bash
# Downloads and extracts the raw LaMAR HGE phone sessions needed by train/lamar/split.yaml.
#
#   bash train/lamar/download_lamar.sh [LAMAR_ROOT] [--all] [--rm-zip]
#
# LAMAR_ROOT (argument or environment variable) defaults to /groups/gai51740/RBT3/umemura/zfloc/lamar. Each
# raw session archive ios_<date>_<time>.zip holds one directory of the same name (sensors.txt, images.txt,
# depths.txt, trajectories.txt, raw_data/{images,depth}), extracted next to the archive. --all downloads every
# session of the HGE raw listing instead of the ones of the split; --rm-zip deletes an archive after extraction.
#
# The benchmark release HGE/sessions/map (floorplan-aligned trajectories.txt and the split sub-sequences in
# raw_data/) is downloaded from BENCH_URL only when it is missing; the URL could not be verified when this script
# was written, so place HGE/sessions/map under LAMAR_ROOT by hand if the download fails. The floorplan hge.png is
# not redistributable and is copied from FLOORPLAN_SRC when that file exists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAMAR_ROOT="${LAMAR_ROOT:-/groups/gai51740/RBT3/umemura/zfloc/lamar}"
RAW_URL="${RAW_URL:-https://cvg-data.inf.ethz.ch/lamar/raw/HGE/sessions}"
METADATA_URL="${METADATA_URL:-https://cvg-data.inf.ethz.ch/lamar/raw/HGE/metadata_phone.json}"
BENCH_URL="${BENCH_URL:-https://cvg-data.inf.ethz.ch/lamar/benchmark/HGE.zip}"
FLOORPLAN_SRC="${FLOORPLAN_SRC:-$SCRIPT_DIR/../../../z-floc/assets/floorplan/hge/hge.png}"

ALL=0
RM_ZIP=0
for arg in "$@"; do
    case "$arg" in
        --all) ALL=1 ;;
        --rm-zip) RM_ZIP=1 ;;
        *) LAMAR_ROOT="$arg" ;;
    esac
done
mkdir -p "$LAMAR_ROOT"

# Raw session names: the split sessions without their _NNN suffix (val shares its raw session with train)
if [ "$ALL" -eq 1 ]; then
    sessions=$(wget -qO- "$RAW_URL/" | grep -o 'ios_[0-9._-]*\.zip' | sed 's/\.zip$//' | sort -u)
else
    sessions=$(grep -o 'ios_[0-9._-]*_[0-9]\{3\}' "$SCRIPT_DIR/split.yaml" | sed 's/_[0-9]\{3\}$//' | sort -u)
fi
echo "raw sessions:"
echo "$sessions" | sed 's/^/  /'

for name in $sessions; do
    if [ -d "$LAMAR_ROOT/$name/raw_data" ]; then
        echo "$name: already extracted"
        continue
    fi
    echo "$name: downloading"
    wget -c -q --show-progress -P "$LAMAR_ROOT" "$RAW_URL/$name.zip"
    echo "$name: extracting"
    unzip -n -q "$LAMAR_ROOT/$name.zip" -d "$LAMAR_ROOT"
    if [ ! -d "$LAMAR_ROOT/$name/raw_data/depth" ]; then
        echo "$name: WARNING no raw_data/depth (see metadata_phone.json for sessions without depth)"
    fi
    if [ "$RM_ZIP" -eq 1 ]; then
        rm -f "$LAMAR_ROOT/$name.zip"
    fi
done

if [ ! -f "$LAMAR_ROOT/metadata_phone.json" ]; then
    wget -q -P "$LAMAR_ROOT" "$METADATA_URL" || echo "WARNING: could not download $METADATA_URL"
fi

# Benchmark release: HGE/sessions/map with the floorplan-aligned trajectories
if [ ! -f "$LAMAR_ROOT/HGE/sessions/map/trajectories.txt" ]; then
    echo "HGE/sessions/map: downloading the benchmark release from $BENCH_URL"
    if wget -c -q --show-progress -P "$LAMAR_ROOT" "$BENCH_URL"; then
        unzip -n -q "$LAMAR_ROOT/$(basename "$BENCH_URL")" -d "$LAMAR_ROOT"
    else
        echo "WARNING: download failed. Obtain the HGE benchmark data from https://lamar.ethz.ch/lamar/ and place"
        echo "         it at $LAMAR_ROOT/HGE/sessions/map (trajectories.txt, sensors.txt, raw_data/ios_*_NNN/)."
    fi
fi

if [ ! -f "$LAMAR_ROOT/hge.png" ]; then
    if [ -f "$FLOORPLAN_SRC" ]; then
        cp "$FLOORPLAN_SRC" "$LAMAR_ROOT/hge.png"
    else
        echo "WARNING: floorplan not found at $FLOORPLAN_SRC; copy hge.png to $LAMAR_ROOT/hge.png"
    fi
fi
echo "done: $LAMAR_ROOT"
