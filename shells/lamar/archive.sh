#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -v RTYPE=rt_HC
#PBS -m abe
#PBS -k oed
# Zips the data directory into <DATA>.zip, tests the archive, then deletes the directory.
# qsub shells/lamar/archive.sh  (DATA: default /groups/gai51740/RBT3/umemura/zfloc/dataset/lamar)
set -euo pipefail
DATA="${DATA:-/groups/gai51740/RBT3/umemura/zfloc/dataset/lamar}"
DATA="${DATA%/}"
log() { echo "[$(date '+%F %T')] $*"; }

[ -d "$DATA" ] || { log "$DATA does not exist"; exit 1; }
[ ! -e "$DATA.zip" ] || { log "$DATA.zip already exists: not overwritten"; exit 1; }
log "$DATA: $(du -sh "$DATA" | cut -f1), $(find "$DATA" -type f | wc -l) files"

log "zip -> $DATA.zip"
cd "$(dirname "$DATA")"
zip -r -q "$DATA.zip" "$(basename "$DATA")"
log "zip done: $(du -sh "$DATA.zip" | cut -f1)"

log "testing the archive"
zip -T "$DATA.zip"  # exits with an error (and nothing is deleted) if the archive is broken
files_dir=$(find "$DATA" -type f | wc -l)
files_zip=$(zipinfo -1 "$DATA.zip" | grep -cv '/$')
[ "$files_dir" -eq "$files_zip" ] || { log "files: $files_dir in the directory, $files_zip in the zip: not deleted"; exit 1; }

log "deleting $DATA"
rm -rf "$DATA"
log "done: $DATA.zip"
