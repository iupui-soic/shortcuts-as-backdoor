#!/usr/bin/env bash
# CheXpert Plus download driver. Sources the Redivis token from env.txt without
# echoing it, and runs the resumable downloader under the isolated venv.
set -uo pipefail
cd "$(dirname "$0")/../.."
VENV=${VENV:?set VENV to the redivis venv}
set -a; . ./env.txt; set +a
export REDIVIS_API_TOKEN="$REDIVIS_TOKEN"
unset REDIVIS_TOKEN KAGGLE_TOKEN
exec "$VENV/bin/python" scripts/revision/download_chexpert_plus.py "$@"
