#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"
if [ ! -f data/ml-100k/u.data ] || [ ! -f data/ml-100k/u.item ]; then
  printf '%s\n' "MovieLens data is missing. Review DATA.md, then run:"
  printf '%s\n' "  python3 download_data.py --accept-license"
  exit 2
fi
python3 run_demo.py
printf '%s\n' "Run 'python3 app.py' and open http://127.0.0.1:8012"
