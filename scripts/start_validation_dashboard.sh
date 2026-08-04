#!/usr/bin/env bash
set -euo pipefail
cd /home/chase/GitHub/RWKV-ECRA/frontend
exec npm run dev -- --host 0.0.0.0 --port 5177
