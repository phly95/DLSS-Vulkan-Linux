#!/bin/bash
# fn.sh <net> <mangled>: one function's disassembly
xz -dc "$(cd -- "$(dirname -- "$0")/../.." && pwd)"/artifacts/neuralhub-v0.8.0/gfx1201/$1.s.xz | awk -v n="<$2>:" '$0 ~ /^[0-9a-f]+ <.*>:$/ {on = (index($0, n) > 0)} on'
