#!/bin/bash
# fn.sh <net> <mangled>: one function's disassembly, from the *.s.xz that
# tools/exe/fetch_neuralhub.sh regenerates into build/nh (DLSS5_AMD_NH overrides it).
NH=${DLSS5_AMD_NH:-$(cd -- "$(dirname -- "$0")/../.." && pwd)/build/nh}
xz -dc "$NH/$1.s.xz" | awk -v n="<$2>:" '$0 ~ /^[0-9a-f]+ <.*>:$/ {on = (index($0, n) > 0)} on'
