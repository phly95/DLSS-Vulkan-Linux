#!/usr/bin/env bash
# tools/scenario/gate_cell.sh — D1 cell gates (see D1-GATES.md; thresholds/sources quoted there).
#
#   bash tools/scenario/gate_cell.sh <repdir> <backend> <frames> [--sl]
#
# <repdir>  directory produced by run_cell.sh (or a hand-made copy of a harness run: it must
#           contain OptiScaler.log, dlss5-amd-ngx.log, client.log|run.stdout.log, eval_ms.csv,
#           summary.txt, stage_manifest.txt)
# <backend> EXPECTED backend, read from D1-MATRIX.csv — never inferred from the log (anti-fallback
#           rule, D1-GATES G-IDENT)
# exit: 0 all PASS (G-TIME may be DEGRADED) · 1 FAIL · 3 BLOCKED
set -uo pipefail

REP=${1:?repdir}; BACKEND=${2:?backend}; FRAMES=${3:?frames}; shift 3 || true
SL=0; [[ "${1:-}" == "--sl" ]] && SL=1

OLOG=$REP/OptiScaler.log
PLOG=$REP/dlss5-amd-ngx.log
PARSE=$REP/client.log; [[ -f $PARSE ]] || PARSE=$REP/run.stdout.log
MANI=$REP/stage_manifest.txt

RC=0
gate() {  # gate <name> <PASS|FAIL|BLOCKED|DEGRADED> <evidence>
    printf 'GATE %-14s %-8s %s\n' "$1" "$2" "$3"
    [[ $2 == FAIL ]] && RC=1
    [[ $2 == BLOCKED ]] && RC=3
}
count() { grep -c -- "$1" "$2" 2>/dev/null || true; }   # grep -c prints 0 on no-match

# --- G-ART ------------------------------------------------------------------------------------
missing=""
for f in "$OLOG" "$PLOG" "$REP/eval_ms.csv" "$REP/summary.txt" "$MANI"; do
    [[ -f $f ]] || missing="$missing $(basename "$f")"
done
if [[ -n $missing || ! -f $PARSE ]]; then
    gate G-ART FAIL "missing:$missing$([[ -f $PARSE ]] || echo ' client.log|run.stdout.log')"
    echo "CELL VERDICT: FAIL (repdir=$REP backend=$BACKEND frames=$FRAMES)"
    exit 1
fi
n_out=$(count '^OUT_NZ [0-9]+ nz=' "$PARSE")
n_csv=$(grep -cE '^[0-9]+,' "$REP/eval_ms.csv" 2>/dev/null || echo 0)
if [[ $n_out -eq $FRAMES && $n_csv -eq $FRAMES ]]; then
    gate G-ART PASS "OUT_NZ x$n_out csv_rows=$n_csv (frames=$FRAMES)"
else
    gate G-ART FAIL "OUT_NZ x$n_out csv_rows=$n_csv (frames=$FRAMES)"
fi

# --- G-IDENT ----------------------------------------------------------------------------------
n_create=$(count -F "Creating new $BACKEND upscaler" "$OLOG")
n_fall=$(count -E 'falling back to FSR|Upscaler can'"'"'t created' "$OLOG")
n_nvngx=$(count -E 'NVNGXProxy::InitNVNGX Loaded from .*_nvngx\.dll' "$OLOG")
n_nvidia=$(count -F 'Running on Nvidia' "$OLOG")
# staged model-DLL sizes: the port's own gate (src/pe/nvngx.cpp:187-191) refuses SR when
# nvngx_dlss.dll is absent/not 58956912 B - L4a proved a plain `cp` of the port over the model DLL
# kills SR. Sizes come from stage_manifest.txt, which run_cell.sh MUST write with `stat -Lc %s`
# (dereference): the entries are symlinks into the game dir, and a bare `stat -c %s` yields the
# LINK length (~82 B). Such link-length artifacts are the signature of a deref-less manifest and are
# reported INVALID/BLOCKED - never as a product FAIL.
if [[ $BACKEND == dlss || $BACKEND == dlssd ]]; then
    s_dlss=$(awk '$1=="nvngx_dlss.dll"{print $2}' "$MANI" 2>/dev/null | head -1)
    if [[ -z ${s_dlss:-} || $s_dlss == NA ]]; then
        gate G-STAGE BLOCKED "nvngx_dlss.dll missing from stage manifest (manifest not written / deref-less run)"
    elif [[ $s_dlss -lt 1000 ]]; then
        gate G-STAGE BLOCKED "manifest size=$s_dlss = SYMLINK LENGTH: stage_manifest.txt was written without 'stat -L' - result INVALID (re-run after run_cell.sh fix)"
    elif [[ $s_dlss != 58956912 ]]; then
        gate G-STAGE FAIL "nvngx_dlss.dll size=$s_dlss want=58956912 (nvngx.cpp:187-191; L4a shows this kills SR)"
    else
        gate G-STAGE PASS "nvngx_dlss.dll=58956912 (port's model-DLL gate satisfied; deref'd)"
    fi
fi

ident_fail=""
[[ $n_create -ge 1 ]] || ident_fail="$ident_fail no-create-line"
[[ $n_fall -eq 0 ]]   || ident_fail="$ident_fail fsr_fallback=$n_fall"
case $BACKEND in dlss|dlssd) [[ $n_nvngx -ge 1 ]] || ident_fail="$ident_fail no-_nvngx.dll-load";; esac
if [[ -z $ident_fail ]]; then
    gate G-IDENT PASS "'Creating new $BACKEND upscaler'x$n_create fsr_fallback=$n_fall nvngx_loaded=$n_nvngx (Running on Nvidia x$n_nvidia)"
else
    g=$(grep -m1 -F 'Creating new ' "$OLOG" | sed 's/.*\(Creating new [a-z0-9]* upscaler\).*/\1/')
    gate G-IDENT FAIL "want=$BACKEND got='${g:-none}'$ident_fail"
fi

# --- G-PORT (+G-PORT-RR / G-PORT-SR) ----------------------------------------------------------
n_ok=$(count -E 'eval #[0-9]+ OK:' "$PLOG")
n_fail=$(count -E 'eval #[0-9]+ FAIL:' "$PLOG")
n_fmt=$(count -F 'fmt col=0 mv=1 dep=0' "$PLOG")
n_d19=$(count -F 'Depth: DXGI format 19 is not one this bridge decodes' "$PLOG")
if [[ $n_ok -ge $FRAMES && $n_fail -eq 0 && $n_fmt -ge 1 && $n_d19 -eq 0 ]]; then
    gate G-PORT PASS "ok=$n_ok/$FRAMES fail=0 fmtX$n_fmt depth19=$n_d19"
else
    gate G-PORT FAIL "ok=$n_ok/$FRAMES fail=$n_fail fmtX$n_fmt depth19=$n_d19"
fi

if [[ $BACKEND == dlssd ]]; then
    size=$(awk '$1=="nvngx_dlssd.dll"{print $2}' "$MANI" 2>/dev/null | head -1)
    if [[ -z ${size:-} || $size == NA ]]; then
        gate G-PORT-RR BLOCKED "nvngx_dlssd.dll absent from stage manifest (N2 staging rule; want 48343664, nvngx.cpp:187-189)"
    elif [[ $size -lt 1000 ]]; then
        gate G-PORT-RR BLOCKED "manifest size=$size = SYMLINK LENGTH: manifest written without 'stat -L' - result INVALID (re-run)"
    elif [[ $size != 48343664 ]]; then
        gate G-PORT-RR FAIL "nvngx_dlssd.dll size=$size want=48343664 (nvngx.cpp:187-189)"
    else
        n_rr=$(count -E 'eval #[0-9]+ OK: id=[0-9]+ feature=13' "$PLOG")
        n_rrchain=$(count -E 'RR [0-9]+x[0-9]+: creating the frame chain from .*/rr' "$PLOG")
        if [[ $n_rr -ge $FRAMES && $n_rrchain -ge 1 ]]; then
            gate G-PORT-RR PASS "feature=13 OK x$n_rr chain-from-rr x$n_rrchain (size 48343664)"
        else
            gate G-PORT-RR FAIL "feature=13 OK x$n_rr/$FRAMES rr-chain x$n_rrchain"
        fi
    fi
elif [[ $BACKEND == dlss || $BACKEND == xess || $BACKEND == fsr21 || $BACKEND == fsr22 || $BACKEND == fsr31 ]]; then
    n_sr13=$(count -E 'feature=13' "$PLOG")
    if [[ $n_sr13 -eq 0 ]]; then
        gate G-PORT-SR PASS "no feature=13 (pure SR cell)"
    else
        gate G-PORT-SR FAIL "feature=13 leaked x$n_sr13 into an RR=off cell"
    fi
fi

# --- G-OUT ------------------------------------------------------------------------------------
viol=$(awk '/^OUT_NZ /{ split($3,a,"="); if (a[2]+0 < 0.10) bad++ } END{ print bad+0 }' "$PARSE")
mn=$(awk '/^OUT_NZ /{ split($3,a,"="); v=a[2]+0; if(!h||v<m){m=v;h=1} } END{ if(h) printf "%.6f",m }' "$PARSE")
[[ $viol -eq 0 ]] && gate G-OUT PASS "all frames nz>=0.10 (min nz=$mn)" \
                  || gate G-OUT FAIL "viol=$viol/$FRAMES min nz=$mn"

# --- G-TIME -----------------------------------------------------------------------------------
stats() {  # stdin ascending numbers -> "n median p95"
    awk '{ a[NR-1]=$1 } END{ n=NR; if(!n){print "0 -1 -1";exit}
        m=(n%2)?a[int(n/2)]:((a[n/2-1]+a[n/2])/2);
        k=int(0.95*n); if(0.95*n>k)k++; if(k<1)k=1; if(k>n)k=n;
        printf "%d %.4f %.4f\n", n, m, a[k-1] }'
}
ev=$(grep -E '^EVAL_MS [0-9]+ [0-9]+(\.[0-9]+)?$' "$PARSE" | awk '$2!="0"{print $3}' | sort -g | stats)
read -r n_e med_e p95_e <<<"$ev"
gp=$(grep -E '^GPU_EVAL_MS [0-9]+ [0-9]+(\.[0-9]+)?$' "$PARSE" | awk '$2!="0"{print $3}' | sort -g | stats)
read -r n_g med_g p95_g <<<"$gp"
n_na=$(count -E '^GPU_EVAL_MS [0-9]+ n/a' "$PARSE")
nan=$(count -E 'nan|inf|NaN|Inf' "$REP/eval_ms.csv")
if [[ $n_e -ge $((FRAMES - 1)) ]] && awk -v m="$med_e" 'BEGIN{exit !(m>0)}'; then
    if [[ $nan -gt 0 ]]; then
        gate G-TIME FAIL "nan/inf token in eval_ms.csv x$nan"
    elif [[ $n_g -gt 0 ]] && awk -v m="$med_g" 'BEGIN{exit !(m>0)}'; then
        gate G-TIME PASS "EVAL_MS n=$n_e med=$med_e p95=$p95_e | GPU_EVAL_MS n=$n_g med=$med_g p95=$p95_g (frames 1..)"
    else
        gate G-TIME DEGRADED "EVAL_MS n=$n_e med=$med_e p95=$p95_e | GPU_EVAL_MS n/a x$n_na (contract README:71-72)"
    fi
else
    gate G-TIME FAIL "EVAL_MS n=$n_e median=$med_e want n>=$((FRAMES-1)) median>0"
fi

# --- G-RR-ASSURE (--sl runs only) ------------------------------------------------------------
if [[ $SL -eq 1 ]]; then
    e=$REP/g5_evidence.txt
    [[ -f $e ]] || { cat "$PARSE" "$OLOG" > "$e" 2>/dev/null; }
    n_caps=$(count 'getSystemCaps' "$e"); n_nvfail=$(count 'NVAPI failed to initialize' "$e")
    n_init=$(count 'Failed to initialize NGX' "$e"); n_drv=$(count 'requires driver .*detected 0\.0\.0' "$e")
    n_miss=$(count 'Missing NGX context' "$e"); n_pl=$(count -F "Loaded plugin 'sl.dlss'" "$e")
    n_pld=$(count -F "Loaded plugin 'sl.dlss_d'" "$e")
    n_ign=$(count -E "Ignoring plugin 'sl.dlss_d'|adapter mask 0x0" "$e")
    fail=""
    [[ $n_caps -ge 1 && $n_nvfail -eq 0 ]] || fail="$fail caps(caps=$n_caps nvfail=$n_nvfail)"
    [[ $n_init -eq 0 ]] || fail="$fail init-ngx"
    [[ $n_drv -eq 0 ]]  || fail="$fail driver-0.0.0"
    [[ $n_miss -eq 0 ]] || fail="$fail missing-ctx"
    [[ $n_pl -ge 1 ]]   || fail="$fail sl.dlss-not-loaded"
    [[ $n_ign -eq 0 ]]  || fail="$fail sl.dlss_d-dropped(x$n_ign)"
    if [[ $BACKEND == dlssd && $n_pld -eq 0 && -n ${GATE_RR_EXPECTED_UNSUPPORTED:-} ]]; then
        gate G-RR-ASSURE BLOCKED "sl.dlss_d not loaded while L4/substitution state unknown (check REPORT.md)"
    elif [[ $BACKEND == dlssd && $n_pld -eq 0 ]]; then
        fail="$fail sl.dlss_d-not-loaded"
    fi
    [[ -z $fail ]] && gate G-RR-ASSURE PASS "caps=$n_caps init-ngx=$n_init drv0=$n_drv ctx=$n_miss plugin_sl.dlss=x$n_pl plugin_sl.dlss_d=x$n_pld dropped=$n_ign" \
                    || gate G-RR-ASSURE FAIL "$fail"
fi

case $RC in 0) v=PASS;; 3) v=BLOCKED;; *) v=FAIL;; esac
echo "CELL VERDICT: $v (repdir=$REP backend=$BACKEND frames=$FRAMES)"
exit $RC