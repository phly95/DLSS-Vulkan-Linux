#!/usr/bin/env bash
# tools/optiscaler-fork/build.sh - build the DLSS5AMD fork of OptiScaler with an msvc-wine
# toolchain (cl.exe/rc.exe/link.exe driven straight through wine - no MSBuild: OptiScaler.sln
# is VS2022-only and this box has no Visual Studio).
#
#   tools/optiscaler-fork/build.sh <patched-scratch-copy>
#
# <patched-scratch-copy> must have been through tools/optiscaler-fork/patch.sh first (this
# script refuses an unpatched tree, so a fork binary can never masquerade as stock).
#
# Environment:
#   MSVC_ROOT   msvc-wine install root (default /tmp/opencode/msvc)
#   WINE        wine binary           (default: Proton 10.0's wine, the measured driver)
#   WORK        build dir (pch/obj/res/dll, REUSED across runs - TUs skip when their .obj is
#               present)             (default /tmp/opencode/fork-build)
#   WINEPREFIX  wine prefix for cl/link (default /tmp/opencode/wine-msvc)
#
# Everything this script knows that the vcxproj does not say out loud (measured 2026-11-01,
# the ffx2 cell): MSVC 14.51.36231 + SDK 10.0.26100.0, 229 TUs, ~13 min cold / ~2 min warm,
# OptiScaler.dll 26,515,968 B, all four ffx2 gates green (REPORT rows S8-FFX2-*).
#   1. resource.h gates VER_*_STR behind OPTISCALER_BUILD_METADATA -> rc needs /d it, plus the
#      two generated headers resource_build_{date,commit}.h (the vcxproj pre-build writes
#      those; this script generates them when absent).
#   2. The vcxproj lists TUs with BACKSLASH paths; wine materialises /Fo targets as SUBDIRS
#      (obj/dlssnr/Foo.obj), so the resume test must translate \ -> / first, and the link
#      list must walk the WHOLE tree (obj/*.obj would see 9 of 231).
#   3. pch.obj (the /Yc producer) lands at the WORK root, not under obj/ - without it the
#      link dies LNK2011 ("precompiled object not linked in") on every /Yu object.
#   4. Two vendored libs sit outside OptiScaler/library: external/freetype (freetype.lib) and
#      external/xess/lib - both need their own /LIBPATH, and d3dx11.lib must be linked too.
set -euo pipefail

FORK=${1:?usage: build.sh <patched-scratch-copy of OptiScaler-NR-src (patch.sh applied)>}
MSVC_ROOT=${MSVC_ROOT:-/tmp/opencode/msvc}
WORK=${WORK:-/tmp/opencode/fork-build}
WINE=${WINE:-"$HOME/.local/share/Steam/steamapps/common/Proton 10.0/files/bin/wine"}
export WINEPREFIX=${WINEPREFIX:-/tmp/opencode/wine-msvc}
export WINEDEBUG=${WINEDEBUG:--all}

[[ -f "$FORK/OptiScaler/OptiScaler.vcxproj" ]] || { echo "FATAL: $FORK is not an OptiScaler tree" >&2; exit 1; }
grep -qF DLSS5AMD "$FORK/OptiScaler/OptiTypes.h" \
    || { echo "FATAL: $FORK is not patched - run tools/optiscaler-fork/patch.sh first" >&2; exit 1; }
CL=$(ls "$MSVC_ROOT"/vc/tools/msvc/*/bin/Hostx64/x64/cl.exe 2>/dev/null | head -1)
LINK=$(ls "$MSVC_ROOT"/vc/tools/msvc/*/bin/Hostx64/x64/link.exe 2>/dev/null | head -1)
RCBIN=$(ls "$MSVC_ROOT"/Windows\ Kits/10/bin/*/x64/rc.exe 2>/dev/null | head -1)
[[ -n "$CL" && -n "$LINK" && -n "$RCBIN" ]] || { echo "FATAL: no msvc-wine toolchain under $MSVC_ROOT" >&2; exit 1; }
MSVCDIR=$(dirname "$(dirname "$(dirname "$CL")")")           # .../bin/Hostx64/x64 -> up to msvc root
LIBMSVC=$(dirname "$MSVCDIR")/lib/x64
KROOT=$(cd "$(dirname "$RCBIN")/../../.." && pwd)            # Windows Kits/10
SDKVER=$(basename "$(dirname "$(dirname "$RCBIN")")")        # 10.0.26100.0
CLW="Z:${CL//\//\\}"; LINKW="Z:${LINK//\//\\}"; RCBW="Z:${RCBIN//\//\\}"
FORKW="Z:${FORK//\//\\}"; WORKW="Z:${WORK//\//\\}"
MSVCW="Z:${MSVC_ROOT//\//\\}"
# cl's /I values: BACKSLASH, NO Z: prefix (drive-relative to the cwd drive) - measured:
# /IZ:Z:\tmp\... -> C1083, /IZ:\tmp\... -> compiles.  rc/link keep the Z: form.
FORKD="${FORK//\//\\}"; MSVCD="${MSVC_ROOT//\//\\}"

mkdir -p "$WORK"
cd "$WORK"

# ---- generated pre-build artifacts (vcxproj pre-build event, translated) ------------------
[[ -f "$FORK/OptiScaler/resource_build_date.h" ]] \
    || printf '#define VER_BUILD_DATE "%s"\n' "$(date +%Y%m%d)" > "$FORK/OptiScaler/resource_build_date.h"
[[ -f "$FORK/OptiScaler/resource_build_commit.h" ]] \
    || printf '#define VER_BUILD_COMMIT "dlss5amd-fork"\n' > "$FORK/OptiScaler/resource_build_commit.h"

# ---- TU list + PCH exceptions, parsed from the vcxproj itself ------------------------------
python3 - "$FORK/OptiScaler/OptiScaler.vcxproj" "$WORK/tus.txt" "$WORK/nopch.txt" <<'PYEOF'
import re, sys
vcx, tus_out, nopch_out = sys.argv[1:4]
s = open(vcx, encoding='utf-8-sig').read()
tus, nopch = [], []
for m in re.finditer(r'<ClCompile Include="([^"]+\.cpp)">(.*?)</ClCompile>|<ClCompile Include="([^"]+\.cpp)"\s*/>',
                     s, re.S):
    path = m.group(1) or m.group(3)
    body = m.group(2) or ""
    tus.append(path)
    if re.search(r'<PrecompiledHeader[^>]*>NotUsing</PrecompiledHeader>', body):
        nopch.append(path)
open(tus_out, 'w').write('\n'.join(tus) + '\n')
open(nopch_out, 'w').write('\n'.join(nopch) + '\n')
print(f"{len(tus)} TUs, {len(nopch)} PCH exceptions (from the vcxproj)")
PYEOF

# ---- compile flags: the vcxproj Release|x64 ClCompile block, translated --------------------
DEFS='/DNDEBUG /D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR /D_WINDOWS /D_USRDLL /DUNICODE /D_UNICODE'
STD='/c /std:c++latest /EHsc /Zc:__cplusplus /MD /O2 /fp:fast /GL- /MP1 /w34996 /Zc:templateScope- /permissive-'

# include dirs (vcxproj IncludePath), then the toolchain + SDK sets.  Entries whose path has
# a space (Windows Kits) are quoted INSIDE the response file.
INCS=(
 "/IZ:$FORKD\\OptiScaler"
 "/IZ:$FORKD\\OptiScaler\\include"
 "/IZ:$FORKD\\external\\vulkan\\include"
 "/IZ:$FORKD\\external\\nvngx_dlss_sdk"
 "/IZ:$FORKD\\external\\xess\\inc\\xess"
 "/IZ:$FORKD\\external\\xess\\inc\\xell"
 "/IZ:$FORKD\\external\\xess\\inc\\xess_fg"
 "/IZ:$FORKD\\external\\FidelityFX-SDK\\ffx-api\\include\\ffx_api"
 "/IZ:$FORKD\\external\\simpleini"
 "/IZ:$FORKD\\external\\unordered_dense\\include"
 "/IZ:$FORKD\\external\\spdlog\\include"
 "/IZ:$FORKD\\external\\freetype"
 "/IZ:$FORKD\\external\\streamline"
 "/IZ:$FORKD\\external\\streamline1"
 "/IZ:$FORKD\\external\\nvapi"
 "/IZ:$FORKD\\external\\nlohmann"
 "/IZ:$FORKD\\external\\magic_enum\\include\\magic_enum"
 "/IZ:$FORKD\\external\\AntiLag2-SDK"
 "/IZ:$FORKD\\external\\latencyflex"
 "/IZ:$MSVCD\\vc\\tools\\msvc"
 "\"/IZ:$MSVCD\\Windows Kits\\10\\Include\\$SDKVER\\ucrt\""
 "\"/IZ:$MSVCD\\Windows Kits\\10\\Include\\$SDKVER\\shared\""
 "\"/IZ:$MSVCD\\Windows Kits\\10\\Include\\$SDKVER\\um\""
 "\"/IZ:$MSVCD\\Windows Kits\\10\\Include\\$SDKVER\\winrt\""
)
# the MSVC include dir needs its version segment (vc/tools/msvc/<ver>/include)
MSVCVER=$(basename "$(dirname "$MSVCDIR")")     # MSVCDIR = .../msvc/<ver>/bin -> <ver>
[[ -d "$MSVC_ROOT/vc/tools/msvc/$MSVCVER/include" ]] \
    || { echo "FATAL: cannot derive the MSVC version from $CL" >&2; exit 1; }
INCS[19]="/IZ:$MSVCD\\vc\\tools\\msvc\\$MSVCVER\\include"
printf '%s\n' "${INCS[@]}" > incs.rf

# ---- 1. pch (once) --------------------------------------------------------------------------
if [[ ! -f pch.pch ]]; then
    echo "[1/4] pch"
    { cat incs.rf; echo "$DEFS"; echo "$STD"; echo '/Ycpch.h /Fppch.pch /Fdpch.pdb /c'; \
      echo "$FORKW\\OptiScaler\\pch.cpp"; } > rf.txt
    "$WINE" "$CLW" @rf.txt
fi

# ---- 2. all TUs (resume: skip when the .obj exists) ------------------------------------------
echo "[2/4] TUs"
i=0; failed=0
while IFS= read -r tu || [[ -n "$tu" ]]; do
    tu=${tu%$'\r'}; [[ -n "$tu" ]] || continue
    tu=${tu//\//\\}                       # normalise to the vcxproj's backslash form
    i=$((i+1))
    objname=$(printf '%s' "${tu%.cpp}").obj
    objposix=$(printf '%s' "$tu" | tr '\\' '/' | sed 's/\.cpp$//')
    obj="obj/$objposix.obj"
    [[ -f "$obj" ]] && continue
    ph='/Yupch.h /Fppch.pch /Fdpch.pdb'
    grep -qxF "$tu" nopch.txt && ph='/Y-'
    { cat incs.rf; echo "$DEFS"; echo "$STD"; echo "$ph"; echo "$FORKW\\OptiScaler\\$tu"; \
      echo "/Fo$WORK\\obj\\$objname"; } > tu.txt
    log="obj/$objposix.log"
    if ! "$WINE" "$CLW" @tu.txt > "$log" 2>&1; then
        echo "FAIL $tu (log: $log)"; failed=$((failed+1))
        if (( failed > 8 )); then echo "too many failures, stopping"; exit 1; fi
    else
        printf '\r%d/%d' "$i" "$(wc -l < tus.txt)"
    fi
done < tus.txt
echo
(( failed == 0 )) || { echo "$failed TUs failed"; exit 1; }

# ---- 3. resource ------------------------------------------------------------------------------
echo "[3/4] rc"
"$WINE" "$RCBW" /d NDEBUG /d _UNICODE /d UNICODE /d OPTISCALER_BUILD_METADATA \
   /I "$FORKW\\OptiScaler" \
   /I "$MSVCW\\Windows Kits\\10\\Include\\$SDKVER\\um" \
   /I "$MSVCW\\Windows Kits\\10\\Include\\$SDKVER\\shared" \
   /I "$MSVCW\\Windows Kits\\10\\Include\\$SDKVER\\ucrt" \
   /fo OptiScaler.res "$FORKW\\OptiScaler\\OptiScaler.rc"

# ---- 4. link ------------------------------------------------------------------------------------
echo "[4/4] link"
# every .obj under WORK: pch.obj sits at the root, TUs in obj/<mirrored vcxproj path>/
python3 - "$WORK" <<'PYEOF'
import os, sys
work = sys.argv[1]
objs = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(work)
              for f in fs if f.endswith(".obj"))
with open(os.path.join(work, "linkobjs.rf"), "w") as out:
    for n in objs:
        out.write('"Z:' + n.replace("/", "\\") + '"\n')
print(len(objs), "objs for the link")
PYEOF
LIBS='winhttp.lib WindowsApp.lib Dbghelp.lib dxgi.lib d3d11.lib d3d12.lib vulkan-1.lib dxguid.lib freetype.lib ffx_fsr2_api_x64.lib ffx_fsr2_api_dx11_x64.lib ffx_fsr2_api_dx12_x64.lib ffx_fsr2_api_vk_x64.lib ffx_fsr2_212_api_dx12_x64.lib ffx_fsr2_212_api_vk_x64.lib ffx_fsr2_212_api_x64.lib d3dcompiler.lib d3dx11.lib ffx_backend_dx11_x64.lib ffx_fsr3_x64.lib ffx_frameinterpolation_x64.lib ffx_fsr3upscaler_x64.lib ffx_opticalflow_x64.lib'
{
    echo '/NOLOGO /DLL /MACHINE:X64'
    echo "/DEF:\"$FORKW\\OptiScaler\\Source.def\""
    echo "/OUT:$WORKW\\OptiScaler.dll"
    for d in fsr2 fsr2_212 fsr31 vulkan d3dx detours; do
        echo "/LIBPATH:\"$FORKW\\OptiScaler\\library\\$d\""
    done
    echo "/LIBPATH:\"$FORKW\\external\\freetype\""
    echo "/LIBPATH:\"$FORKW\\external\\xess\\lib\""
    echo "/LIBPATH:\"$MSVCW\\vc\\tools\\msvc\\$MSVCVER\\lib\\x64\""
    echo "/LIBPATH:\"$MSVCW\\Windows Kits\\10\\Lib\\$SDKVER\\um\\x64\""
    echo "/LIBPATH:\"$MSVCW\\Windows Kits\\10\\Lib\\$SDKVER\\ucrt\\x64\""
    echo "$LIBS"
    echo '/DELAYLOAD:vulkan-1.dll /DELAYLOAD:dxgi.dll /DELAYLOAD:d3d12.dll /DELAYLOAD:D3DCOMPILER_47.dll delayimp.lib'
    cat linkobjs.rf
    echo OptiScaler.res
} > linkall.rf
"$WINE" "$LINKW" @linkall.rf 2>&1 | tail -20
ls -la "$WORK/OptiScaler.dll"