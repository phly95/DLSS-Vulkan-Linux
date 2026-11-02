#!/usr/bin/env bash
# tools/optiscaler-fork/patch.sh - apply the DLSS5AMD fork diff to a FRESH COPY of the vendored
# OptiScaler-NR-src tree (GPL-3.0; the fork stays GPL - see README.md).
#
#   tools/optiscaler-fork/patch.sh <scratch-copy-root>
#
# <scratch-copy-root> must be a *copy* of reference-nr-vulkan/artifacts/ref/OptiScaler-NR-src
# (e.g. /tmp/opencode/optiscaler-fork). The vendored tree itself is REFUSED as the target.
# Idempotent: every edit checks whether it is already applied, and fails loudly when the vendored
# lines it keys on are gone (the tree moved under us).
set -euo pipefail

ROOT=${1:?usage: patch.sh <scratch-copy of OptiScaler-NR-src>}
cd "$ROOT"
echo "fork patch target: $ROOT"

if [[ ! -f OptiScaler/OptiTypes.h || ! -f OptiScaler/OptiScaler.vcxproj ]]; then
    echo "FATAL: $ROOT does not look like the OptiScaler-NR-src tree" >&2
    exit 1
fi
# Refuse to patch the vendored original in place.
if [[ "$(git rev-parse --git-dir 2>/dev/null || true)" != "" ]] \
   && git -C "$ROOT" remote -v 2>/dev/null | grep -q "OptiScaler-NR-src"; then
    echo "FATAL: refusing to patch the vendored reference tree in place" >&2
    exit 1
fi

die() { echo "FATAL: $*" >&2; exit 1; }
guard_absent() {  # file pattern -> ok if the pattern is ALREADY absent (not-yet-patched or n/a)
    if grep -qF "$2" "$1" 2>/dev/null; then echo "FATAL: $1 still contains '$2' (unexpected)" >&2; exit 1; fi
}

# ---------------------------------------------------------------- 1. the enum value
F=OptiScaler/OptiTypes.h
grep -qF "DLSS5AMD" "$F" || true   # idempotence probe only; the real check is the anchor below
if ! grep -qF "    DLSSD, // \"dlssd\", used for the DLSS-D/Ray Reconstruction upscaler+denoiser backend" "$F"; then
    die "$F: anchor for the Upscaler enum not found (tree changed?)"
fi
if ! grep -qF "DLSS5AMD, // \"dlss5_amd\", DLSS5-AMD FFX front-end running the repo's Vulkan chain" "$F"; then
    python3 - "$F" <<'EOF'
import sys
p = sys.argv[1]
s = open(p, newline='').read()
anchor = '    DLSSD, // "dlssd", used for the DLSS-D/Ray Reconstruction upscaler+denoiser backend\n'
add = '    DLSS5AMD, // "dlss5_amd", DLSS5-AMD FFX front-end running the repo\'s Vulkan chain\n'
assert anchor in s, "enum anchor not found"
s = s.replace(anchor, anchor + add, 1)
open(p, 'w', newline='').write(s)
EOF
    echo "  $F: DLSS5AMD enum value added"
else
    echo "  $F: DLSS5AMD already present"
fi

# ---------------------------------------------------------------- 2. display name + short name + IsFsr + code tables
F=OptiScaler/OptiTypes.cpp
if ! grep -qF 'case Upscaler::DLSS5AMD:' "$F"; then
    python3 - "$F" <<'EOF'
import sys
p = sys.argv[1]
s = open(p, newline='').read()

# UpscalerDisplayName: add the DLSS5-AMD name right after the DLSSD case (the switch has no
# default; every enumerator must return or fall through the end).
anchor = '''    case Upscaler::DLSSD:
        return "DLSSD";
    }

    return "????";'''
if anchor not in s:
    import re
    m = re.search(r'(    case Upscaler::DLSSD:\n        return "[^"]+";\n)(    \})', s)
    assert m, "DLSSD display case not found"
    s = s.replace(m.group(0), m.group(1) + '\n    case Upscaler::DLSS5AMD:\n        return "DLSS5-AMD";\n' + m.group(2), 1)
else:
    s = s.replace(anchor,
                  '''    case Upscaler::DLSSD:
        return "DLSSD";

    case Upscaler::DLSS5AMD:
        return "DLSS5-AMD";
    }

    return "????";''', 1)

# UpscalerShortName: IsFsr's members come back "FSR"; DLSS5AMD is NOT IsFsr's business (the FFX
# feature class does not need it), so give it its own short name next to DLSSD's.
anchor2 = '''    case Upscaler::DLSSD:
        return "DLSSD";
    }'''
if anchor2 in s:
    s = s.replace(anchor2,
                  '''    case Upscaler::DLSSD:
        return "DLSSD";

    case Upscaler::DLSS5AMD:
        return "DLSS5-AMD";
    }''', 1)

# UpscalerToCode: the ini value round-trip needs the save side too.
anchor3 = '''    case Upscaler::DLSSD:
        return "dlssd";
    case Upscaler::FSR31: // DX11 only
        return "fsr31";'''
if anchor3 in s:
    s = s.replace(anchor3,
                  '''    case Upscaler::DLSSD:
        return "dlssd";
    case Upscaler::DLSS5AMD:
        return "dlss5_amd";
    case Upscaler::FSR31: // DX11 only
        return "fsr31";''', 1)

# CodeToUpscaler: the ini value -> enum side ("dlss5_amd" selects the fork entry).
anchor4 = '''        { "dlss", Upscaler::DLSS },         { "dlssd", Upscaler::DLSSD },
        { "dlss_12", Upscaler::DLSS_on12 }, { "fsr31", Upscaler::FSR31 },'''
if anchor4 in s:
    s = s.replace(anchor4,
                  '''        { "dlss", Upscaler::DLSS },         { "dlssd", Upscaler::DLSSD },
        { "dlss_12", Upscaler::DLSS_on12 }, { "fsr31", Upscaler::FSR31 },
        { "dlss5_amd", Upscaler::DLSS5AMD }, { "dlss5amd", Upscaler::DLSS5AMD },''', 1)

open(p, 'w', newline='').write(s)
EOF
    echo "  $F: display/short/code/parse cases added"
else
    echo "  $F: DLSS5AMD cases already present"
fi
grep -qF 'case Upscaler::DLSS5AMD:' "$F" || die "$F: patch did not land"
grep -qF '"dlss5_amd", Upscaler::DLSS5AMD' "$F" || die "$F: CodeToUpscaler mapping did not land"

# ---------------------------------------------------------------- 3. menu: the DX12 combo list
F=OptiScaler/menu/menu_common.cpp
grep -qF "void MenuCommon::AddDx12Backends" "$F" || die "$F: AddDx12Backends not found"
if ! grep -qF "Upscaler::DLSS5AMD" "$F"; then
    python3 - "$F" <<'EOF'
import sys
p = sys.argv[1]
s = open(p, newline='').read()
anchor = '''void MenuCommon::AddDx12Backends(Upscaler upscaler)
{
    RenderUpscalerCombo(API::DX12, upscaler,
                        { Upscaler::XeSS, Upscaler::FSR21, Upscaler::FSR22, Upscaler::FFX, Upscaler::DLSS });
}'''
new = '''void MenuCommon::AddDx12Backends(Upscaler upscaler)
{
    RenderUpscalerCombo(API::DX12, upscaler,
                        { Upscaler::XeSS, Upscaler::FSR21, Upscaler::FSR22, Upscaler::FFX, Upscaler::DLSS5AMD,
                          Upscaler::DLSS });
}'''
assert anchor in s, "AddDx12Backends anchor not found"
s = s.replace(anchor, new, 1)
open(p, 'w', newline='').write(s)
EOF
    echo "  $F: DLSS5AMD added to the DX12 backend combo"
else
    echo "  $F: DLSS5AMD already in the combo"
fi

# ---------------------------------------------------------------- 4. routing: DLSS5AMD takes the FFX DX12 feature path
F=OptiScaler/upscalers/FeatureProvider_Dx12.cpp
grep -qF "case Upscaler::FFX:" "$F" || die "$F: FFX case not found"
if ! grep -qF "case Upscaler::DLSS5AMD:" "$F"; then
    python3 - "$F" <<'EOF'
import sys
p = sys.argv[1]
s = open(p, newline='').read()
anchor = '''    case Upscaler::FFX:
        *feature = std::make_unique<FFXFeatureDx12>(handleId, parameters);
        break;'''
new = '''    case Upscaler::FFX:
        *feature = std::make_unique<FFXFeatureDx12>(handleId, parameters);
        break;

    case Upscaler::DLSS5AMD:
        // DLSS5-AMD (repo fork): the same FFX feature class, i.e. the same FfxApiProxy SR slot
        // and the same Config [Libraries] FfxDx12SRPath override resolution - no path is
        // hardcoded here; the override mechanism stays authoritative.
        *feature = std::make_unique<FFXFeatureDx12>(handleId, parameters);
        break;'''
assert anchor in s, "FFX case anchor not found"
s = s.replace(anchor, new, 1)
open(p, 'w', newline='').write(s)
EOF
    echo "  $F: DLSS5AMD case routed to the FFX feature class"
else
    echo "  $F: DLSS5AMD case already present"
fi
grep -qF "case Upscaler::DLSS5AMD:" "$F" || die "$F: routing did not land"

# ---------------------------------------------------------------- 5. compile-demand sites:
# C++ switch over Upscaler with no default where the compiler needs every enumerator
# (UpscalerShortName already got its case; the two GetFeature switches have default:).
# Any other switch that MSVC's /W3 treats as warning-only stays untouched on purpose.

echo "patch applied OK"
