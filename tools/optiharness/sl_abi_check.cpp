// sl_abi_check: compile-time proof that the vendored Streamline v2.14.1 headers match the
// layouts this harness depends on before opti_client.cpp is ever shipped (build.sh runs this
// through the host g++ with -fsyntax-only when one exists; the same asserts are pinned inside
// opti_client.cpp for every mingw build).
//
// Expected numbers below were derived BY HAND from the fetched header source (citations), then
// cross-checked by this translation unit - a mismatch fails the build instead of producing a
// wrong-offset sl::Preferences handed to the game's interposer:
//   sl::BaseStructure   sl_struct.h: "BaseStructure* next; StructType structType; size_t structVersion"
//                       = 8 + 16 + 8 = 32 (x86-64, default packing)
//   sl::Preferences     sl_core_types.h SL_STRUCT_BEGIN(Preferences, ...): base 32, then
//                       bool showConsole(32), LogLevel u32(36), wchar_t**(40), u32 numPaths(48),
//                       wchar_t*(56), 3 callbacks (64/72/80), PreferenceFlags u64(88),
//                       Feature*(96), u32 numFeatures(104), u32 applicationId(108),
//                       EngineType u32(112), const char* engineVersion(120),
//                       const char* projectId(128), RenderAPI u32(136) -> sizeof 144
//   sl::Version         sl_version.h: 3 x u32 = 12
//   kSDKVersion         sl_version.h: (2<<48)|(14<<32)|(1<<16)|0xfedc  (the game's host SDK 2.14.1)
#include <cstddef>
#include <cstdint>

#include "sl-include/sl.h"

static_assert(sizeof(sl::StructType) == 16, "GUID as sl_struct.h declares it");
static_assert(sizeof(sl::BaseStructure) == 32, "next + structType + size_t structVersion");
static_assert(sizeof(sl::Version) == 12, "3 x uint32_t");
static_assert(sizeof(sl::Preferences) == 144, "the slInit parameter block, hand-derived above");
static_assert(offsetof(sl::Preferences, logMessageCallback) == 80, "callback slot the harness installs");
static_assert(offsetof(sl::Preferences, featuresToLoad) == 96, "features list");
static_assert(offsetof(sl::Preferences, applicationId) == 108, "the game's 101640911");
static_assert(offsetof(sl::Preferences, renderAPI) == 136, "sl::RenderAPI");
static_assert(sizeof(sl::LogLevel) == 4 && sizeof(sl::RenderAPI) == 4 && sizeof(sl::EngineType) == 4,
              "the enums are uint32_t-backed");
static_assert(sizeof(sl::PreferenceFlags) == 8, "uint64_t-backed flag word");
static_assert(sl::kSDKVersion == ((uint64_t(2) << 48) | (uint64_t(14) << 32) | (uint64_t(1) << 16) | 0xfedc),
              "host SDK v2.14.1, the version the game's archive reports");
static_assert(int(sl::Result::eOk) == 0, "sl::Result defaults to 0.");
static_assert(sizeof(sl::AdapterInfo) == 56,           // base32 + LUID ptr8 + u32 size4 + pad4 + vkptr8
              "probe struct slIsFeatureSupported takes");
static_assert(sizeof(sl::FeatureRequirements) > 32,
              "the two probe structs slIsFeatureSupported / slGetFeatureRequirements take");

int main() { return 0; }