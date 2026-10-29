// rr_texfmt.glsl: the RR input-format decoder shared by rr_gather and rr_enc0_feat (opus_impl:
// rr_enc0_feat reads the albedo / specular-albedo inputs directly instead of through rr_gather's
// f32x4 copies). Moved verbatim from rr_gather.comp.
#ifndef RR_TEXFMT_GLSL
#define RR_TEXFMT_GLSL

// Bytes per texel of each RR format code (tools/verify/texfmt.py RR_BPP).
uint rr_bpp(int code) {
    if (code == 0) return 8u;
    if (code == 1) return 4u;
    if (code == 2) return 4u;
    if (code == 3) return 4u;
    if (code == 4) return 2u;
    if (code == 5) return 4u;
    if (code == 6) return 4u;
    if (code == 7) return 4u;
    if (code == 8) return 1u;
    if (code == 9) return 2u;
    if (code == 10) return 8u;
    if (code == 11) return 16u;
    if (code == 12) return 4u;
    if (code == 13) return 4u;
    if (code == 14) return 8u;
    if (code == 15) return 8u;
    if (code == 16) return 4u;
    if (code == 17) return 4u;
    if (code == 18) return 4u;
    if (code == 19) return 4u;
    if (code == 20) return 2u;
    if (code == 21) return 2u;
    if (code == 22) return 2u;
    return 1u; // code == 23
}

float snorm_f(uint v, uint bits) {
    uint h = 1u << (bits - 1u);
    int s = (v >= h) ? int(v - (h << 1u)) : int(v);
    float m = float(h - 1u);
    return clamp(float(s) / m, -1.0, 1.0);
}

// Decode one texel of the shared 24-code table. c0..c3 are the four consecutive words
// from the texel's byte offset (missing channels pad as texfmt.encode does: 0,0,0,1).
vec4 dec_rr(int code, uint c0, uint c1, uint c2, uint c3) {
    if (code == 0) { // RGBA16F
        return vec4(float(uint16BitsToFloat16(uint16_t(c0 & 0xffffu))),
                    float(uint16BitsToFloat16(uint16_t((c0 >> 16) & 0xffffu))),
                    float(uint16BitsToFloat16(uint16_t(c1 & 0xffffu))),
                    float(uint16BitsToFloat16(uint16_t((c1 >> 16) & 0xffffu))));
    } else if (code == 1) { // R11G11B10F
        uint16_t r = uint16_t((c0 << 4) & 0x7ff0u);
        uint16_t g = uint16_t((c0 >> 7) & 0x7ff0u);
        uint16_t b = uint16_t((c0 >> 17) & 0x7fe0u);
        return vec4(float(uint16BitsToFloat16(r)), float(uint16BitsToFloat16(g)),
                    float(uint16BitsToFloat16(b)), 1.0);
    } else if (code == 2) { // RG16F
        return vec4(float(uint16BitsToFloat16(uint16_t(c0 & 0xffffu))),
                    float(uint16BitsToFloat16(uint16_t((c0 >> 16) & 0xffffu))), 0.0, 1.0);
    } else if (code == 3) { // R32F
        return vec4(uintBitsToFloat(c0), 0.0, 0.0, 1.0);
    } else if (code == 4) { // R16F
        return vec4(float(uint16BitsToFloat16(uint16_t(c0 & 0xffffu))), 0.0, 0.0, 1.0);
    } else if (code == 5) { // RGBA8 UNORM
        return vec4(float(c0 & 0xffu), float((c0 >> 8) & 0xffu),
                    float((c0 >> 16) & 0xffu), float(c0 >> 24)) / 255.0;
    } else if (code == 6) { // BGRA8 (byte0=b): r=byte2, g=byte1, b=byte0, a=byte3
        return vec4(float((c0 >> 16) & 0xffu), float((c0 >> 8) & 0xffu),
                    float(c0 & 0xffu), float(c0 >> 24)) / 255.0;
    } else if (code == 7) { // RGB10A2 UNORM
        return vec4(float(c0 & 0x3ffu) / 1023.0, float((c0 >> 10) & 0x3ffu) / 1023.0,
                    float((c0 >> 20) & 0x3ffu) / 1023.0, float(c0 >> 30) / 3.0);
    } else if (code == 8) { // R8 UNORM
        return vec4(float(c0 & 0xffu) / 255.0, 0.0, 0.0, 1.0);
    } else if (code == 9) { // R16 UNORM
        return vec4(float(c0 & 0xffffu) / 65535.0, 0.0, 0.0, 1.0);
    } else if (code == 10) { // RG32F
        return vec4(uintBitsToFloat(c0), uintBitsToFloat(c1), 0.0, 1.0);
    } else if (code == 11) { // RGBA32F
        return vec4(uintBitsToFloat(c0), uintBitsToFloat(c1),
                    uintBitsToFloat(c2), uintBitsToFloat(c3));
    } else if (code == 12) { // RGB9E5
        int exp_val = int((c0 >> 27) & 0x1fu) - 24;
        float scale = ldexp(1.0, exp_val);
        return vec4(float(c0 & 0x1ffu) * scale, float((c0 >> 9) & 0x1ffu) * scale,
                    float((c0 >> 18) & 0x1ffu) * scale, 1.0);
    } else if (code == 13) { // RGBA8 SNORM
        return vec4(snorm_f(c0 & 0xffu, 8u), snorm_f((c0 >> 8) & 0xffu, 8u),
                    snorm_f((c0 >> 16) & 0xffu, 8u), snorm_f(c0 >> 24, 8u));
    } else if (code == 14) { // RGBA16 UNORM
        return vec4(float(c0 & 0xffffu), float(c0 >> 16),
                    float(c1 & 0xffffu), float(c1 >> 16)) / 65535.0;
    } else if (code == 15) { // RGBA16 SNORM
        return vec4(snorm_f(c0 & 0xffffu, 16u), snorm_f(c0 >> 16, 16u),
                    snorm_f(c1 & 0xffffu, 16u), snorm_f(c1 >> 16, 16u));
    } else if (code == 16) { // D24 (low 24 bits)
        return vec4(float(c0 & 0xffffffu) / 16777215.0, 0.0, 0.0, 1.0);
    } else if (code == 17) { // BGRX8: r=byte2, g=byte1, b=byte0, a=1
        return vec4(float((c0 >> 16) & 0xffu) / 255.0, float((c0 >> 8) & 0xffu) / 255.0,
                    float(c0 & 0xffu) / 255.0, 1.0);
    } else if (code == 18) { // RG16 UNORM
        return vec4(float(c0 & 0xffffu) / 65535.0, float(c0 >> 16) / 65535.0, 0.0, 1.0);
    } else if (code == 19) { // RG16 SNORM
        return vec4(snorm_f(c0 & 0xffffu, 16u), snorm_f(c0 >> 16, 16u), 0.0, 1.0);
    } else if (code == 20) { // RG8 UNORM
        return vec4(float(c0 & 0xffu) / 255.0, float((c0 >> 8) & 0xffu) / 255.0, 0.0, 1.0);
    } else if (code == 21) { // RG8 SNORM
        return vec4(snorm_f(c0 & 0xffu, 8u), snorm_f((c0 >> 8) & 0xffu, 8u), 0.0, 1.0);
    } else if (code == 22) { // R16 SNORM
        return vec4(snorm_f(c0 & 0xffffu, 16u), 0.0, 0.0, 1.0);
    } else { // code == 23, R8 SNORM
        return vec4(snorm_f(c0 & 0xffu, 8u), 0.0, 0.0, 1.0);
    }
}

// One texel of a raw input at (x, y): byte offset, word-aligned window, decode (rr_gather's
// per-slot fetch, as a macro so it works on any buffer).
#define RR_FETCH(BUF, PITCH, FMT, X, Y, OUT) { \
    uint o_ = uint(Y) * (PITCH) + uint(X) * rr_bpp(FMT); \
    uint base_ = o_ >> 2u; uint s_ = (o_ & 3u) * 8u; \
    uint w0_ = BUF[base_], w1_ = BUF[base_ + 1u]; \
    uint w2_ = (rr_bpp(FMT) > 8u) ? BUF[base_ + 2u] : 0u; \
    uint w3_ = (rr_bpp(FMT) > 8u) ? BUF[base_ + 3u] : 0u; \
    uint c0_ = (s_ == 0u) ? w0_ : ((w0_ >> s_) | (w1_ << (32u - s_))); \
    uint c1_ = (s_ == 0u) ? w1_ : ((w1_ >> s_) | (w2_ << (32u - s_))); \
    uint c2_ = (s_ == 0u) ? w2_ : ((w2_ >> s_) | (w3_ << (32u - s_))); \
    uint c3_ = (s_ == 0u) ? w3_ : (w3_ >> s_); \
    OUT = dec_rr(FMT, c0_, c1_, c2_, c3_); }

#endif
