import pefile, struct, sys
class PE:
    def __init__(s, path):
        s.pe = pefile.PE(path, fast_load=True); s.base = s.pe.OPTIONAL_HEADER.ImageBase
        s.img = s.pe.get_memory_mapped_image()
    def rd(s, va, n): o = va - s.base; return s.img[o:o+n]
    def q(s, va): return struct.unpack('<Q', s.rd(va, 8))[0]
    def d(s, va): return struct.unpack('<I', s.rd(va, 4))[0]
    def i(s, va): return struct.unpack('<i', s.rd(va, 4))[0]
    def f(s, va): return struct.unpack('<f', s.rd(va, 4))[0]
    def cstr(s, va):
        b = s.rd(va, 256); return b.split(b'\0')[0].decode('latin1')
    def isptr(s, v): return s.base <= v < s.base + len(s.img)
