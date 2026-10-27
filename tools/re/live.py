"""tools/re/live.py SSA_FILE ADDR REG...: for patch probes (tools/re/patchprobe.py): is each register still holding its last pre-ADDR definition at ADDR?
(destination annotations '; vN', '; s[a:b]' from tools/re/ssa.py; textual order)"""
import re, sys
ssa, at = sys.argv[1], int(sys.argv[2], 16)
want = sys.argv[3:]
def regs(ann):
    m = re.match(r'([vs])\[(\d+):(\d+)\]', ann)
    if m: return [f'{m.group(1)}{i}' for i in range(int(m.group(2)), int(m.group(3)) + 1)]
    return [ann]
last = {}
for line in open(ssa):
    m = re.match(r'\s*([0-9a-f]+)\s+(%\d+)\s*=.*;\s*(\S+)\s*$', line)
    if not m: continue
    a = int(m.group(1), 16)
    if a >= at: break
    for r in regs(m.group(3)): last[r] = (hex(a), m.group(2))
for r in want: print(r, 'last written before', hex(at), 'at', last.get(r, ('never (input)',))[0], last.get(r, ('', ''))[1] if r in last else '')
