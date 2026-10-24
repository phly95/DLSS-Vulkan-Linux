import re,subprocess,sys
d=open('decomp.c').read()
pairs=re.findall(r'FUN_180153bd0\(\w+,&(PTR_\w+|DAT_\w+),\s*"(_Z\w+)"',d)
dem=subprocess.run(['c++filt'],input='\n'.join(n for _,n in pairs),capture_output=True,text=True).stdout.split('\n')
m={h:dem[i] for i,(h,_) in enumerate(pairs)}
print(len(m),file=sys.stderr)
open('handles.tsv','w').write(''.join(f'{h}\t{n}\n' for h,n in m.items()))
pat=re.compile('|'.join(re.escape(h)+r'\b' for h in sorted(m,key=len,reverse=True)))
open('decomp_ann.c','w').write(pat.sub(lambda x:x.group(0)+'/*K:'+m[x.group(0)]+'*/',d))
