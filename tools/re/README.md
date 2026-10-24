# tools/re: reading NeuralHub's host DLL

The host program (`neuralhub_host.dll`, out of the installer by `tools/exe/extract.py`) holds what the
code objects do not: the order of kernel launches, the parameter structs and the weight packing.
These scripts decompile it with Ghidra (headless, in podman) so the port can follow it.
Nothing they produce may be committed: it is NeuralHub's code.

    # once: Ghidra 12.1.4 under build/ghidra/, a JDK 21 image localhost/ghidra:1 (see docs/sr/HOST-RE.md)
    podman run --rm -v $PWD/build/ghidra:/g -v $PWD/build/nh:/nh:ro localhost/ghidra:1 \
        /g/ghidra_12.1.4_PUBLIC/support/analyzeHeadless /g/proj nh -import /nh/neuralhub_host.dll
    # every function decompiled to build/ghidra/out/decomp.c, string xrefs to strings.tsv
    ... analyzeHeadless /g/proj nh -process neuralhub_host.dll -noanalysis -scriptPath /g/scripts -postScript DumpAll.java
    # kernel handles (the `&PTR_...` passed to __hipRegisterFunction) named in the decompile
    (cd build/ghidra/out && python3 ../../../tools/re/annotate.py)
    # who launches which kernel, three levels up
    ... -postScript Xrefs.java

- `DumpAll.java`: decompile every function; string -> referencing function index.
- `annotate.py`: map each kernel handle to its demangled name and annotate `decomp.c` with `/*K:name*/`.
- `Xrefs.java`: references to each kernel handle and their callers.
- `pe.py`: read the DLL's memory image by virtual address (tables, constants, strings).
