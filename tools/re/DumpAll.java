// Decompile every function into /g/out/decomp.c and write a string xref index to /g/out/strings.tsv.
import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.symbol.*;
import ghidra.program.model.data.StringDataInstance;
import java.io.*;

public class DumpAll extends GhidraScript {
    @Override
    public void run() throws Exception {
        new File("/g/out").mkdirs();
        try (PrintWriter s = new PrintWriter(new FileWriter("/g/out/strings.tsv"))) {
            for (Data d : currentProgram.getListing().getDefinedData(true)) {
                if (!d.hasStringValue()) continue;
                String v = StringDataInstance.getStringDataInstance(d).getStringValue();
                if (v == null) continue;
                for (Reference r : getReferencesTo(d.getAddress())) {
                    Function f = getFunctionContaining(r.getFromAddress());
                    s.println(d.getAddress() + "\t" + r.getFromAddress() + "\t" + (f == null ? "-" : f.getEntryPoint()) + "\t" + v.replace("\n", "\\n").replace("\t", "\\t"));
                }
            }
        }
        DecompInterface di = new DecompInterface();
        di.openProgram(currentProgram);
        try (PrintWriter w = new PrintWriter(new FileWriter("/g/out/decomp.c"))) {
            for (Function f : currentProgram.getFunctionManager().getFunctions(true)) {
                if (monitor.isCancelled()) break;
                DecompileResults r = di.decompileFunction(f, 60, monitor);
                w.println("// ==== " + f.getEntryPoint() + " " + f.getName());
                if (r != null && r.decompileCompleted()) w.println(r.getDecompiledFunction().getC());
                else w.println("// decompile failed");
            }
        }
    }
}
