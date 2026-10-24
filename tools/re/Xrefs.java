// For every address in /g/out/handles.tsv (Ghidra label names), print every reference to it, and
// for each referencing function, its callers (two levels). Output /g/out/xrefs.tsv.
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.symbol.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class Xrefs extends GhidraScript {
    PrintWriter w;
    String fn(Address a) { Function f = getFunctionContaining(a); return f == null ? "-" : f.getEntryPoint().toString(); }
    void refs(String tag, Address a, int depth) {
        for (Reference r : getReferencesTo(a)) {
            Address from = r.getFromAddress();
            w.println(tag + "\t" + depth + "\t" + a + "\t" + from + "\t" + fn(from) + "\t" + r.getReferenceType());
            if (depth < 3) {
                Function f = getFunctionContaining(from);
                if (f != null) refs(tag, f.getEntryPoint(), depth + 1);
                else refs(tag, from, depth + 1);   // data table: who reads the table
            }
        }
    }
    @Override
    public void run() throws Exception {
        w = new PrintWriter(new FileWriter("/g/out/xrefs.tsv"));
        for (String line : Files.readAllLines(Paths.get("/g/out/handles.tsv"))) {
            String[] p = line.split("\t");
            String hex = p[0].substring(p[0].lastIndexOf('_') + 1);
            refs(p[1], toAddr(Long.parseUnsignedLong(hex, 16)), 0);
        }
        w.close();
    }
}
