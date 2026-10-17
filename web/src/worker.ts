// Hosts the wasm core. Files are read here: ONNX through ranged FileReaderSync reads (weights are skipped,
// never copied), OpenVINO IR .xml whole, bundles through the browser's gzip DecompressionStream.

import init, { Session } from "../pkg/afterpass_core.js";

export type Request =
  | { id: number; op: "load"; files: File[]; flat?: boolean }
  | { id: number; op: "toggle" | "step" | "select"; pane: number; box: number; delta?: number };

let session: Session | null = null;
const wasm = init();

async function open(file: File): Promise<Session> {
  const name = file.name.toLowerCase();
  if (name.endsWith(".onnx")) {
    const reader = new FileReaderSync();
    const read = (offset: number, length: number) => new Uint8Array(reader.readAsArrayBuffer(file.slice(offset, offset + length)));
    return Session.load_onnx(read, file.size);
  }
  if (name.endsWith(".xml")) return Session.load_ir(new Uint8Array(await file.arrayBuffer()));
  let bytes = new Uint8Array(await file.arrayBuffer());
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  return Session.load_bundle(bytes);
}

function scene(pane: number) {
  const s = session!.scene(pane) as Record<string, unknown>;
  return { scene: s, transfer: ["boxes", "kinds", "flags", "times", "segs"].map((k) => (s[k] as ArrayBufferView).buffer as ArrayBuffer) };
}

async function handle(req: Request): Promise<[unknown, Transferable[]]> {
  const memory = (await wasm).memory;
  if (req.op === "load") {
    const t0 = performance.now();
    const next = await open(req.files[0]);
    if (req.files[1]) next.attach(await open(req.files[1]));
    if (req.flat) next.flatten();
    session?.free();
    session = next;
    const panes = session.panes();
    const scenes = [], transfer: Transferable[] = [];
    for (let p = 0; p < panes; p++) {
      const s = scene(p);
      scenes.push(s.scene);
      transfer.push(...s.transfer);
    }
    const nodes = Array.from({ length: panes }, (_, p) => session!.node_count(p));
    return [{ scenes, nodes, label: session.label(), latency: session.latency_us(), ms: performance.now() - t0, wasmBytes: memory.buffer.byteLength }, transfer];
  }
  if (!session) throw new Error("nothing loaded");
  if (req.op === "select") {
    const linked = session.linked(req.pane, req.box) as { boxes?: Uint32Array; grades?: Uint8Array };
    return [{ boxes: linked.boxes ?? new Uint32Array(), grades: linked.grades ?? new Uint8Array(), text: session.describe(req.pane, req.box) }, []];
  }
  const anchor = req.op === "toggle" ? session.toggle(req.pane, req.box) : session.step(req.pane, req.box, req.delta ?? 1);
  if (anchor < 0) return [{ anchor }, []];
  const s = scene(req.pane);
  return [{ anchor, scene: s.scene, wasmBytes: memory.buffer.byteLength }, s.transfer];
}

self.onmessage = async (e: MessageEvent<Request>) => {
  try {
    const [result, transfer] = await handle(e.data);
    self.postMessage({ id: e.data.id, result }, { transfer });
  } catch (err) {
    self.postMessage({ id: e.data.id, error: String(err instanceof Error ? err.message : err) });
  }
};
