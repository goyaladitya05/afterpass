// Findings and Inspector panels. Built with DOM nodes and textContent only: names come from model files.

export interface Finding {
  section: string;
  severity: "info" | "warn";
  title: string;
  detail: string;
  /** [pane, node or group id, 1 when a group] */
  targets: [number, number, number][];
}

type Child = Node | string | null | undefined | false;

export function h(tag: string, attrs: Record<string, string> = {}, ...children: Child[]): HTMLElement {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  for (const c of children) if (c !== null && c !== undefined && c !== false) el.append(c);
  return el;
}

const pct = (share: number) => (share < 0.001 ? "<0.1%" : `${(share * 100).toFixed(1)}%`);

export const us = (t: number | null | undefined) =>
  t == null ? "" : t >= 1000 ? `${(t / 1000).toFixed(2)} ms` : `${t.toFixed(t < 10 ? 1 : 0)} us`;

const SECTIONS: [string, string][] = [
  ["time", "time sinks"],
  ["unaccounted", "unaccounted source ops"],
  ["inserted", "kernels without a source op"],
  ["precision", "precision fallbacks"],
  ["outliers", "outlier blocks"],
];

/** Render Findings; `pick(item, k)` is called with the k-th target when an item is clicked repeatedly. */
export function renderFindings(body: HTMLElement, items: Finding[], pick: (f: Finding, k: number) => void) {
  body.replaceChildren();
  if (!items.length) {
    body.append(h("p", { class: "none" }, "No findings for this graph."));
    return;
  }
  SECTIONS.forEach(([key, title], i) => {
    const list = items.filter((f) => f.section === key);
    if (!list.length) return;
    const ul = h("ul");
    for (const f of list) {
      let k = 0;
      const count = f.targets.length > 1 ? h("em", {}, `${f.targets.length} nodes`) : null;
      const li = h("li", { class: `${f.severity}${f.targets.length ? "" : " idle"}`, title: f.detail },
        h("div", { class: "t" }, h("span", {}, f.title), count), h("div", { class: "d" }, f.detail));
      if (f.targets.length) {
        li.addEventListener("click", () => {
          pick(f, k);
          if (count) count.textContent = `${k + 1}/${f.targets.length}`;
          k = (k + 1) % f.targets.length;
        });
      }
      ul.append(li);
    }
    body.append(h("h3", {}, h("span", { class: "num" }, String(i + 1).padStart(2, "0")), title), ul);
  });
}

interface Tensor { name: string; dtype: string; shape: string | null; layout: string; from: string | null; constant: boolean }
interface Inspect {
  pane: number; kind: string; count: number; time_us: number | null; share: number | null;
  links: { node: number; name: string; op: string; grade: string; via: string; time_us: number | null }[];
  link_count: number;
  node?: { id: number; name: string; op: string; kernel: string; precision: string; time_us: number | null;
    attrs: [string, string][]; inputs: (Tensor | null)[]; outputs: Tensor[]; claims: string[] };
}

function tensors(title: string, list: (Tensor | null)[]) {
  const rows = list.filter((t): t is Tensor => !!t).map((t) =>
    h("tr", {}, h("td", {}, t.dtype), h("td", {}, t.shape ?? "?"), h("td", {}, t.layout), h("td", { class: "wrap" }, t.constant ? `${t.name} (weight)` : t.from ? `${t.name} <- ${t.from}` : t.name)));
  if (!rows.length) return null;
  return h("table", {}, h("tr", {}, h("th", {}, title), h("th", {}, "shape"), h("th", {}, "layout"), h("th", {}, "tensor")), ...rows);
}

export function renderInspector(main: HTMLElement, side: HTMLElement, meta: HTMLElement, d: Inspect, paneNames: string[],
  title: string, follow: (pane: number, node: number) => void) {
  const other = paneNames[1 - d.pane] ?? "";
  meta.textContent = `${paneNames[d.pane]} ${d.kind}`;
  main.replaceChildren();
  const n = d.node;
  if (n) {
    const kv: Child[] = [];
    const add = (k: string, v: string | null | undefined) => { if (v) kv.push(h("span", {}, k), h("span", {}, v)); };
    add("op", n.op);
    add("kernel", n.kernel);
    add("precision", n.precision);
    add("time", us(n.time_us) + (d.share ? `  (${pct(d.share)} of latency)` : ""));
    if (n.claims.length) add("claims", n.claims.join(", "));
    main.append(h("h4", {}, n.name || `(${n.op})`), h("div", { class: "kv" }, ...kv));
    const ins = tensors("in", n.inputs), outs = tensors("out", n.outputs);
    if (ins) main.append(ins);
    if (outs) main.append(outs);
    if (n.attrs.length) main.append(h("div", { class: "kv" }, ...n.attrs.flatMap(([k, v]) => [h("span", {}, k), h("span", {}, v)])));
  } else {
    const time = d.time_us ? `  ${us(d.time_us)}${d.share ? ` (${pct(d.share)} of latency)` : ""}` : "";
    main.append(h("h4", {}, title), h("p", { class: "hint" }, `${d.count} ops${time}`));
  }
  side.replaceChildren();
  if (paneNames.length < 2) return;
  const head = h("p", { class: "label" }, d.link_count ? `provenance: ${d.link_count} in ${other}` : `no ${other} node linked`);
  const ul = h("ul", { class: "links" });
  for (const l of d.links) {
    const li = h("li", { title: `${l.grade}${l.via ? ` via ${l.via}` : ""}` },
      h("span", { class: `chip ${l.grade}` }, l.via === "structure" ? "struct" : l.grade),
      h("span", {}, `${l.op}  ${l.name}`), h("span", { class: "faint" }, us(l.time_us)));
    li.addEventListener("click", () => follow(1 - d.pane, l.node));
    ul.append(li);
  }
  side.append(head, ul);
  if (d.link_count > d.links.length) side.append(h("p", { class: "faint" }, `${d.link_count - d.links.length} more`));
}
