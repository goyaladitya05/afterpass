// Chrome effects for the landing page and buttons: label scramble, loader, ticker, ASCII texture.
// All honour prefers-reduced-motion.

const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const TOKENS = "fuse lower tile emit pass kernel reorder fp16 int8 bf16 nchw nhwc %0 %1 -> => :: {} [] () 0x1f gemm conv mvn rope sdpa x4 x12 x64 ; | / * + -".split(" ");

let seed = 7;
const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const noise = (s: string) => [...s].map((c) => (c === " " ? " " : GLYPHS[Math.floor(Math.random() * GLYPHS.length)])).join("");

/** Button labels resolve left to right from random characters on hover and focus (340 ms). */
export function scramble(root: ParentNode) {
  root.querySelectorAll<HTMLElement>("[data-label]").forEach((el) => {
    const host = el.closest("button") ?? el;
    let raf = 0;
    const run = () => {
      if (reduced()) return;
      cancelAnimationFrame(raf);
      const text = el.dataset.label!, start = performance.now();
      const frame = (now: number) => {
        const t = Math.min(1, (now - start) / 340);
        const solid = Math.floor(text.length * (1 - 2 ** (-10 * t)));
        el.textContent = text.slice(0, solid) + noise(text.slice(solid));
        if (t < 1) raf = requestAnimationFrame(frame);
        else el.textContent = text;
      };
      raf = requestAnimationFrame(frame);
    };
    host.addEventListener("pointerenter", run);
    host.addEventListener("focus", run);
    host.addEventListener("pointerleave", () => { cancelAnimationFrame(raf); el.textContent = el.dataset.label!; });
  });
}

/** Token soup behind the landing page. */
export function asciiTexture(pre: HTMLElement, rows = 80, cols = 260) {
  pre.textContent = Array.from({ length: rows }, () => {
    let line = "";
    while (line.length < cols) line += (rand() < 0.12 ? "    " : TOKENS[Math.floor(rand() * TOKENS.length)]) + " ";
    return line;
  }).join("\n");
}

export function ticker(track: HTMLElement, words: string[]) {
  const html = (copy: boolean) => words.map((w) => `<span${copy ? ' aria-hidden="true"' : ""}>${w}</span>`).join("");
  track.innerHTML = html(false) + html(true); // duplicated so translateX(-50%) loops seamlessly
}

/** Short loading screen: rows of compiler tokens resolve into text, then fade. Skipped on any input. */
export function loader(el: HTMLElement) {
  const pre = el.querySelector("pre")!;
  const finish = () => {
    el.classList.add("done");
    setTimeout(() => el.remove(), 360);
  };
  if (reduced()) {
    el.remove();
    return;
  }
  const width = Math.min(96, Math.floor(innerWidth / 6.2));
  const pinned = new Map([[5, "afterpass"], [7, "source graph  ->  compiled graph"], [9, "every kernel traced to its ops"]]);
  const rows = Array.from({ length: 15 }, (_, r) => {
    const text = pinned.get(r);
    if (text) return text.padStart(Math.floor((width + text.length) / 2)).padEnd(width);
    let line = "";
    while (line.length < width) line += TOKENS[Math.floor(rand() * TOKENS.length)] + " ";
    return line.slice(0, width);
  });
  const start = performance.now(), total = 1600;
  let done = false;
  const stop = () => {
    if (done) return;
    done = true;
    for (const ev of ["pointerdown", "keydown", "wheel"]) removeEventListener(ev, stop);
    finish();
  };
  for (const ev of ["pointerdown", "keydown", "wheel"]) addEventListener(ev, stop, { once: true });
  const frame = (now: number) => {
    if (done) return;
    const t = now - start;
    pre.innerHTML = "";
    rows.forEach((row, r) => {
      const local = Math.max(0, Math.min(1, (t - r * 30) / 600));
      const shown = Math.floor(row.length * (1 - 2 ** (-10 * local)));
      const settle = Math.max(0, Math.min(1, (t - r * 30 - 300) / 500));
      const solid = Math.floor(shown * settle);
      const line = row.slice(0, solid) + noise(row.slice(solid, shown));
      const node = pinned.has(r) ? document.createElement("b") : document.createTextNode("");
      node.textContent = line + "\n";
      pre.appendChild(node);
    });
    if (t < total) requestAnimationFrame(frame);
    else stop();
  };
  requestAnimationFrame(frame);
}
