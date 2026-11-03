// User-sized layout: drag the gutters between panes and panels (double-click one to reset it), and show one pane
// full screen. Findings width and inspector height are remembered; pane widths reset on every load.

const KEY = "afterpass.layout";
const MIN = 120;

function remembered(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "{}");
  } catch {
    return {};
  }
}

function remember(name: string, value: string | null) {
  try {
    const saved = remembered();
    if (value === null) delete saved[name];
    else saved[name] = value;
    localStorage.setItem(KEY, JSON.stringify(saved));
  } catch {
    // storage blocked: sizes last for this visit only
  }
}

/** Calls `move` with the pointer position while a gutter is dragged. */
function drag(gutter: HTMLElement, move: (e: PointerEvent) => void, reset: () => void) {
  gutter.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    gutter.setPointerCapture(e.pointerId);
    document.body.classList.add("resizing");
    const up = () => {
      gutter.removeEventListener("pointermove", move);
      document.body.classList.remove("resizing");
    };
    gutter.addEventListener("pointermove", move);
    gutter.addEventListener("pointerup", up, { once: true });
    gutter.addEventListener("pointercancel", up, { once: true });
  });
  gutter.addEventListener("dblclick", reset);
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** `entered` is called with a pane that has just gone full screen. */
export function initLayout(workspace: HTMLElement, panesEl: HTMLElement, entered: (pane: HTMLElement) => void) {
  const saved = remembered();
  for (const name of ["--findings-w", "--inspector-h"]) if (saved[name]) workspace.style.setProperty(name, saved[name]);

  // the findings panel (its left edge) and the inspector (its top edge)
  const panel = (sel: string, name: string, size: (box: DOMRect, e: PointerEvent) => number, max: (box: DOMRect) => number) => {
    drag(workspace.querySelector<HTMLElement>(`${sel} > .gutter`)!, (e) => {
      const box = workspace.getBoundingClientRect(), px = `${Math.round(clamp(size(box, e), MIN, max(box)))}px`;
      workspace.style.setProperty(name, px);
      remember(name, px);
    }, () => {
      workspace.style.removeProperty(name);
      remember(name, null);
    });
  };
  panel(".findings", "--findings-w", (b, e) => b.right - e.clientX, (b) => b.width - 2 * MIN);
  panel(".inspector", "--inspector-h", (b, e) => b.bottom - e.clientY, (b) => b.height - 2 * MIN);

  // each pane's right edge moves the border it shares with the next visible pane
  for (const pane of panesEl.querySelectorAll<HTMLElement>(".pane")) {
    drag(pane.querySelector<HTMLElement>(":scope > .gutter")!, (e) => {
      const shown = [...panesEl.querySelectorAll<HTMLElement>(".pane")].filter((p) => p.offsetWidth).sort((a, b) => a.offsetLeft - b.offsetLeft);
      const i = shown.indexOf(pane), next = shown[i + 1];
      if (!next) return;
      const left = pane.getBoundingClientRect().left, both = pane.offsetWidth + next.offsetWidth;
      const w = clamp(e.clientX - left, MIN, both - MIN);
      const widths = shown.map((p) => (p === pane ? w : p === next ? both - w : p.offsetWidth));
      panesEl.style.setProperty("--cols", widths.map((x) => `${x}fr`).join(" "));
    }, () => panesEl.style.removeProperty("--cols"));
    pane.querySelector(".full")!.addEventListener("click", () => {
      toggleFull(pane);
      if (pane.classList.contains("full")) entered(pane);
    });
  }
  document.addEventListener("fullscreenchange", () => {
    label();
    if (document.fullscreenElement instanceof HTMLElement) entered(document.fullscreenElement);
  });
}

/** A new set of panes starts evenly split. */
export function resetPanes(panesEl: HTMLElement) {
  panesEl.style.removeProperty("--cols");
}

const panesOf = () => document.querySelectorAll<HTMLElement>(".pane");

/** Each pane's button says whether it would enter or leave full screen. */
function label() {
  for (const p of panesOf()) p.querySelector(".full")!.textContent = document.fullscreenElement === p || p.classList.contains("full") ? "exit" : "full";
}

/** Show one pane full screen, or leave full screen. Uses the Fullscreen API where there is one (not on iPhone),
 * else covers the window. */
function toggleFull(pane: HTMLElement) {
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else if (pane.classList.contains("full")) {
    pane.classList.remove("full");
  } else if (pane.requestFullscreen && document.fullscreenEnabled) {
    pane.requestFullscreen().catch(() => pane.classList.add("full")).then(label);
  } else {
    pane.classList.add("full");
  }
  label();
}

/** Leaves the window-covering fallback; returns whether a pane was in it. */
export function exitFull(): boolean {
  const full = document.querySelector(".pane.full");
  full?.classList.remove("full");
  label();
  return !!full;
}
