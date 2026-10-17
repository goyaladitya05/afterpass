// Signed distance field glyph atlas for a monospace font, generated at startup with a 2D canvas.
// Distance transform: Felzenszwalb and Huttenlocher, the same approach as Mapbox TinySDF.

export const CHARS = Array.from({ length: 95 }, (_, i) => String.fromCharCode(32 + i)).join("") + "×…";

export interface Atlas {
  data: Uint8Array;
  width: number;
  height: number;
  /** atlas pixels per glyph cell side */
  cell: number;
  /** font size in atlas pixels */
  size: number;
  /** padding around each glyph inside its cell, atlas pixels */
  buffer: number;
  /** baseline offset from the top of a cell, atlas pixels */
  baseline: number;
  /** horizontal advance as a fraction of the font size */
  advance: number;
  index: Map<string, number>;
  cols: number;
}

const INF = 1e20;

function edt1d(grid: Float64Array, offset: number, stride: number, length: number, f: Float64Array, v: Uint16Array, z: Float64Array) {
  v[0] = 0;
  z[0] = -INF;
  z[1] = INF;
  for (let q = 0; q < length; q++) f[q] = grid[offset + q * stride];
  for (let q = 1, k = 0, s = 0; q < length; q++) {
    do {
      const r = v[k];
      s = (f[q] - f[r] + q * q - r * r) / (q - r) / 2;
    } while (s <= z[k] && --k > -1);
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = INF;
  }
  for (let q = 0, k = 0; q < length; q++) {
    while (z[k + 1] < q) k++;
    const r = v[k];
    grid[offset + q * stride] = f[r] + (q - r) * (q - r);
  }
}

function edt(grid: Float64Array, w: number, h: number) {
  const n = Math.max(w, h);
  const f = new Float64Array(n), z = new Float64Array(n + 1), v = new Uint16Array(n);
  for (let x = 0; x < w; x++) edt1d(grid, x, w, h, f, v, z);
  for (let y = 0; y < h; y++) edt1d(grid, y * w, 1, w, f, v, z);
}

export function buildAtlas(fontFamily: string, size = 40, buffer = 6, radius = 8): Atlas {
  const cell = size + buffer * 2;
  const cols = 16;
  const rows = Math.ceil(CHARS.length / cols);
  const width = cols * cell, height = rows * cell;
  const data = new Uint8Array(width * height);
  const canvas = new OffscreenCanvas(cell, cell);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.font = `${size}px ${fontFamily}`;
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "#000";
  const baseline = buffer + Math.round(size * 0.8);
  const advance = ctx.measureText("M").width / size;
  const outer = new Float64Array(cell * cell), inner = new Float64Array(cell * cell);
  const index = new Map<string, number>();
  [...CHARS].forEach((ch, i) => {
    index.set(ch, i);
    ctx.clearRect(0, 0, cell, cell);
    ctx.fillText(ch, buffer, baseline);
    const alpha = ctx.getImageData(0, 0, cell, cell).data;
    for (let p = 0; p < cell * cell; p++) {
      const a = alpha[p * 4 + 3] / 255;
      outer[p] = a === 1 ? 0 : a === 0 ? INF : Math.max(0, 0.5 - a) ** 2;
      inner[p] = a === 1 ? INF : a === 0 ? 0 : Math.max(0, a - 0.5) ** 2;
    }
    edt(outer, cell, cell);
    edt(inner, cell, cell);
    const ox = (i % cols) * cell, oy = Math.floor(i / cols) * cell;
    for (let y = 0; y < cell; y++) {
      for (let x = 0; x < cell; x++) {
        const p = y * cell + x;
        const d = Math.sqrt(outer[p]) - Math.sqrt(inner[p]);
        data[(oy + y) * width + ox + x] = Math.max(0, Math.min(255, Math.round(255 - 255 * (d / radius + 0.25))));
      }
    }
  });
  return { data, width, height, cell, size, buffer, baseline, advance, index, cols };
}
