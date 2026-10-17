// Load budget and frame rate measurement in a real Chrome with hardware WebGL.
// usage: npm run build && npx vite preview   (then, in another shell)
//        node scripts/bench.mjs [--flat] [--url http://localhost:4173/] <file> [file2]
// Files go through the file input (the real user path). Prints the app's load and bench lines.

import { chromium } from "playwright-core";

const args = process.argv.slice(2);
const flat = args.includes("--flat");
const u = args.indexOf("--url");
const base = u >= 0 ? args[u + 1] : "http://localhost:4173/";
const files = args.filter((a, i) => !a.startsWith("--") && (u < 0 || i !== u + 1));

const browser = await chromium.launch({
  executablePath: process.env.CHROME ?? "/usr/bin/google-chrome",
  args: ["--use-angle=gl-egl", "--ignore-gpu-blocklist"], // hardware WebGL in headless Chrome on Linux
});
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
page.on("pageerror", (e) => console.error("page error:", e.message));
page.on("console", (m) => {
  if (m.text().startsWith("[afterpass]") || m.type() === "error") console.log(m.text());
});
await page.goto(`${base}?bench=1${flat ? "&flat" : ""}`);
console.log("webgl:", await page.evaluate(() => {
  const gl = document.createElement("canvas").getContext("webgl2");
  const ext = gl?.getExtension("WEBGL_debug_renderer_info");
  return gl ? gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER) : "none";
}));
await page.setInputFiles("#file", files);
await page.waitForFunction(() => document.querySelector("#perf")?.textContent?.includes("bench:") || document.querySelector("#info")?.textContent?.startsWith("error"), null, { timeout: 180000 });
const info = await page.evaluate(() => document.querySelector("#info").textContent);
if (info.startsWith("error")) console.log(info);
await browser.close();
