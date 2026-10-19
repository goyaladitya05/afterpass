// Smoke test: open the built site, load every demo, open its first Finding, and fail on any page or console
// error (CSP violations included).
// usage: node scripts/smoke.mjs [url]   (default http://localhost:4173/, a running `vite preview`)

import { chromium } from "playwright-core";

const url = process.argv[2] ?? "http://localhost:4173/";
const browser = await chromium.launch({
  executablePath: process.env.CHROME ?? "/usr/bin/google-chrome",
  // CI machines have no GPU: Chrome's software renderer (SwiftShader); elsewhere hardware WebGL
  args: process.env.CI ? ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] : ["--use-angle=gl-egl", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage({ viewport: { width: 1400, height: 800 } });
const errors = [];
page.on("pageerror", (e) => errors.push(`page error: ${e.message}`));
// a refused load is logged as an error too; `refusing` names the one expected
let refusing = null;
page.on("console", (m) => m.type() === "error" && !(refusing && m.text().includes(refusing)) && errors.push(`console: ${m.text()}`));
await page.goto(url);
await page.waitForSelector("#demos button");
const demos = await page.locator("#demos button").count();
for (let i = 0; i < demos; i++) {
  await page.locator("#demos button").nth(i).click();
  await page.waitForFunction(() => document.querySelector("#loading").hidden && document.querySelector("#findings-body li[role=button]"), null, { timeout: 60000 });
  await page.locator("#findings-body li[role=button]").first().click();
  await page.waitForTimeout(300);
  const info = await page.evaluate(() => document.querySelector("#info").textContent);
  if (info.startsWith("error")) errors.push(`demo ${i}: ${info}`);
  console.log(`ok  ${await page.evaluate(() => document.querySelector("#crumbs").textContent)}`);
  await page.click("#home");
}

// benchmark_app output: the toy model with its execution graph and report, then the execution graph alone
const fixtures = new URL("../../core/tests/fixtures/", import.meta.url).pathname;
await page.setInputFiles("#file", ["toy.onnx", "toy.exec.xml", "toy.benchmark_report.csv"].map((f) => fixtures + f));
await page.waitForFunction(() => document.querySelector("#loading").hidden && document.querySelector("#findings-body li"), null, { timeout: 60000 });
const head = await page.evaluate(() => document.querySelector('[data-pane="1"] .pane-head').textContent);
if (!head.includes("compiled by OpenVINO") || !head.includes("imported from benchmark_app")) errors.push(`import: pane header ${head}`);
console.log(`ok  import: ${head}`);
refusing = "toy.exec.xml is an OpenVINO execution graph";
await page.setInputFiles("#file", [fixtures + "toy.exec.xml"]);
await page.waitForFunction(() => !document.querySelector("#load-error").hidden, null, { timeout: 60000 });
const refused = await page.evaluate(() => document.querySelector("#load-error").textContent);
if (!refused.includes("open it together with the model")) errors.push(`import: exec.xml alone gave ${refused}`);
console.log(`ok  refused: ${refused}`);
await browser.close();
if (errors.length || !demos) {
  console.error(errors.join("\n") || "no demos found");
  process.exit(1);
}
console.log(`${demos} demos loaded, no errors`);
