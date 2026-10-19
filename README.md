# afterpass

See what a deployment compiler did to your model. afterpass puts the source graph next to the compiled graph and traces every fused kernel back to the source ops it came from: Compiler Explorer, for model graphs.

It answers five questions:

1. **Did my fusion happen?** Select source ops and see which compiled kernels they landed in.
2. **Where did the time go?** Kernels coloured by measured time, and a ranked list of time sinks.
3. **Where did precision or layout change?** fp32 islands in an int8 model, inserted reorders.
4. **Why is runtime A slower than B?** Two compiled graphs of one model, compared through the shared source graph.
5. **What changed between runtime versions?** Fusions, kernels and times, diffed.

Runtimes: OpenVINO (CPU, GPU, NPU) and TensorRT (10 and 11).

## Open a model

The viewer is a static web page. Everything runs in your browser; nothing is uploaded, and the page's Content Security Policy forbids it from sending data anywhere. It reads files and never compiles anything: a compiled graph always comes from `afterpass capture`, run on a machine with the runtime and the hardware. Drop in one of these:

* an ONNX model or an OpenVINO IR `.xml`, to browse its graph;
* a bundle from `afterpass capture` (below), to see the compiled graph, provenance and timing;
* two bundles of the same model, to compare runtimes, devices or versions.

Weights are never read, so a 30B-parameter model opens in well under a second.

## Capture a bundle

A bundle records what a runtime compiled your model into, with per-kernel times. Capture it on the machine that has the runtime:

```
uv tool install "git+https://github.com/goyaladitya05/afterpass#subdirectory=capture"
afterpass capture model.onnx --runtime openvino --device CPU
afterpass capture model.onnx --runtime tensorrt --device GPU     # needs the tensorrt extra, see below
```

For TensorRT, install with the extra: `uv tool install "afterpass[tensorrt] @ git+https://github.com/goyaladitya05/afterpass#subdirectory=capture"`.

Useful options:
* `--shape name=d0,d1,...` fixes a dynamic input. For LLM IRs, pass `input_ids`, `attention_mask`, `position_ids` and `beam_idx`.
* `--iterations` and `--warmup` set how many runs are timed and how many are discarded first.

A bundle is gzip JSON, typically 20 to 350 KB. It holds:
* the source topology (never weights);
* the compiled graph;
* the profile;
* the provenance links;
* the machine description.

## Check for regressions in CI

`afterpass-diff` compares two bundles of the same model (the viewer's compare mode, as text) and can fail a build:

```
cargo install --git https://github.com/goyaladitya05/afterpass afterpass-core --features cli
afterpass-diff baseline.afterpass.json.gz new.afterpass.json.gz --max-slower 10 --fail-on-lost-fusion
```

* `--max-slower PCT` fails (exit 1) when the second bundle's median latency is more than PCT% above the first's, unless the gap is run-to-run noise: the second bundle's fastest run is no slower than the first's median.
* `--fail-on-lost-fusion` fails when the second bundle runs more kernels than the first for some region of source ops.
* `--json` prints the findings and the gate results as JSON.
* Exit 2 means a usage or input error (one line on stderr naming the file).

A GitHub Actions step, after your job has captured `new.afterpass.json.gz` and fetched the baseline:

```
- run: |
    cargo install --git https://github.com/goyaladitya05/afterpass afterpass-core --features cli
    afterpass-diff baseline.afterpass.json.gz new.afterpass.json.gz --max-slower 10 --fail-on-lost-fusion
```

## How sure is a link?

Every link from a compiled kernel to a source op has a grade:
* **explicit:** the runtime named the op;
* **inferred:** matched by name or by graph structure;
* **unknown.**

Inferred links are never shown as certain. When several neighbouring kernels have all lost their names, an inferred link means "computed in one of these kernels".

## Build from source

```
cd core && cargo test --release                     # Rust core: parsers, collapsing, layout, analysis
cd core && cargo test --release --features cli      # also the afterpass-diff tests
cd web && npm ci && npm run wasm && npm run build   # needs wasm-pack and the wasm32-unknown-unknown target
cd web && npx vite preview                          # serve the built site
cd capture && uv run --group dev pytest tests       # capture CLI tests (OpenVINO CPU)
```

Layout:
* `capture/`: the Python capture CLI.
* `core/`: Rust compiled to WebAssembly. It parses, collapses repeated blocks, lays out, and analyses.
* `web/`: the TypeScript and WebGL2 viewer.

## Licence

Apache-2.0, see `LICENSE`. The bundled fonts, Source Code Pro and Inter Tight, are under the SIL Open Font License; the licence texts sit next to them in `web/public/fonts/`.
