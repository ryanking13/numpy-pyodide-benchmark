# NumPy Pyodide Benchmark

Benchmarks comparing NumPy performance across different compiler optimization flags when running in [Pyodide](https://pyodide.org/) (WebAssembly) on Node.js.

## Variants

Each variant is a NumPy 2.2.5 wheel compiled for Pyodide with a different optimization flag:

| Variant | Flag | Description |
|---------|------|-------------|
| `o2` | `-O2` | Moderate optimization (baseline) |
| `o3` | `-O3` | Aggressive optimization |
| `os` | `-Os` | Optimize for size |
| `oz` | `-Oz` | Aggressively optimize for size |

## Scripts

| Script | Command | Description |
|--------|---------|-------------|
| `benchmark.mjs` | `npm run benchmark` | Runs 14 NumPy benchmarks per variant and compares performance |
| `compare-size.mjs` | `npm run compare-size` | Compares compressed and uncompressed sizes of native `.so` modules in each wheel |

## Running

```bash
npm install
npm run compare-size
npm run benchmark
```

## Benchmarks

14 benchmarks from the [Pyodide benchmark suite](https://github.com/pyodide/pyodide/tree/main/benchmark/benchmarks/numpy_benchmarks) (originally from [serge-sans-paille/numpy-benchmarks](https://github.com/serge-sans-paille/numpy-benchmarks)):

`allpairs_distances`, `arc_distance`, `check_mask`, `create_grid`, `cronbach`, `diffusion`, `grayscott`, `l2norm`, `log_likelihood`, `mandel`, `multiple_sum`, `periodic_dist`, `reverse_cumsum`, `rosen`

3 benchmarks were removed after variance analysis (run 3× and measured cross-run coefficient of variation):
- `evolve` — `np.roll` creates 4 full 512×512 copies per call, GC-sensitive (ratio CoV > 6%)
- `harris` — creates many temporary 512×512 sliced arrays, GC-sensitive (ratio CoV > 6%)
- `lstsqr` — baseline raw time unstable at 500K elements (CoV > 5%)

Each benchmark runs with 5 warmup iterations (discarded), then 10 timed repeats of 10 executions each. The min and max repeats are dropped and the remaining are averaged.

## CI

A GitHub Actions workflow runs on every push/PR to `main` (and on manual dispatch). It:

1. Compares native module sizes
2. Runs the full benchmark suite
3. Uploads `results.json`, `results.md`, and `size-results.md` as artifacts
4. Posts the benchmark results as a sticky PR comment

## Output

After running, the scripts generate:

| File | Description |
|------|-------------|
| `results.json` | Raw benchmark timings (structured JSON) |
| `results.md` | Performance comparison table (Markdown) |
| `size-results.md` | Native module size comparison (Markdown) |
