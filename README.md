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
| `benchmark.mjs` | `npm run benchmark` | Runs 17 NumPy benchmarks per variant and compares performance |
| `compare-size.mjs` | `npm run compare-size` | Compares compressed and uncompressed sizes of native `.so` modules in each wheel |

## Running

```bash
npm install
npm run compare-size
npm run benchmark
```

## Benchmarks

17 benchmarks from the [Pyodide benchmark suite](https://github.com/pyodide/pyodide/tree/main/benchmark/benchmarks/numpy_benchmarks) (originally from [serge-sans-paille/numpy-benchmarks](https://github.com/serge-sans-paille/numpy-benchmarks)):

`allpairs_distances`, `arc_distance`, `check_mask`, `create_grid`, `cronbach`, `diffusion`, `evolve`, `grayscott`, `harris`, `l2norm`, `log_likelihood`, `lstsqr`, `mandel`, `multiple_sum`, `periodic_dist`, `reverse_cumsum`, `rosen`

Each benchmark runs with 5 warmup iterations (discarded), then 10 timed repeats of 10 executions each. The min and max repeats are dropped and the remaining are averaged.

## CI

A GitHub Actions workflow runs on every push/PR to `main` (and on manual dispatch). It:

1. Compares native module sizes
2. Runs the full benchmark suite
3. Uploads `results.json`, `results.md`, and `size-results.md` as artifacts
4. Posts the benchmark results as a sticky PR comment

## Results

### Size

| | o2 | o3 | os | oz |
|---|---|---|---|---|
| Wheel (total) | 2.96 MB | 2.99 MB (+1.0%) | 2.81 MB (-5.1%) | 2.69 MB (-9.0%) |
| Native `.so` (uncompressed) | 5.82 MB | 5.94 MB (+2.0%) | 5.16 MB (-11.4%) | 4.71 MB (-19.2%) |
| Native `.so` (compressed) | 1.69 MB | 1.72 MB (+1.7%) | 1.54 MB (-8.7%) | 1.43 MB (-15.6%) |

### Performance

| Benchmark | o2 | o3 | os | oz |
|---|---|---|---|---|
| allpairs_distances | 747.2us | ⚪ 0.966x (-3.4%) | ⚪ 1.045x (+4.5%) | 🔴 1.110x (+11.0%) |
| arc_distance | 2.02ms | ⚪ 1.004x (+0.4%) | 🔴 1.105x (+10.5%) | 🔴 1.148x (+14.8%) |
| check_mask | 1.0147s | ⚪ 1.009x (+0.9%) | 🔴 1.067x (+6.7%) | 🔴 1.071x (+7.1%) |
| create_grid | 921.2us | ⚪ 0.973x (-2.7%) | ⚪ 1.034x (+3.4%) | ⚪ 1.022x (+2.2%) |
| cronbach | 5.20ms | ⚪ 0.999x (-0.1%) | 🔴 1.077x (+7.7%) | 🔴 1.097x (+9.7%) |
| diffusion | 28.64ms | ⚪ 1.008x (+0.8%) | ⚪ 1.024x (+2.4%) | 🔴 1.071x (+7.1%) |
| evolve | 33.77ms | ⚪ 1.002x (+0.2%) | 🔴 1.173x (+17.3%) | 🔴 1.090x (+9.0%) |
| grayscott | 47.10ms | ⚪ 0.993x (-0.7%) | ⚪ 1.015x (+1.5%) | ⚪ 1.045x (+4.5%) |
| harris | 20.79ms | ⚪ 1.012x (+1.2%) | 🟢 0.888x (-11.2%) | 🟢 0.879x (-12.1%) |
| l2norm | 1.97ms | ⚪ 1.003x (+0.3%) | ⚪ 0.998x (-0.2%) | ⚪ 1.012x (+1.2%) |
| log_likelihood | 6.90ms | ⚪ 1.050x (+5.0%) | 🔴 1.054x (+5.4%) | 🔴 1.112x (+11.2%) |
| lstsqr | 8.90ms | ⚪ 1.023x (+2.3%) | ⚪ 0.998x (-0.2%) | ⚪ 1.018x (+1.8%) |
| mandel | 44.37ms | ⚪ 1.013x (+1.3%) | ⚪ 0.998x (-0.2%) | ⚪ 0.993x (-0.7%) |
| multiple_sum | 11.69ms | ⚪ 1.003x (+0.3%) | ⚪ 0.990x (-1.0%) | ⚪ 1.000x (-0.0%) |
| periodic_dist | 5.74ms | ⚪ 0.988x (-1.2%) | ⚪ 1.006x (+0.6%) | ⚪ 1.035x (+3.5%) |
| reverse_cumsum | 20.04ms | ⚪ 0.980x (-2.0%) | ⚪ 1.034x (+3.4%) | 🔴 1.059x (+5.9%) |
| rosen | 21.12ms | ⚪ 0.988x (-1.2%) | ⚪ 0.997x (-0.3%) | ⚪ 1.019x (+1.9%) |
| **GEOMETRIC MEAN** | **1.000x (baseline)** | ⚪ **1.001x (+0.1%)** | ⚪ **1.028x (+2.8%)** | ⚪ **1.044x (+4.4%)** |

> 🟢 Faster than baseline | 🔴 Slower than baseline | ⚪ Within ±5%
> Ratio < 1 means faster. Times are wall-clock for 10 iterations.
