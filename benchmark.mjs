import { loadPyodide } from "pyodide";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const WHEEL_DIR = path.join(__dirname, "numpy-wheels");
const VARIANTS = fs
  .readdirSync(WHEEL_DIR)
  .filter((d) => fs.statSync(path.join(WHEEL_DIR, d)).isDirectory())
  .sort();

const BASELINE = "o2";
const WARMUP = 5; // warmup iterations (discarded) to stabilize JIT/wasm
const REPEAT = 10; // timeit repeat count
const NUMBER = 10; // timeit number count (executions per repeat)

// ---------------------------------------------------------------------------
// Benchmark definitions (from pyodide/benchmark/benchmarks/numpy_benchmarks)
//
// Each entry: { name, setup, run, code }
//   – setup/run come from the # setup: / # run: comments
//   – code is the function body
//   – fft is skipped (uses wrong API np.fft vs np.fft.fft)
//   – hyantes is skipped (extremely slow triple-nested Python loop)
// ---------------------------------------------------------------------------
const BENCHMARKS = [
  {
    name: "allpairs_distances",
    setup:
      "import numpy as np ; N = 50 ; X, Y = np.random.randn(100,N), np.random.randn(40,N)",
    run: "allpairs_distances(X, Y)",
    code: `
import numpy as np

def allpairs_distances(A, B):
    A2 = np.einsum("ij,ij->i", A, A)
    B2 = np.einsum("ij,ij->i", B, B)
    return A2[:, None] + B2[None, :] - 2 * np.dot(A, B.T)
`,
  },
  {
    name: "arc_distance",
    setup:
      "import numpy as np ; N = 5000 ; t0, p0, t1, p1 = np.random.randn(N), np.random.randn(N), np.random.randn(N), np.random.randn(N)",
    run: "arc_distance(t0, p0, t1, p1)",
    code: `
import numpy as np

def arc_distance(theta_1, phi_1, theta_2, phi_2):
    temp = (
        np.sin((theta_2 - theta_1) / 2) ** 2
        + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2) ** 2
    )
    distance_matrix = 2 * (np.arctan2(np.sqrt(temp), np.sqrt(1 - temp)))
    return distance_matrix
`,
  },
  {
    name: "check_mask",
    setup:
      "import numpy as np ; N = 100000 ; groups = np.random.randint(0, 1000, size=N) ; data = np.random.rand(N) ; mask = np.random.randint(0, 2, size=N).astype(bool)",
    run: "check_mask(groups, data, mask)",
    code: `
import numpy as np

def check_mask(groups, data, mask):
    unique_groups = np.unique(groups)
    result = np.zeros(len(unique_groups))
    for i, g in enumerate(unique_groups):
        group_mask = groups == g
        combined = group_mask & mask
        if combined.any():
            result[i] = data[combined].mean()
    return result
`,
  },
  {
    name: "create_grid",
    setup: "import numpy as np ; N = 200",
    run: "create_grid(N)",
    code: `
import numpy as np

def create_grid(N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    return np.sqrt(X**2 + Y**2)
`,
  },
  {
    name: "cronbach",
    setup: "import numpy as np ; N = 600 ; items = np.random.rand(N,N)",
    run: "cronbach(items)",
    code: `
def cronbach(itemscores):
    itemvars = itemscores.var(axis=1, ddof=1)
    tscores = itemscores.sum(axis=0)
    nitems = len(itemscores)
    return nitems / (nitems - 1) * (1 - itemvars.sum() / tscores.var(ddof=1))
`,
  },
  {
    name: "diffusion",
    setup:
      "import numpy as np;lx,ly=(2**6,2**6);u=np.zeros([lx,ly],dtype=np.double);u[lx//2,ly//2]=1000.0;tempU=np.zeros([lx,ly],dtype=np.double)",
    run: "diffusion(u,tempU,100)",
    code: `
def diffusion(u, tempU, iterNum):
    mu = 0.1
    for _ in range(iterNum):
        tempU[1:-1, 1:-1] = u[1:-1, 1:-1] + mu * (
            u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1]
            + u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2]
        )
        u[:, :] = tempU[:, :]
        tempU[:, :] = 0.0
`,
  },
  {
    name: "evolve",
    setup:
      "import numpy as np ; grid_shape = (512, 512) ; grid = np.zeros(grid_shape) ; block_low = int(grid_shape[0] * .4) ; block_high = int(grid_shape[0] * .5) ; grid[block_low:block_high, block_low:block_high] = 0.005",
    run: "evolve(grid, 0.1)",
    code: `
import numpy as np

def laplacian(grid):
    return (
        np.roll(grid, +1, 0) + np.roll(grid, -1, 0)
        + np.roll(grid, +1, 1) + np.roll(grid, -1, 1)
        - 4 * grid
    )

def evolve(grid, dt, D=1):
    return grid + dt * D * laplacian(grid)
`,
  },
  {
    name: "grayscott",
    setup: "pass",
    run: "grayscott(40, 0.16, 0.08, 0.04, 0.06)",
    code: `
import numpy as np

def grayscott(counts, Du, Dv, F, k):
    n = 100
    U = np.zeros((n + 2, n + 2), dtype=np.float32)
    V = np.zeros((n + 2, n + 2), dtype=np.float32)
    u, v = U[1:-1, 1:-1], V[1:-1, 1:-1]
    r = 20
    u[:] = 1.0
    U[n // 2 - r : n // 2 + r, n // 2 - r : n // 2 + r] = 0.50
    V[n // 2 - r : n // 2 + r, n // 2 - r : n // 2 + r] = 0.25
    u += 0.15 * np.random.random((n, n))
    v += 0.15 * np.random.random((n, n))
    for _ in range(counts):
        Lu = U[0:-2, 1:-1] + U[1:-1, 0:-2] - 4*U[1:-1, 1:-1] + U[1:-1, 2:] + U[2:, 1:-1]
        Lv = V[0:-2, 1:-1] + V[1:-1, 0:-2] - 4*V[1:-1, 1:-1] + V[1:-1, 2:] + V[2:, 1:-1]
        uvv = u * v * v
        u += Du * Lu - uvv + F * (1 - u)
        v += Dv * Lv + uvv - (F + k) * v
    return V
`,
  },
  {
    name: "harris",
    setup:
      "import numpy as np ; M, N = 512, 512 ; X = np.random.randn(M,N)",
    run: "harris(X)",
    code: `
def harris(X):
    m, n = X.shape
    dx = (X[1:, :] - X[:m-1, :])[:, 1:]
    dy = (X[:, 1:] - X[:, :n-1])[1:, :]
    A = dx * dx
    B = dy * dy
    C = dx * dy
    tr = A + B
    det = A * B - C * C
    k = 0.05
    return det - k * tr * tr
`,
  },
  {
    name: "l2norm",
    setup: "import numpy as np ; N = 1000; x = np.random.rand(N,N)",
    run: "l2norm(x)",
    code: `
import numpy as np

def l2norm(x):
    return np.sqrt(np.einsum("ij,ij->i", x, x))
`,
  },
  {
    name: "log_likelihood",
    setup:
      "import numpy as np ; N = 100000 ; a = np.random.random(N); b = 0.1; c =1.1",
    run: "log_likelihood(a, b, c)",
    code: `
import numpy

def log_likelihood(data, mean, sigma):
    s = (data - mean) ** 2 / (2 * (sigma**2))
    pdfs = numpy.exp(-s)
    pdfs /= numpy.sqrt(2 * numpy.pi) * sigma
    return numpy.log(pdfs).sum()
`,
  },
  {
    name: "lstsqr",
    setup:
      "import numpy as np ; N = 500000 ; X, Y = np.random.rand(N), np.random.rand(N)",
    run: "lstsqr(X, Y)",
    code: `
import numpy as np

def lstsqr(x, y):
    x_avg = np.average(x)
    y_avg = np.average(y)
    dx = x - x_avg
    var_x = np.sum(dx**2)
    cov_xy = np.sum(dx * (y - y_avg))
    slope = cov_xy / var_x
    y_interc = y_avg - slope * x_avg
    return (slope, y_interc)
`,
  },
  {
    name: "mandel",
    setup:
      "import numpy as np; image = np.zeros((64, 32), dtype = np.uint8)",
    run: "mandel(-2.0, 1.0, -1.0, 1.0, image, 20)",
    code: `
def kernel(x, y, max_iters):
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i
    return max_iters

def mandel(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = kernel(real, imag, iters)
            image[y, x] = color
`,
  },
  {
    name: "multiple_sum",
    setup: "import numpy as np ; r = np.random.rand(100,100)",
    run: "multiple_sum(r)",
    code: `
import numpy as np

def multiple_sum(array):
    rows = array.shape[0]
    cols = array.shape[1]
    out = np.zeros((rows, cols))
    for row in range(0, rows):
        out[row, :] = np.sum(array - array[row, :], 0)
    return out
`,
  },
  {
    name: "periodic_dist",
    setup:
      "import numpy as np ; N = 20 ; x = y = z = np.arange(0., N, 0.1) ; L = 4 ; periodic = True",
    run: "periodic_dist(x, x, x, L,periodic, periodic, periodic)",
    code: `
import numpy as np

def periodic_dist(x, y, z, L, periodicX, periodicY, periodicZ):
    N = len(x)
    xtemp = np.tile(x, (N, 1))
    dx = xtemp - xtemp.T
    ytemp = np.tile(y, (N, 1))
    dy = ytemp - ytemp.T
    ztemp = np.tile(z, (N, 1))
    dz = ztemp - ztemp.T
    if periodicX:
        dx[dx > L / 2] = dx[dx > L / 2] - L
        dx[dx < -L / 2] = dx[dx < -L / 2] + L
    if periodicY:
        dy[dy > L / 2] = dy[dy > L / 2] - L
        dy[dy < -L / 2] = dy[dy < -L / 2] + L
    if periodicZ:
        dz[dz > L / 2] = dz[dz > L / 2] - L
        dz[dz < -L / 2] = dz[dz < -L / 2] + L
    d = np.sqrt(dx**2 + dy**2 + dz**2)
    d[d == 0] = -1
    return d, dx, dy, dz
`,
  },
  {
    name: "reverse_cumsum",
    setup: "import numpy as np ; r = np.random.rand(1000000)",
    run: "reverse_cumsum(r)",
    code: `
import numpy as np

def reverse_cumsum(x):
    return np.cumsum(x[::-1])[::-1]
`,
  },
  {
    name: "rosen",
    setup: "import numpy as np; r = np.arange(1000000, dtype=float)",
    run: "rosen(r)",
    code: `
import numpy as np

def rosen(x):
    t0 = 100 * (x[1:] - x[:-1] ** 2) ** 2
    t1 = (1 - x[:-1]) ** 2
    return np.sum(t0 + t1)
`,
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function findWheel(variantDir) {
  const files = fs.readdirSync(variantDir);
  const whl = files.find((f) => f.endsWith(".whl"));
  if (!whl) throw new Error(`No .whl file found in ${variantDir}`);
  return path.join(variantDir, whl);
}

function buildBenchmarkScript(bench) {
  // Mirrors pyodide/benchmark/benchmark.py's get_benchmark_scripts()
  // Adds a warmup phase to let JIT/wasm compilation settle before measuring
  return `
import numpy as np
_ = np.empty(())

${bench.code}

setup = ${JSON.stringify(bench.setup)}
setup = setup + '\\nfrom __main__ import ${bench.name}'
run = ${JSON.stringify(bench.run)}

from timeit import Timer
t = Timer(run, setup)

# Warmup: run a few untimed iterations so JIT/wasm caches stabilize
t.repeat(${WARMUP}, ${NUMBER})

# Actual measurement
r = t.repeat(${REPEAT}, ${NUMBER})
r.remove(min(r))
r.remove(max(r))
float(np.mean(r))
`;
}

function formatTime(seconds) {
  if (seconds < 0.001) return `${(seconds * 1_000_000).toFixed(1)}µs`;
  if (seconds < 1) return `${(seconds * 1000).toFixed(2)}ms`;
  return `${seconds.toFixed(4)}s`;
}

// ---------------------------------------------------------------------------
// Run benchmarks for one variant
// ---------------------------------------------------------------------------
async function runVariant(variantName) {
  const variantDir = path.join(WHEEL_DIR, variantName);
  const wheelPath = findWheel(variantDir);
  const wheelUrl = `file://${wheelPath}`;

  console.log(`\n${"=".repeat(60)}`);
  console.log(`  Loading variant: ${variantName}`);
  console.log(`  Wheel: ${path.basename(wheelPath)} (${(fs.statSync(wheelPath).size / 1024).toFixed(0)} KB)`);
  console.log(`${"=".repeat(60)}`);

  const pyodide = await loadPyodide();
  await pyodide.loadPackage(`${wheelUrl}`);
  await pyodide.runPythonAsync(`
import numpy as np
print(f"numpy {np.__version__} loaded")
`);

  const results = {};

  for (const bench of BENCHMARKS) {
    const script = buildBenchmarkScript(bench);
    try {
      const elapsed = await pyodide.runPythonAsync(script);
      results[bench.name] = elapsed;
      console.log(`  ✓ ${bench.name.padEnd(24)} ${formatTime(elapsed)}`);
    } catch (err) {
      console.log(`  ✗ ${bench.name.padEnd(24)} ERROR: ${err.message.split("\n").pop()}`);
      results[bench.name] = null;
    }
  }

  return results;
}

// ---------------------------------------------------------------------------
// Comparison table
// ---------------------------------------------------------------------------
function printComparisonTable(allResults) {
  const baseline = allResults[BASELINE];
  if (!baseline) {
    console.error(`Baseline variant "${BASELINE}" not found in results`);
    return;
  }

  console.log(`\n${"=".repeat(80)}`);
  console.log("  COMPARISON TABLE  (baseline: " + BASELINE + ")");
  console.log(`${"=".repeat(80)}`);

  // Header
  const nameCol = 24;
  let header = "Benchmark".padEnd(nameCol);
  for (const v of VARIANTS) {
    header += `| ${v.padEnd(14)}`;
  }
  console.log(header);
  console.log("-".repeat(header.length));

  for (const bench of BENCHMARKS) {
    let row = bench.name.padEnd(nameCol);
    for (const v of VARIANTS) {
      const val = allResults[v]?.[bench.name];
      const baseVal = baseline[bench.name];
      if (val == null || baseVal == null) {
        row += `| ${"N/A".padEnd(14)}`;
      } else {
        const ratio = val / baseVal;
        const pctChange = ((ratio - 1) * 100).toFixed(1);
        const arrow = ratio < 1 ? "↑" : ratio > 1 ? "↓" : "=";
        const label =
          v === BASELINE
            ? formatTime(val)
            : `${ratio.toFixed(3)}x ${arrow} (${pctChange > 0 ? "+" : ""}${pctChange}%)`;
        row += `| ${label.padEnd(14)}`;
      }
    }
    console.log(row);
  }

  // Summary row
  console.log("-".repeat(header.length));
  let summaryRow = "GEOMETRIC MEAN".padEnd(nameCol);
  for (const v of VARIANTS) {
    const ratios = [];
    for (const bench of BENCHMARKS) {
      const val = allResults[v]?.[bench.name];
      const baseVal = baseline[bench.name];
      if (val != null && baseVal != null) {
        ratios.push(val / baseVal);
      }
    }
    if (ratios.length === 0) {
      summaryRow += `| ${"N/A".padEnd(14)}`;
    } else {
      const geoMean = Math.exp(
        ratios.reduce((sum, r) => sum + Math.log(r), 0) / ratios.length
      );
      const pctChange = ((geoMean - 1) * 100).toFixed(1);
      const arrow = geoMean < 1 ? "↑" : geoMean > 1 ? "↓" : "=";
      const label =
        v === BASELINE
          ? "1.000x (baseline)"
          : `${geoMean.toFixed(3)}x ${arrow} (${pctChange > 0 ? "+" : ""}${pctChange}%)`;
      summaryRow += `| ${label.padEnd(14)}`;
    }
  }
  console.log(summaryRow);
}

// ---------------------------------------------------------------------------
// Markdown summary (for GitHub Actions)
// ---------------------------------------------------------------------------
function generateMarkdownSummary(allResults) {
  const baseline = allResults[BASELINE];
  if (!baseline) return "";

  let md = "## NumPy Pyodide Benchmark Results\n\n";
  md += `**Baseline**: \`${BASELINE}\` | **Warmup**: ${WARMUP} | **Repeat**: ${REPEAT} | **Number**: ${NUMBER}\n\n`;

  // Table header
  md += "| Benchmark |";
  for (const v of VARIANTS) {
    md += ` ${v} |`;
  }
  md += "\n|---|";
  for (const _v of VARIANTS) {
    md += "---|";
  }
  md += "\n";

  // Rows
  for (const bench of BENCHMARKS) {
    md += `| ${bench.name} |`;
    for (const v of VARIANTS) {
      const val = allResults[v]?.[bench.name];
      const baseVal = baseline[bench.name];
      if (val == null || baseVal == null) {
        md += " N/A |";
      } else if (v === BASELINE) {
        md += ` ${formatTime(val)} |`;
      } else {
        const ratio = val / baseVal;
        const pctChange = ((ratio - 1) * 100).toFixed(1);
        const emoji = ratio < 0.95 ? "🟢" : ratio > 1.05 ? "🔴" : "⚪";
        md += ` ${emoji} ${ratio.toFixed(3)}x (${pctChange > 0 ? "+" : ""}${pctChange}%) |`;
      }
    }
    md += "\n";
  }

  // Geo mean row
  md += `| **GEOMETRIC MEAN** |`;
  for (const v of VARIANTS) {
    const ratios = [];
    for (const bench of BENCHMARKS) {
      const val = allResults[v]?.[bench.name];
      const baseVal = baseline[bench.name];
      if (val != null && baseVal != null) {
        ratios.push(val / baseVal);
      }
    }
    if (ratios.length === 0) {
      md += " N/A |";
    } else {
      const geoMean = Math.exp(
        ratios.reduce((sum, r) => sum + Math.log(r), 0) / ratios.length
      );
      if (v === BASELINE) {
        md += " **1.000x (baseline)** |";
      } else {
        const pctChange = ((geoMean - 1) * 100).toFixed(1);
        const emoji = geoMean < 0.95 ? "🟢" : geoMean > 1.05 ? "🔴" : "⚪";
        md += ` ${emoji} **${geoMean.toFixed(3)}x (${pctChange > 0 ? "+" : ""}${pctChange}%)** |`;
      }
    }
  }
  md += "\n";

  md += "\n> 🟢 Faster than baseline | 🔴 Slower than baseline | ⚪ Within ±5%\n";
  md += `> Ratio < 1 means faster. Times are wall-clock for ${NUMBER} iterations.\n`;

  return md;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  console.log("NumPy Pyodide Benchmark");
  console.log(`Variants: ${VARIANTS.join(", ")}`);
  console.log(`Baseline: ${BASELINE}`);
  console.log(`Benchmarks: ${BENCHMARKS.length}`);
  console.log(`Warmup: ${WARMUP}, Repeat: ${REPEAT}, Number: ${NUMBER}`);

  const allResults = {};

  for (const variant of VARIANTS) {
    allResults[variant] = await runVariant(variant);
  }

  // Print comparison
  printComparisonTable(allResults);

  // Write JSON results
  const outputPath = path.join(__dirname, "results.json");
  fs.writeFileSync(
    outputPath,
    JSON.stringify(
      {
        metadata: {
          baseline: BASELINE,
          warmup: WARMUP,
          repeat: REPEAT,
          number: NUMBER,
          variants: VARIANTS,
          benchmarks: BENCHMARKS.map((b) => b.name),
          timestamp: new Date().toISOString(),
        },
        results: allResults,
      },
      null,
      2
    )
  );
  console.log(`\nResults written to ${outputPath}`);

  // Write markdown summary (for GitHub Actions)
  const mdPath = path.join(__dirname, "results.md");
  const md = generateMarkdownSummary(allResults);
  fs.writeFileSync(mdPath, md);
  console.log(`Markdown summary written to ${mdPath}`);

  // Also append to $GITHUB_STEP_SUMMARY if available
  if (process.env.GITHUB_STEP_SUMMARY) {
    fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, md);
    console.log("Summary appended to GitHub Step Summary");
  }
}

main().catch((err) => {
  console.error("Benchmark failed:", err);
  process.exit(1);
});
