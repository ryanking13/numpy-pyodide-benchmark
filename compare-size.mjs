import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const WHEEL_DIR = path.join(__dirname, "numpy-wheels");
const BASELINE = "o2";
const VARIANTS = fs
  .readdirSync(WHEEL_DIR)
  .filter((d) => fs.statSync(path.join(WHEEL_DIR, d)).isDirectory())
  .sort();

// ---------------------------------------------------------------------------
// Minimal zip central-directory parser (metadata only, no extraction)
//
// ZIP end-of-central-directory record (EOCD):
//   offset 0:  signature  0x06054b50
//   offset 16: central directory size (4 bytes)
//   offset 12: central directory offset (4 bytes) — actually offset 16 is size, 12 is start offset
//
// Central directory file header:
//   offset 0:  signature  0x02014b50
//   offset 20: compressed size (4 bytes)
//   offset 24: uncompressed size (4 bytes)
//   offset 28: filename length (2 bytes)
//   offset 30: extra field length (2 bytes)
//   offset 32: file comment length (2 bytes)
//   offset 46: filename (variable)
// ---------------------------------------------------------------------------

function readZipEntries(filePath) {
  const buf = fs.readFileSync(filePath);
  const entries = [];

  // Find EOCD record — scan backwards from end (max 65KB comment)
  let eocdOffset = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 65558); i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) {
      eocdOffset = i;
      break;
    }
  }
  if (eocdOffset === -1) throw new Error(`No EOCD record found in ${filePath}`);

  const cdSize = buf.readUInt32LE(eocdOffset + 12);
  const cdOffset = buf.readUInt32LE(eocdOffset + 16);

  let pos = cdOffset;
  while (pos < cdOffset + cdSize) {
    const sig = buf.readUInt32LE(pos);
    if (sig !== 0x02014b50) break;

    const compressedSize = buf.readUInt32LE(pos + 20);
    const uncompressedSize = buf.readUInt32LE(pos + 24);
    const fnLen = buf.readUInt16LE(pos + 28);
    const extraLen = buf.readUInt16LE(pos + 30);
    const commentLen = buf.readUInt16LE(pos + 32);
    const filename = buf.toString("utf8", pos + 46, pos + 46 + fnLen);

    entries.push({ filename, compressedSize, uncompressedSize });
    pos += 46 + fnLen + extraLen + commentLen;
  }

  return entries;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function findWheel(variantDir) {
  const files = fs.readdirSync(variantDir);
  const whl = files.find((f) => f.endsWith(".whl"));
  if (!whl) throw new Error(`No .whl file found in ${variantDir}`);
  return path.join(variantDir, whl);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function shortName(filename) {
  // numpy/_core/_multiarray_umath.cpython-313-wasm32-emscripten.so → _core/_multiarray_umath
  return filename
    .replace(/^numpy\//, "")
    .replace(/\.cpython-[^.]+\.so$/, "");
}

// ---------------------------------------------------------------------------
// Collect sizes
// ---------------------------------------------------------------------------

/** @returns {{ soFiles: Map<string, { compressed: number, uncompressed: number }>, wheelSize: number }} */
function collectVariant(variantName) {
  const wheelPath = findWheel(path.join(WHEEL_DIR, variantName));
  const wheelSize = fs.statSync(wheelPath).size;
  const entries = readZipEntries(wheelPath);
  const soFiles = new Map();

  for (const entry of entries) {
    if (entry.filename.endsWith(".so")) {
      soFiles.set(shortName(entry.filename), {
        compressed: entry.compressedSize,
        uncompressed: entry.uncompressedSize,
      });
    }
  }

  return { soFiles, wheelSize };
}

// ---------------------------------------------------------------------------
// Print tables
// ---------------------------------------------------------------------------

function printTable(allData) {
  const baselineData = allData.get(BASELINE);
  if (!baselineData) {
    console.error(`Baseline variant "${BASELINE}" not found`);
    return;
  }

  // Collect all .so module names (union across variants)
  const allModules = new Set();
  for (const { soFiles } of allData.values()) {
    for (const name of soFiles.keys()) allModules.add(name);
  }
  const modules = [...allModules].sort();

  // --- Wheel-level summary ---
  console.log("=".repeat(80));
  console.log("  WHEEL SIZE COMPARISON");
  console.log("=".repeat(80));
  console.log("");

  const nameCol = 40;
  let header = "".padEnd(nameCol);
  for (const v of VARIANTS) header += `| ${v.padEnd(18)}`;
  console.log(header);
  console.log("-".repeat(header.length));

  let row = "Wheel (total)".padEnd(nameCol);
  for (const v of VARIANTS) {
    const { wheelSize } = allData.get(v);
    const baseSize = baselineData.wheelSize;
    const diff = ((wheelSize / baseSize - 1) * 100).toFixed(1);
    const label =
      v === BASELINE
        ? formatBytes(wheelSize)
        : `${formatBytes(wheelSize)} (${diff > 0 ? "+" : ""}${diff}%)`;
    row += `| ${label.padEnd(18)}`;
  }
  console.log(row);
  console.log("");

  // --- Uncompressed .so sizes ---
  console.log("=".repeat(80));
  console.log("  NATIVE MODULE SIZES — UNCOMPRESSED");
  console.log("=".repeat(80));
  console.log("");

  header = "Module".padEnd(nameCol);
  for (const v of VARIANTS) header += `| ${v.padEnd(18)}`;
  console.log(header);
  console.log("-".repeat(header.length));

  let totalUncompressed = Object.fromEntries(VARIANTS.map((v) => [v, 0]));

  for (const mod of modules) {
    let row = mod.padEnd(nameCol);
    for (const v of VARIANTS) {
      const entry = allData.get(v).soFiles.get(mod);
      const baseEntry = baselineData.soFiles.get(mod);
      if (!entry) {
        row += `| ${"—".padEnd(18)}`;
      } else {
        totalUncompressed[v] += entry.uncompressed;
        if (!baseEntry || v === BASELINE) {
          row += `| ${formatBytes(entry.uncompressed).padEnd(18)}`;
        } else {
          const diff = ((entry.uncompressed / baseEntry.uncompressed - 1) * 100).toFixed(1);
          row += `| ${(formatBytes(entry.uncompressed) + ` (${diff > 0 ? "+" : ""}${diff}%)`).padEnd(18)}`;
        }
      }
    }
    console.log(row);
  }

  console.log("-".repeat(header.length));
  row = "TOTAL".padEnd(nameCol);
  for (const v of VARIANTS) {
    const base = totalUncompressed[BASELINE];
    const val = totalUncompressed[v];
    const diff = ((val / base - 1) * 100).toFixed(1);
    const label =
      v === BASELINE
        ? formatBytes(val)
        : `${formatBytes(val)} (${diff > 0 ? "+" : ""}${diff}%)`;
    row += `| ${label.padEnd(18)}`;
  }
  console.log(row);
  console.log("");

  // --- Compressed .so sizes ---
  console.log("=".repeat(80));
  console.log("  NATIVE MODULE SIZES — COMPRESSED (in wheel)");
  console.log("=".repeat(80));
  console.log("");

  header = "Module".padEnd(nameCol);
  for (const v of VARIANTS) header += `| ${v.padEnd(18)}`;
  console.log(header);
  console.log("-".repeat(header.length));

  let totalCompressed = Object.fromEntries(VARIANTS.map((v) => [v, 0]));

  for (const mod of modules) {
    let row = mod.padEnd(nameCol);
    for (const v of VARIANTS) {
      const entry = allData.get(v).soFiles.get(mod);
      const baseEntry = baselineData.soFiles.get(mod);
      if (!entry) {
        row += `| ${"—".padEnd(18)}`;
      } else {
        totalCompressed[v] += entry.compressed;
        if (!baseEntry || v === BASELINE) {
          row += `| ${formatBytes(entry.compressed).padEnd(18)}`;
        } else {
          const diff = ((entry.compressed / baseEntry.compressed - 1) * 100).toFixed(1);
          row += `| ${(formatBytes(entry.compressed) + ` (${diff > 0 ? "+" : ""}${diff}%)`).padEnd(18)}`;
        }
      }
    }
    console.log(row);
  }

  console.log("-".repeat(header.length));
  row = "TOTAL".padEnd(nameCol);
  for (const v of VARIANTS) {
    const base = totalCompressed[BASELINE];
    const val = totalCompressed[v];
    const diff = ((val / base - 1) * 100).toFixed(1);
    const label =
      v === BASELINE
        ? formatBytes(val)
        : `${formatBytes(val)} (${diff > 0 ? "+" : ""}${diff}%)`;
    row += `| ${label.padEnd(18)}`;
  }
  console.log(row);
}

// ---------------------------------------------------------------------------
// Markdown output
// ---------------------------------------------------------------------------

function generateMarkdown(allData) {
  const baselineData = allData.get(BASELINE);
  if (!baselineData) return "";

  const allModules = new Set();
  for (const { soFiles } of allData.values()) {
    for (const name of soFiles.keys()) allModules.add(name);
  }
  const modules = [...allModules].sort();

  let md = "## NumPy Native Module Size Comparison\n\n";
  md += `**Baseline**: \`${BASELINE}\`\n\n`;

  // --- Wheel size ---
  md += "### Wheel Size\n\n";
  md += "| |";
  for (const v of VARIANTS) md += ` ${v} |`;
  md += "\n|---|";
  for (const _v of VARIANTS) md += "---|";
  md += "\n| Wheel (total) |";
  for (const v of VARIANTS) {
    const { wheelSize } = allData.get(v);
    const baseSize = baselineData.wheelSize;
    if (v === BASELINE) {
      md += ` ${formatBytes(wheelSize)} |`;
    } else {
      const diff = ((wheelSize / baseSize - 1) * 100).toFixed(1);
      md += ` ${formatBytes(wheelSize)} (${diff > 0 ? "+" : ""}${diff}%) |`;
    }
  }
  md += "\n\n";

  // Helper to build a size table
  function sizeTable(title, sizeKey) {
    md += `### ${title}\n\n`;
    md += "| Module |";
    for (const v of VARIANTS) md += ` ${v} |`;
    md += "\n|---|";
    for (const _v of VARIANTS) md += "---|";
    md += "\n";

    const totals = Object.fromEntries(VARIANTS.map((v) => [v, 0]));

    for (const mod of modules) {
      md += `| ${mod} |`;
      for (const v of VARIANTS) {
        const entry = allData.get(v).soFiles.get(mod);
        const baseEntry = baselineData.soFiles.get(mod);
        if (!entry) {
          md += " — |";
        } else {
          totals[v] += entry[sizeKey];
          if (!baseEntry || v === BASELINE) {
            md += ` ${formatBytes(entry[sizeKey])} |`;
          } else {
            const diff = ((entry[sizeKey] / baseEntry[sizeKey] - 1) * 100).toFixed(1);
            md += ` ${formatBytes(entry[sizeKey])} (${diff > 0 ? "+" : ""}${diff}%) |`;
          }
        }
      }
      md += "\n";
    }

    md += `| **TOTAL** |`;
    for (const v of VARIANTS) {
      const base = totals[BASELINE];
      const val = totals[v];
      if (v === BASELINE) {
        md += ` **${formatBytes(val)}** |`;
      } else {
        const diff = ((val / base - 1) * 100).toFixed(1);
        md += ` **${formatBytes(val)} (${diff > 0 ? "+" : ""}${diff}%)** |`;
      }
    }
    md += "\n\n";
  }

  sizeTable("Uncompressed Size", "uncompressed");
  sizeTable("Compressed Size (in wheel)", "compressed");

  return md;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

console.log("NumPy Native Module Size Comparison");
console.log(`Variants: ${VARIANTS.join(", ")}`);
console.log(`Baseline: ${BASELINE}`);
console.log("");

const allData = new Map();
for (const v of VARIANTS) {
  allData.set(v, collectVariant(v));
}

printTable(allData);

// Write markdown
const mdPath = path.join(__dirname, "size-results.md");
const md = generateMarkdown(allData);
fs.writeFileSync(mdPath, md);
console.log(`\nMarkdown written to ${mdPath}`);

// Append to GitHub Step Summary if available
if (process.env.GITHUB_STEP_SUMMARY) {
  fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, md);
  console.log("Summary appended to GitHub Step Summary");
}
