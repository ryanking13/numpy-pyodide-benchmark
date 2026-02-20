#!/usr/bin/env python3
"""Compare compressed and uncompressed sizes of native .so modules across numpy wheel variants."""

import json
import os
import re
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WHEEL_DIR = Path(__file__).parent / "numpy-wheels"
BASELINE = "o2"


def discover_variants():
    return sorted(d.name for d in WHEEL_DIR.iterdir() if d.is_dir())


def find_wheel(variant_dir: Path) -> Path:
    wheels = list(variant_dir.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError(f"No .whl file found in {variant_dir}")
    return wheels[0]


def short_name(filename: str) -> str:
    """numpy/_core/_multiarray_umath.cpython-313-wasm32-emscripten.so → _core/_multiarray_umath"""
    name = filename.removeprefix("numpy/")
    return re.sub(r"\.cpython-[^.]+\.so$", "", name)


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def diff_str(val: int, base: int) -> str:
    pct = (val / base - 1) * 100
    sign = "+" if pct > 0 else ""
    return f"{format_bytes(val)} ({sign}{pct:.1f}%)"


# ---------------------------------------------------------------------------
# Collect sizes
# ---------------------------------------------------------------------------
def collect_variant(variant: str) -> dict:
    wheel_path = find_wheel(WHEEL_DIR / variant)
    wheel_size = wheel_path.stat().st_size
    so_files = {}

    with zipfile.ZipFile(wheel_path) as zf:
        for info in zf.infolist():
            if info.filename.endswith(".so"):
                so_files[short_name(info.filename)] = {
                    "compressed": info.compress_size,
                    "uncompressed": info.file_size,
                }

    return {"wheel_size": wheel_size, "so_files": so_files}


# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------
def print_table(all_data: dict, variants: list[str]):
    baseline_data = all_data[BASELINE]
    all_modules = sorted({mod for v in all_data.values() for mod in v["so_files"]})

    name_col = 40

    # --- Wheel size ---
    print("=" * 80)
    print("  WHEEL SIZE COMPARISON")
    print("=" * 80)
    print()

    header = "".ljust(name_col) + "".join(f"| {v:<18}" for v in variants)
    print(header)
    print("-" * len(header))

    row = "Wheel (total)".ljust(name_col)
    for v in variants:
        ws = all_data[v]["wheel_size"]
        if v == BASELINE:
            row += f"| {format_bytes(ws):<18}"
        else:
            row += f"| {diff_str(ws, baseline_data['wheel_size']):<18}"
    print(row)
    print()

    # --- Per-module tables ---
    for label, key in [
        ("NATIVE MODULE SIZES — UNCOMPRESSED", "uncompressed"),
        ("NATIVE MODULE SIZES — COMPRESSED (in wheel)", "compressed"),
    ]:
        print("=" * 80)
        print(f"  {label}")
        print("=" * 80)
        print()

        header = "Module".ljust(name_col) + "".join(f"| {v:<18}" for v in variants)
        print(header)
        print("-" * len(header))

        totals = {v: 0 for v in variants}

        for mod in all_modules:
            row = mod.ljust(name_col)
            for v in variants:
                entry = all_data[v]["so_files"].get(mod)
                base_entry = baseline_data["so_files"].get(mod)
                if not entry:
                    row += f"| {'—':<18}"
                else:
                    totals[v] += entry[key]
                    if not base_entry or v == BASELINE:
                        row += f"| {format_bytes(entry[key]):<18}"
                    else:
                        row += f"| {diff_str(entry[key], base_entry[key]):<18}"
            print(row)

        print("-" * len(header))
        row = "TOTAL".ljust(name_col)
        for v in variants:
            if v == BASELINE:
                row += f"| {format_bytes(totals[v]):<18}"
            else:
                row += f"| {diff_str(totals[v], totals[BASELINE]):<18}"
        print(row)
        print()


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------
def generate_markdown(all_data: dict, variants: list[str]) -> str:
    baseline_data = all_data[BASELINE]
    all_modules = sorted({mod for v in all_data.values() for mod in v["so_files"]})

    md = "## NumPy Native Module Size Comparison\n\n"
    md += f"**Baseline**: `{BASELINE}`\n\n"

    # --- Wheel size ---
    md += "### Wheel Size\n\n"
    md += "| |" + "".join(f" {v} |" for v in variants) + "\n"
    md += "|---|" + "---|" * len(variants) + "\n"
    md += "| Wheel (total) |"
    for v in variants:
        ws = all_data[v]["wheel_size"]
        if v == BASELINE:
            md += f" {format_bytes(ws)} |"
        else:
            md += f" {diff_str(ws, baseline_data['wheel_size'])} |"
    md += "\n\n"

    # --- Per-module tables ---
    for title, key in [
        ("Uncompressed Size", "uncompressed"),
        ("Compressed Size (in wheel)", "compressed"),
    ]:
        md += f"### {title}\n\n"
        md += "| Module |" + "".join(f" {v} |" for v in variants) + "\n"
        md += "|---|" + "---|" * len(variants) + "\n"

        totals = {v: 0 for v in variants}

        for mod in all_modules:
            md += f"| {mod} |"
            for v in variants:
                entry = all_data[v]["so_files"].get(mod)
                base_entry = baseline_data["so_files"].get(mod)
                if not entry:
                    md += " — |"
                else:
                    totals[v] += entry[key]
                    if not base_entry or v == BASELINE:
                        md += f" {format_bytes(entry[key])} |"
                    else:
                        md += f" {diff_str(entry[key], base_entry[key])} |"
            md += "\n"

        md += "| **TOTAL** |"
        for v in variants:
            if v == BASELINE:
                md += f" **{format_bytes(totals[v])}** |"
            else:
                md += f" **{diff_str(totals[v], totals[BASELINE])}** |"
        md += "\n\n"

    return md


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    variants = discover_variants()
    print("NumPy Native Module Size Comparison")
    print(f"Variants: {', '.join(variants)}")
    print(f"Baseline: {BASELINE}")
    print()

    all_data = {v: collect_variant(v) for v in variants}

    print_table(all_data, variants)

    # Write markdown
    script_dir = Path(__file__).parent
    md_path = script_dir / "size-results.md"
    md = generate_markdown(all_data, variants)
    md_path.write_text(md)
    print(f"\nMarkdown written to {md_path}")

    # Append to GitHub Step Summary if available
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(md)
        print("Summary appended to GitHub Step Summary")


if __name__ == "__main__":
    main()
