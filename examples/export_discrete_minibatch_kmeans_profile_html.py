"""Export a self-contained responsive HTML heatmap fragment.

The fragment embeds the median summary rows directly, so it can be opened or
hosted without a server and does not depend on a network fetch or a plotting
library.  It is intentionally an HTML fragment (style, section, and script
only) rather than a complete document.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "artifacts" / "profiling" / "discrete_minibatch_kmeans" / "profile_summary.csv"
DEFAULT_OUTPUT = Path(
    "/Users/niklasmelton/.codex/visualizations/2026/08/08/"
    "019fdf2c-e00b-7532-a2ab-5474e9f56056/minibatch-kmeans-profile.html"
)


HTML_TEMPLATE = r'''<style>
.oi-runtime-profile {
  --oi-ink: var(--foreground);
  --oi-muted: var(--foreground);
  --oi-grid: var(--border);
  --oi-popover: var(--popover);
  --oi-cluster: var(--viz-series-1);
  --oi-score: var(--viz-series-2);
  --oi-other: var(--viz-series-3);
  box-sizing: border-box;
  max-width: 1380px;
  margin: 0 auto;
  padding: 22px;
  color: var(--oi-ink);
  background: transparent;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.oi-runtime-profile *, .oi-runtime-profile *::before, .oi-runtime-profile *::after {
  box-sizing: border-box;
}
.oi-runtime-header {
  margin: 0 0 18px;
}
.oi-runtime-title {
  margin: 0;
  font-size: clamp(1.25rem, 2vw, 1.8rem);
  letter-spacing: -0.02em;
}
.oi-runtime-subtitle {
  margin: 6px 0 0;
  color: var(--oi-muted);
  font-size: 0.92rem;
}
.oi-runtime-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}
.oi-runtime-card {
  min-width: 0;
  overflow: hidden;
  border: 0;
  background: transparent;
}
.oi-runtime-card h2 {
  margin: 0;
  padding: 15px 17px 0;
  font-size: 1rem;
  line-height: 1.25;
}
.oi-runtime-chart {
  width: 100%;
  min-height: 300px;
  padding: 7px 10px 12px;
}
.oi-runtime-chart svg {
  display: block;
  width: 100%;
  height: auto;
}
.oi-runtime-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 6px 17px 16px;
  color: var(--oi-muted);
  opacity: 0.78;
  font-size: 0.78rem;
}
.oi-runtime-legend span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.oi-runtime-swatch {
  width: 11px;
  height: 11px;
  border-radius: 3px;
  background: var(--swatch);
}
.oi-runtime-note {
  margin: 16px 2px 0;
  color: var(--oi-muted);
  font-size: 0.78rem;
  line-height: 1.45;
}
@media (max-width: 780px) {
  .oi-runtime-profile { padding: 14px; }
  .oi-runtime-grid { grid-template-columns: 1fr; }
  .oi-runtime-chart { min-height: 260px; }
}
</style>

<section id="oi-minibatch-profile" class="oi-runtime-profile" aria-label="MiniBatchKMeans runtime profile">
  <header class="oi-runtime-header">
    <h1 class="oi-runtime-title">Discrete OverlapIndex MiniBatchKMeans runtime profile</h1>
    <p class="oi-runtime-subtitle">Median fit time by sample count and feature dimensionality. The fourth panel marks the component with the largest median time.</p>
  </header>
  <div class="oi-runtime-grid">
    <article class="oi-runtime-card">
      <h2>Clustering time (median seconds)</h2>
      <div class="oi-runtime-chart" data-chart="clustering"></div>
    </article>
    <article class="oi-runtime-card">
      <h2>Overlap scoring time (median seconds)</h2>
      <div class="oi-runtime-chart" data-chart="score"></div>
    </article>
    <article class="oi-runtime-card">
      <h2>Other fit work (median seconds)</h2>
      <div class="oi-runtime-chart" data-chart="other"></div>
    </article>
    <article class="oi-runtime-card">
      <h2>Dominant component (median time)</h2>
      <div class="oi-runtime-chart" data-chart="dominant"></div>
      <div class="oi-runtime-legend" aria-label="Dominant component legend">
        <span><i class="oi-runtime-swatch" style="--swatch: var(--oi-cluster)"></i>Cluster (C)</span>
        <span><i class="oi-runtime-swatch" style="--swatch: var(--oi-score)"></i>Score (S)</span>
        <span><i class="oi-runtime-swatch" style="--swatch: var(--oi-other)"></i>Other (O)</span>
      </div>
    </article>
  </div>
  <p class="oi-runtime-note">Components are exhaustive within each fit: other = total − clustering − overlap scoring. Raw repetitions and the median summary are stored beside the source benchmark.</p>
</section>

<script>
(function () {
  "use strict";
  const rows = /*DATA*/;
  const root = document.getElementById("oi-minibatch-profile");
  if (!root || !rows.length) return;

  const sampleValues = [...new Set(rows.map((row) => Number(row.n_samples)))].sort((a, b) => a - b);
  const dimensionValues = [...new Set(rows.map((row) => Number(row.n_dimensions)))].sort((a, b) => a - b);
  const components = [
    { id: "clustering", key: "median_clustering_seconds", short: "C", css: "--oi-cluster", label: "Clustering" },
    { id: "score", key: "median_overlap_scoring_seconds", short: "S", css: "--oi-score", label: "Overlap scoring" },
    { id: "other", key: "median_other_seconds", short: "O", css: "--oi-other", label: "Other fit work" },
  ];
  const rowsByCell = new Map(rows.map((row) => [`${row.n_samples}:${row.n_dimensions}`, row]));
  const ns = 640;
  const nh = 412;
  const margin = { top: 22, right: 76, bottom: 55, left: 72 };
  const plotWidth = ns - margin.left - margin.right;
  const plotHeight = nh - margin.top - margin.bottom;

  function cssColor(variable) {
    return getComputedStyle(root).getPropertyValue(variable).trim();
  }
  function parseColor(value) {
    if (!value) return null;
    const hex = value.trim().replace("#", "");
    if (/^[0-9a-f]{3,8}$/i.test(hex)) {
      const full = hex.length === 3 ? hex.split("").map((part) => part + part).join("") : hex;
      return [parseInt(full.slice(0, 2), 16), parseInt(full.slice(2, 4), 16), parseInt(full.slice(4, 6), 16)];
    }
    const channels = value.match(/rgba?\(([^)]+)\)/i);
    if (!channels) return null;
    const values = channels[1].split(",").slice(0, 3).map((part) => Number.parseFloat(part.trim()));
    return values.length === 3 && values.every(Number.isFinite) ? values : null;
  }
  function mixColor(base, amount) {
    const rgb = parseColor(base);
    if (!rgb) return base;
    const neutral = parseColor(cssColor("--border")) || rgb;
    const [r, g, b] = rgb;
    const target = neutral;
    const strength = 0.18 + 0.82 * amount;
    return `rgb(${target.map((channel, index) => Math.round(channel * (1 - strength) + [r, g, b][index] * strength)).join(",")})`;
  }
  function textColor(base, amount) {
    const rgb = parseColor(base);
    if (!rgb) return cssColor("--foreground");
    const [r, g, b] = rgb;
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) * (0.2 + 0.8 * amount);
    return luminance < 145 ? cssColor("--oi-popover") : cssColor("--foreground");
  }
  function valueAt(sample, dimension, key) {
    const row = rowsByCell.get(`${sample}:${dimension}`);
    return row ? Number(row[key]) : NaN;
  }
  function fmt(value) {
    if (!Number.isFinite(value)) return "–";
    if (value >= 0.1) return value.toFixed(2);
    if (value >= 0.01) return value.toFixed(3);
    return value.toPrecision(2);
  }
  function svgText(x, y, text, options) {
    const opts = options || {};
    return `<text x="${x}" y="${y}" text-anchor="${opts.anchor || "middle"}" dominant-baseline="middle" fill="${opts.fill || cssColor("--foreground")}" font-size="${opts.size || 11}" font-family="ui-sans-serif,system-ui,sans-serif">${text}</text>`;
  }
  function shell(title, body) {
    return `<svg viewBox="0 0 ${ns} ${nh}" role="img" aria-label="${title}"><title>${title}</title>${body}</svg>`;
  }
  function renderTiming(host, component) {
    const values = [];
    sampleValues.forEach((sample) => dimensionValues.forEach((dimension) => {
      const value = valueAt(sample, dimension, component.key);
      if (Number.isFinite(value)) values.push(value);
    }));
    const low = Math.min(...values, 0);
    const high = Math.max(...values, 1e-12);
    const base = cssColor(component.css);
    const cellWidth = plotWidth / dimensionValues.length;
    const cellHeight = plotHeight / sampleValues.length;
    let body = "";
    sampleValues.forEach((sample, rowIndex) => dimensionValues.forEach((dimension, columnIndex) => {
      const value = valueAt(sample, dimension, component.key);
      const amount = Number.isFinite(value) && high > low ? (value - low) / (high - low) : 0;
      const x = margin.left + columnIndex * cellWidth;
      const y = margin.top + (sampleValues.length - rowIndex - 1) * cellHeight;
      const fill = Number.isFinite(value) ? mixColor(base, Math.max(0, Math.min(1, amount))) : cssColor("--border");
      body += `<rect x="${x}" y="${y}" width="${cellWidth}" height="${cellHeight}" fill="${fill}" stroke="${cssColor("--oi-popover")}" stroke-width="1"/>`;
      body += svgText(x + cellWidth / 2, y + cellHeight / 2, fmt(value), { fill: textColor(base, amount), size: 10 });
    }));
    dimensionValues.forEach((dimension, index) => {
      body += svgText(margin.left + (index + 0.5) * cellWidth, nh - margin.bottom + 18, dimension, { size: 10 });
    });
    sampleValues.forEach((sample, index) => {
      const y = margin.top + (sampleValues.length - index - 0.5) * cellHeight;
      body += svgText(margin.left - 9, y, sample, { anchor: "end", size: 10 });
    });
    body += svgText(margin.left + plotWidth / 2, nh - 10, "Feature dimensions", { size: 11 });
    body += `<text transform="translate(15 ${margin.top + plotHeight / 2}) rotate(-90)" text-anchor="middle" fill="${cssColor("--foreground")}" font-size="11" font-family="ui-sans-serif,system-ui,sans-serif">Samples</text>`;
    const barX = ns - margin.right + 23;
    const barHeight = plotHeight;
    for (let index = 0; index < 22; index += 1) {
      const amount = index / 21;
      body += `<rect x="${barX}" y="${margin.top + (1 - amount) * barHeight - barHeight / 22}" width="13" height="${barHeight / 22 + 1}" fill="${mixColor(base, amount)}"/>`;
    }
    body += svgText(barX + 6, margin.top - 8, fmt(high), { size: 9 });
    body += svgText(barX + 6, margin.top + barHeight + 12, fmt(low), { size: 9 });
    host.innerHTML = shell(`${component.label} time heatmap`, body);
  }
  function renderDominant(host) {
    const cellWidth = plotWidth / dimensionValues.length;
    const cellHeight = plotHeight / sampleValues.length;
    let body = "";
    sampleValues.forEach((sample, rowIndex) => dimensionValues.forEach((dimension, columnIndex) => {
      const values = components.map((component) => valueAt(sample, dimension, component.key));
      const winner = values.every(Number.isFinite) ? values.indexOf(Math.max(...values)) : -1;
      const x = margin.left + columnIndex * cellWidth;
      const y = margin.top + (sampleValues.length - rowIndex - 1) * cellHeight;
      const color = winner >= 0 ? cssColor(components[winner].css) : cssColor("--border");
      body += `<rect x="${x}" y="${y}" width="${cellWidth}" height="${cellHeight}" fill="${color}" stroke="${cssColor("--oi-popover")}" stroke-width="1"/>`;
      if (winner >= 0) body += svgText(x + cellWidth / 2, y + cellHeight / 2, components[winner].short, { fill: cssColor("--oi-popover"), size: 13 });
    }));
    dimensionValues.forEach((dimension, index) => {
      body += svgText(margin.left + (index + 0.5) * cellWidth, nh - margin.bottom + 18, dimension, { size: 10 });
    });
    sampleValues.forEach((sample, index) => {
      const y = margin.top + (sampleValues.length - index - 0.5) * cellHeight;
      body += svgText(margin.left - 9, y, sample, { anchor: "end", size: 10 });
    });
    body += svgText(margin.left + plotWidth / 2, nh - 10, "Feature dimensions", { size: 11 });
    body += `<text transform="translate(15 ${margin.top + plotHeight / 2}) rotate(-90)" text-anchor="middle" fill="${cssColor("--foreground")}" font-size="11" font-family="ui-sans-serif,system-ui,sans-serif">Samples</text>`;
    host.innerHTML = shell("Dominant component heatmap", body);
  }
  root.querySelector('[data-chart="clustering"]').replaceChildren();
  renderTiming(root.querySelector('[data-chart="clustering"]'), components[0]);
  renderTiming(root.querySelector('[data-chart="score"]'), components[1]);
  renderTiming(root.querySelector('[data-chart="other"]'), components[2]);
  renderDominant(root.querySelector('[data-chart="dominant"]'));
})();
</script>
'''


def _load_rows(path: Path) -> list[dict[str, int | float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        source = csv.DictReader(handle)
        rows = []
        for row in source:
            rows.append(
                {
                    "n_samples": int(row["n_samples"]),
                    "n_dimensions": int(row["n_dimensions"]),
                    "median_clustering_seconds": float(row["median_clustering_seconds"]),
                    "median_overlap_scoring_seconds": float(row["median_overlap_scoring_seconds"]),
                    "median_other_seconds": float(row["median_other_seconds"]),
                }
            )
    if not rows:
        raise ValueError(f"Summary file is empty: {path}")
    return rows


def export_fragment(summary_path: Path, output_path: Path) -> Path:
    """Embed summary rows in the fragment and write it to ``output_path``."""

    rows = _load_rows(summary_path)
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    html = HTML_TEMPLATE.replace("/*DATA*/", payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(export_fragment(args.summary.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
