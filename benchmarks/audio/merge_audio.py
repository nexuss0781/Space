#!/usr/bin/env python3
"""Merge per-model audio benchmark JSONs into a single markdown report."""

import argparse
import json
import pathlib


def gather(input_dir):
    jsons = sorted(pathlib.Path(input_dir).rglob("benchmark_*.json"))
    models = [json.loads(p.read_text()) for p in jsons]
    return sorted(models, key=lambda m: m["name"])


def render(models):
    head = """# Audio-Benchmark Report (all candidates)

Consolidated comparison of every audio candidate benchmarked on the same
10-clip speech task set with llama.cpp (`llama-mtmd-cli`, build from source).
Reference transcript is shown as `>>>`.

"""

    head += "| Candidate | Model file | Weights | Avg tok/s | Wall (10 clips) | Tasks |\n|---|---|---:|---:|---:|---:|\n"
    for m in models:
        head += (f"| {m['name']} | {m['model_file']} | {m['model_gb']} GB "
                 f"| {m['avg_tokens_per_sec']} | {m['wall_s_total']}s | {len(m['rows'])} |\n")
    head += "\n"

    sections = []
    for m in models:
        s = f"## {m['name']}\n\n- Source: `{m['source_repo']}`\n"
        s += f"- Files: `{m['model_file']}` ({m['model_gb']} GB) + `{m['mmproj_file']}`\n"
        s += f"- System: {m['system']['machine']}, {m['system']['cpu_cores']} cores, {m['system']['total_ram_gb']} GB RAM\n"
        s += f"- Wall (10 tasks): {m['wall_s_total']}s | avg throughput: {m['avg_tokens_per_sec']} tok/s\n\n"
        s += "| # | Audio | Out tok | eval (ms) | tok/s | peak RSS (MB) | wall (s) |\n|---|---|---:|---:|---:|---:|---:|\n"
        for r in m["rows"]:
            s += (f"| {r['index']} | {r['audio']} | {r['output_tokens']} | {r['eval_ms']} "
                  f"| {r['tokens_per_sec']} | {r['peak_rss_mb']} | {r['wall_s']} |\n")
        s += "\n### Responses\n\n"
        for r in m["rows"]:
            s += f"**Task {r['index']} ({r['audio']}):** {r['prompt'][:60]}...\n"
            s += f"**Reference:** `{r['ref']}`\n\n```text\n{r['response']}\n```\n\n"
        sections.append(s)

    return head + "\n---\n\n".join(sections)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("--out", default="reports/audio-benchmark.md")
    args = ap.parse_args()

    models = gather(args.input_dir)
    if not models:
        raise SystemExit("no benchmark_*.json found under input dir")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(models))
    print(f"[merge] wrote {out} ({len(models)} models)")


if __name__ == "__main__":
    main()