#!/usr/bin/env python3
"""Vision-leg benchmark: run N image tasks through llama.cpp llama-mtmd-cli.

Environment/args drive model download + evaluation. Each task = image + prompt.
Records wall time, generated tokens, llama.cpp-reported tok/s and peak RSS.
"""

import argparse
import json
import os
import pathlib
import re
import resource
import subprocess
import time
from datetime import datetime, timezone

import psutil

from huggingface_hub import snapshot_download


def rss_mb():
    return round(psutil.Process().memory_info().rss / 1e6, 2)


def peak_rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)


def system_info():
    return {
        "machine": os.uname().machine,
        "cpu_cores": os.cpu_count(),
        "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "python": __import__("platform").python_version(),
    }


def download(source_repo, model_file, mmproj_file, local_dir):
    snapshot_download(
        repo_id=source_repo,
        allow_patterns=[model_file],
        local_dir=f"{local_dir}/model",
    )
    snapshot_download(
        repo_id=source_repo,
        allow_patterns=[mmproj_file],
        local_dir=f"{local_dir}/mmproj",
    )
    model_path = next(pathlib.Path(f"{local_dir}/model").rglob(model_file))
    mmproj_path = next(pathlib.Path(f"{local_dir}/mmproj").rglob(mmproj_file))
    return model_path, mmproj_path


def run_mtmd(mtmd_bin, model_path, mmproj_path, image_path, prompt, ctx, max_new):
    cmd = [
        mtmd_bin, "-m", str(model_path), "--mmproj", str(mmproj_path),
        "-c", str(ctx), "--image", str(image_path),
        "-p", prompt, "-n", str(max_new), "-t", "4", "--seed", "42",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["/usr/bin/time", "-v", *cmd],
        capture_output=True, text=True, timeout=600,
    )
    wall = time.perf_counter() - t0
    out = proc.stdout
    err = proc.stderr

    tps = None
    for m in re.finditer(r"([\d.]+) tokens per second", out + "\n" + err):
        tps = float(m.group(1))
    m = re.search(r"eval time\s*=\s*([\d.]+) ms\s*/\s*(\d+)\s*runs", out + "\n" + err)
    eval_ms, out_toks = None, None
    if m:
        eval_ms = round(float(m.group(1)), 1)
        out_toks = int(m.group(2))

    rss = None
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", err)
    if m:
        rss = round(int(m.group(1)) / 1024, 2)

    response = extract_response(out)
    return {
        "wall_s": round(wall, 3),
        "output_tokens": out_toks,
        "eval_ms": eval_ms,
        "tokens_per_sec": tps,
        "peak_rss_mb": rss,
        "response": response,
        "rc": proc.returncode,
    }


def extract_response(out):
    cutoff = None
    for line in out.splitlines():
        if re.match(r"^(llama_|print_timings|sample time|prompt eval|eval time|total time)", line.strip()) \
                or "tokens per second" in line:
            cutoff = line
            break
    text = out.split("\n", 1)[0] if "\assistant" not in out else ""
    if cutoff:
        text = out.split(cutoff, 1)[0]
    else:
        text = out
    for tag in ("<assistant>", "<|assistant|>", "<end_of_turn>", "<|im_end|>", "</s>", "<s>"):
        text = text.replace(tag, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:600]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--source-repo", required=True)
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--mmproj-file", required=True)
    ap.add_argument("--mtmd-bin", required=True)
    ap.add_argument("--images-dir", default="benchmarks/vision/images")
    ap.add_argument("--tasks-file", default="benchmarks/vision/tasks.tsv")
    ap.add_argument("--report-dir", default="results/vision")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--local-dir", default="models/vision")
    args = ap.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)

    info = system_info()
    print(f"[benchmark] {args.name} | {info['machine']} | {info['cpu_cores']} cores | {info['total_ram_gb']} GB RAM")

    model_path, mmproj_path = download(
        args.source_repo, args.model_file, args.mmproj_file, f"{args.local_dir}/{args.name}"
    )
    model_gb = round(model_path.stat().st_size / 1e9, 2)
    print(f"[benchmark] model: {model_path.name} ({model_gb} GB) | mmproj: {mmproj_path.name}")

    tasks = []
    with open(args.tasks_file) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                tasks.append((parts[0], parts[1]))
    print(f"[benchmark] running {len(tasks)} image tasks (max_new_tokens={args.max_new_tokens})")

    rows = []
    for i, (img_rel, prompt) in enumerate(tasks, 1):
        img = os.path.join(args.images_dir, img_rel)
        r = run_mtmd(args.mtmd_bin, model_path, mmproj_path, img, prompt, args.ctx, args.max_new_tokens)
        r.update({"index": i, "image": img_rel, "prompt": prompt})
        rows.append(r)
        print(f"  [{i:02d}] {img_rel} | out={r['output_tokens']} tok | eval={r['eval_ms']} ms"
              f" | {r['tokens_per_sec']} tok/s | rss={r['peak_rss_mb']} MB | wall={r['wall_s']}s | rc={r['rc']}")

    wall = round(sum(r["wall_s"] for r in rows), 2)
    tps = [r["tokens_per_sec"] for r in rows if r["tokens_per_sec"]]
    whole = {
        "name": args.name,
        "source_repo": args.source_repo,
        "model_file": model_path.name,
        "model_gb": model_gb,
        "mmproj_file": mmproj_path.name,
        "system": info,
        "max_new_tokens": args.max_new_tokens,
        "wall_s_total": wall,
        "avg_tokens_per_sec": round(sum(tps) / len(tps), 2) if tps else None,
        "rows": rows,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    md = f"""# Vision Benchmark Report

- **Model:** `{args.name}` (`{args.source_repo}`)
- **Files:** {model_path.name} ({model_gb} GB) + {mmproj_path.name}
- **max_new_tokens:** {args.max_new_tokens}

## System

| Property | Value |
|---|---|
| Machine | {info['machine']} |
| Cores | {info['cpu_cores']} |
| RAM | {info['total_ram_gb']} GB |
| Python | {info['python']} |

## Aggregate

| Metric | Value |
|---|---|
| Tasks | {len(rows)} |
| Wall clock | {wall}s |
| Avg throughput | {whole['avg_tokens_per_sec']} tok/s |

## Per-task results

| # | Image | Out tok | eval (ms) | tok/s | peak RSS (MB) | wall (s) |
|---|---|---|---|---|---|---|
"""
    for r in rows:
        md += f"| {r['index']} | {r['image']} | {r['output_tokens']} | {r['eval_ms']} | {r['tokens_per_sec']} | {r['peak_rss_mb']} | {r['wall_s']} |\n"
    md += "\n## Responses\n\n"
    for r in rows:
        md += f"### Task {r['index']}: {r['image']}\n> {r['prompt']}\n\n```text\n{r['response']}\n```\n\n"

    md_path = os.path.join(args.report_dir, f"report_{args.name}_{ts}.md")
    json_path = os.path.join(args.report_dir, f"benchmark_{args.name}_{ts}.json")
    with open(md_path, "w") as f:
        f.write(md)
    with open(json_path, "w") as f:
        json.dump(whole, f, indent=2)

    print("=== SUMMARY ===")
    print(f"{args.name}        : {whole['avg_tokens_per_sec']} tok/s avg | {wall}s wall | model {model_gb} GB")
    print(f"report        : {md_path}")
    print(f"json          : {json_path}")


if __name__ == "__main__":
    main()