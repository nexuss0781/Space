#!/usr/bin/env python3
"""SmolLM3-3B CPU/GPU benchmark: clone model, run N prompts, record RAM + performance."""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import psutil

try:
    import torch
except ModuleNotFoundError:
    torch = None


def system_info():
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": subprocess.getoutput("grep 'model name' /proc/cpuinfo | head -1").split(":")[-1].strip() if os.path.exists("/proc/cpuinfo") else platform.processor(),
        "cpu_cores": os.cpu_count(),
        "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "python": platform.python_version(),
        "torch": (torch.__version__ if torch else "not installed"),
        "cuda_available": bool(torch and torch.cuda.is_available()),
    }
    if torch and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


def peak_rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)


def rss_mb():
    return round(psutil.Process().memory_info().rss / 1e6, 2)


def load_transformers(model_id, local_dir, device, dtype, hf_token):
    os.environ.setdefault("HF_TOKEN", hf_token or "")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(local_dir or model_id)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir or model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    load_s = time.perf_counter() - t0

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "backend": "transformers",
        "dtype": str(dtype).split("(")[-1].split(")")[0],
        "model": model,
        "tokenizer": tokenizer,
        "load_s": round(load_s, 2),
        "weights_gb": round((param_bytes + buf_bytes) / 1e9, 2),
        "device": device,
    }


def load_llamacpp(gguf_dir, gguf_repo, gguf_pattern, hf_token):
    os.environ.setdefault("HF_TOKEN", hf_token or "")
    from huggingface_hub import snapshot_download
    from llama_cpp import Llama

    t0 = time.perf_counter()
    local = snapshot_download(
        repo_id=gguf_repo,
        allow_patterns=gguf_pattern,
        local_dir=gguf_dir,
    )
    gguf_path = next(
        f for f in _walk_gguf(local) if "q4_k_m" in f.lower() or "Q4_K_M" in f
    )
    model = Llama(model_path=gguf_path, n_ctx=2048, verbose=False)
    load_s = time.perf_counter() - t0
    gguf_size = round(os.path.getsize(gguf_path) / 1e9, 2)
    return {
        "backend": "llama-cpp (GGUF q4_k_m)",
        "model": model,
        "gguf_path": gguf_path,
        "gguf_gb": gguf_size,
        "load_s": round(load_s, 2),
        "weights_gb": gguf_size,
    }


def _walk_gguf(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            yield os.path.join(dirpath, f)


def gen_transformers(ctx, prompt, max_new_tokens):
    tokenizer, model = ctx["tokenizer"], ctx["model"]
    device = ctx["device"]
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt",
    ).to(device)
    prompt_tokens = inputs["input_ids"].shape[-1]

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.6, top_p=0.95,
                                 pad_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - start

    new_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    out_tokens = new_ids.shape[0]
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    ttft = _ttft_estimate(ctx, prompt, device)
    return prompt_tokens, out_tokens, elapsed, ttft, response


def _ttft_estimate(ctx, prompt, device):
    tokenizer, model = ctx["tokenizer"], ctx["model"]
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt",
    ).to(device)
    start = time.perf_counter()
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=1,
                           pad_token_id=tokenizer.eos_token_id)
    return time.perf_counter() - start


def gen_llamacpp(ctx, prompt, max_new_tokens):
    model = ctx["model"]
    start = time.perf_counter()
    res = model.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0.6,
        top_p=0.95,
    )
    elapsed = time.perf_counter() - start
    response = res["choices"][0]["message"]["content"]
    out_tokens = res["usage"]["completion_tokens"]
    prompt_tokens = res["usage"]["prompt_tokens"]
    ttft = 0.0
    return prompt_tokens, out_tokens, elapsed, ttft, response


def toks_per_sec(out_tokens, elapsed):
    return round(out_tokens / elapsed, 2) if elapsed > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--local-dir", default="models/SmolLM3-3B")
    ap.add_argument("--gguf-local-dir", default="gguf/SmolLM2-360M-Instruct")
    ap.add_argument("--gguf-repo", default="QuantFactory/SmolLM2-360M-Instruct-GGUF")
    ap.add_argument("--gguf-pattern", default="*Q4_K_M*.gguf")
    ap.add_argument("--prompts-file", default="benchmarks/smollm3/prompts.txt")
    ap.add_argument("--report-dir", default="results")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--force-gguf", action="store_true")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    os.makedirs(args.report_dir, exist_ok=True)

    info = system_info()
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    avail_ram_gb = round(psutil.virtual_memory().available / 1e9, 2)
    use_gguf = args.force_gguf or (torch is None) or (device == "cpu" and avail_ram_gb < 9.0)

    print(f"[benchmark] machine: {info['machine']} | {info['cpu']} | {info['cpu_cores']} cores")
    print(f"[benchmark] total RAM: {info['total_ram_gb']} GB | available: {avail_ram_gb} GB | device: {device}")
    print(f"[benchmark] backend choice: {'llama-cpp GGUF' if use_gguf else 'transformers'} (force_gguf={args.force_gguf})")

    if use_gguf:
        ctx = load_llamacpp(args.gguf_local_dir, args.gguf_repo, args.gguf_pattern, hf_token)
        gen = gen_llamacpp
    else:
        dtype = torch.bfloat16 if device == "cpu" else torch.float16
        ctx = load_transformers(args.model_id, args.local_dir, device, dtype, hf_token)
        gen = gen_transformers

    print(f"[benchmark] loaded '{args.model_id}' via {ctx['backend']} in {ctx['load_s']}s"
          f" | weights: {ctx['weights_gb']} GB | peak RSS so far: {peak_rss_mb()} MB")

    with open(args.prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    prompts = prompts[:10]
    print(f"[benchmark] running {len(prompts)} prompts (max_new_tokens={args.max_new_tokens})")

    rows = []
    t_all_start = time.perf_counter()
    for i, p in enumerate(prompts, 1):
        p_tok, o_tok, elapsed, ttft, resp = gen(ctx, p, args.max_new_tokens)
        tps = toks_per_sec(o_tok, elapsed)
        rows.append({
            "index": i,
            "prompt": p,
            "response": resp,
            "prompt_tokens": p_tok,
            "output_tokens": o_tok,
            "ttft_s": round(ttft, 3),
            "total_s": round(elapsed, 3),
            "tokens_per_sec": tps,
            "rss_mb": rss_mb(),
        })
        print(f"  [{i:02d}] out={o_tok:4d} tok | ttft={ttft:6.2f}s | total={elapsed:7.2f}s | {tps:7.2f} tok/s | rss={rss_mb()} MB")

    wall_s = round(time.perf_counter() - t_all_start, 2)
    peak = peak_rss_mb()

    avg_tps = round(sum(r["tokens_per_sec"] for r in rows) / len(rows), 2)
    avg_total = round(sum(r["total_s"] for r in rows) / len(rows), 3)
    total_tokens = sum(r["output_tokens"] for r in rows)

    summary = {
        "model_id": args.model_id,
        "backend": ctx["backend"],
        "dtype": ctx.get("dtype", "q4_k_m"),
        "weights_gb": ctx["weights_gb"],
        "load_s": ctx["load_s"],
        "device": device,
        "wall_clock_s": wall_s,
        "total_output_tokens": total_tokens,
        "avg_tokens_per_sec": avg_tps,
        "avg_total_s_per_prompt": avg_total,
        "peak_rss_mb": peak,
        "system": info,
    }

    report = render_report(args, info, ctx, prompts, rows, summary)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    md_path = os.path.join(args.report_dir, f"report_{stamp}.md")
    json_path = os.path.join(args.report_dir, f"benchmark_{stamp}.json")
    with open(md_path, "w") as f:
        f.write(report)
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"backend       : {summary['backend']} ({summary['dtype']})")
    print(f"weights       : {ctx['weights_gb']} GB | load: {ctx['load_s']}s")
    print(f"10 prompts    : wall {wall_s}s | avg {avg_total}s/prompt")
    print(f"throughput    : {avg_tps} tok/s (avg) | {total_tokens} tokens total")
    print(f"peak RAM      : {peak} MB (process RSS) of {info['total_ram_gb']} GB system")
    print(f"report        : {md_path}")
    print(f"json          : {json_path}")

    if not use_gguf and device == "cpu" and avail_ram_gb >= 9.0:
        print("\nNOTE: runner RAM >= 9GB, used full-precision transformers path.")
    print("\n::set-output name=peak_rss_mb::%d" % peak)


def render_report(args, info, ctx, prompts, rows, summary):
    lines = []
    lines.append(f"# SmolLM3-3B Benchmark Report")
    lines.append(f"")
    lines.append(f"- **Model:** `{args.model_id}`")
    lines.append(f"- **Backend:** {ctx['backend']} (dtype {ctx.get('dtype', 'q4_k_m')})")
    lines.append(f"- **Weights:** {ctx['weights_gb']} GB on disk | load time {ctx['load_s']}s")
    lines.append(f"- **Device:** {summary['device']}")
    lines.append(f"- **max_new_tokens:** {args.max_new_tokens}")
    lines.append(f"")
    lines.append(f"## System")
    lines.append(f"")
    lines.append(f"| Property | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| OS | {info['platform']} |")
    lines.append(f"| Machine | {info['machine']} |")
    lines.append(f"| CPU | {info['cpu']} |")
    lines.append(f"| Cores | {info['cpu_cores']} |")
    lines.append(f"| RAM (total) | {info['total_ram_gb']} GB |")
    lines.append(f"| Python | {info['python']} |")
    lines.append(f"| PyTorch | {info['torch']} |")
    if "gpu" in info:
        lines.append(f"| GPU | {info['gpu']} ({info['gpu_vram_gb']} GB) |")
    lines.append(f"")
    lines.append(f"## Memory")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Model weights | {ctx['weights_gb']} GB |")
    lines.append(f"| Peak process RSS | {summary['peak_rss_mb']} MB |")
    lines.append(f"| Report includes 10 prompts | yes |")
    lines.append(f"")
    lines.append(f"## Aggregate")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Wall clock (10 prompts) | {summary['wall_clock_s']}s |")
    lines.append(f"| Output tokens total | {summary['total_output_tokens']} |")
    lines.append(f"| Avg per prompt | {summary['avg_total_s_per_prompt']}s |")
    lines.append(f"| Avg throughput | {summary['avg_tokens_per_sec']} tok/s |")
    lines.append(f"")
    lines.append(f"## Per-prompt results")
    lines.append(f"")
    lines.append(f"| # | Prompt | Prompt tok | Out tok | TTFT (s) | Total (s) | tok/s |")
    lines.append(f"|---|---|---|---|---|---|---|")
    for r in rows:
        short = r["prompt"][:60].replace("|", "/")
        lines.append(f"| {r['index']} | {short} | {r['prompt_tokens']} | {r['output_tokens']} | {r['ttft_s']} | {r['total_s']} | {r['tokens_per_sec']} |")
    lines.append(f"")
    lines.append(f"## Full transcript")
    lines.append(f"")
    for r in rows:
        lines.append(f"### Prompt {r['index']}")
        lines.append(f"")
        lines.append(f"> {r['prompt']}")
        lines.append(f"")
        lines.append(f"**Response:** ({r['output_tokens']} tokens, {r['tokens_per_sec']} tok/s)")
        lines.append(f"")
        lines.append(f"```text")
        lines.append(r["response"])
        lines.append(f"```")
        lines.append(f"")
    return "\n".join(lines)


if __name__ == "__main__":
    main()