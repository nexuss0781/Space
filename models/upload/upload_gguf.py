#!/usr/bin/env python3
"""Upload the benchmarked Q4_K_M GGUF model to HuggingFace Hub.

Reads everything from the environment (set by the GitHub Actions matrix):
  HF_TOKEN, HF_USERNAME, REPO_NAME, SOURCE_REPO, GGUF_PATTERN,
  BASE_MODEL, MODEL_LICENSE
"""

import os
import pathlib

from huggingface_hub import HfApi, snapshot_download


def main():
    token = os.environ["HF_TOKEN"]
    username = os.environ["HF_USERNAME"]
    repo_name = os.environ["REPO_NAME"]
    source = os.environ["SOURCE_REPO"]
    pattern = os.environ["GGUF_PATTERN"]
    base_model = os.environ["BASE_MODEL"]
    model_license = os.environ["MODEL_LICENSE"]

    repo_id = f"{username}/{repo_name}"
    api = HfApi(token=token)

    who = api.whoami()
    print(f"[upload] authenticated as {who['name']}")

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"[upload] repo ok: https://huggingface.co/{repo_id}")

    local = snapshot_download(
        repo_id=source,
        allow_patterns=pattern,
        local_dir=f"models/{repo_name}",
    )
    ggufs = sorted(p for p in pathlib.Path(local).rglob("*.gguf"))
    assert ggufs, "no .gguf matched"
    gguf = ggufs[0]
    print(f"[upload] gguf: {gguf} ({gguf.stat().st_size / 1e9:.2f} GB)")

    readme = f"""---
license: {model_license}
base_model: {base_model}
quantized_by: Nexuss0781
---

# {repo_name}

4-bit k-quantized (Q4_K_M) GGUF of **{base_model}**.

- Quant type: Q4_K_M (llama.cpp)
- Source quantizer: `{source}`
- Uploaded from the Space GitHub Actions benchmark pipeline.
- See `Space` repo: https://github.com/nexuss0781/Space
"""

    readme_path = pathlib.Path("README.md")
    readme_path.write_text(readme)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )

    commit = api.upload_file(
        path_or_fileobj=str(gguf),
        path_in_repo=gguf.name,
        repo_id=repo_id,
        commit_message=f"Upload {gguf.name} (Q4_K_M)",
    )
    print(f"[upload] commit: {commit.commit_hash}")
    print(f"[upload] DONE: https://huggingface.co/{repo_id}/blob/main/{gguf.name}")


if __name__ == "__main__":
    main()