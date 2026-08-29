#!/usr/bin/env python3
"""Push the project README as the HuggingFace model card for the SPACE repo.

Rewrites the GitHub README into a neutral model card (no internal framing).
"""

import os
import re
import pathlib

from huggingface_hub import HfApi

FRONTMATTER = """---
license: other
pipeline_tag: image-text-to-text
tags:
  - text-generation
  - image-text-to-text
quantized_by: Nexuss0781
---
"""

# Neutral wording for sensitive/internal phrases.
REWRITES = [
    (r"Micro-tier perception stack for a self-improving AI",
     "Micro-tier AI stack"),
    (r"self-improving AI system", "system"),
    (r"AGI", "AI"),
]


def build_card(github_readme: str) -> str:
    body = github_readme
    for pattern, repl in REWRITES:
        body = re.sub(pattern, repl, body, flags=re.IGNORECASE)
    body = re.sub(r"\(https://huggingface\.co/Nexuss0781/SPACE\)", "", body)
    body = re.sub(
        r"Full reports: `reports/vision-benchmark\.md`\.",
        "", body.rstrip()).rstrip()
    return FRONTMATTER + "\n" + body.strip() + "\n"


def main():
    token = os.environ["HF_TOKEN"]
    username = os.environ["HF_USERNAME"]
    target_id = f"{username}/{os.environ['TARGET_REPO']}"

    github_readme = pathlib.Path("README.md").read_text()
    card = build_card(github_readme)

    card_path = pathlib.Path("HF_README.md")
    card_path.write_text(card)

    api = HfApi(token=token)
    commit = api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=target_id,
        commit_message="Add model card (neutral wording)",
    )
    print(f"[card] README commit: {commit.commit_url}")
    print(f"[card] DONE: https://huggingface.co/{target_id}")


if __name__ == "__main__":
    main()