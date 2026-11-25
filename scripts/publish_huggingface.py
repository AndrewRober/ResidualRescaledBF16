"""Publish `ResidualRescaledBF16` Research Monograph, Code, Figures & Verified Scoreboards to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

HF_FRONTMATTER = """---
license: mit
library_name: pytorch
tags:
  - bfloat16
  - mixed-precision
  - iterative-refinement
  - diffusion
  - flow-matching
  - spectral-methods
  - cuda
  - ptx
  - tensor-cores
---

"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="AndrewR123/ResidualRescaledBF16")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)

    readme_body = (ROOT / "README.md").read_text(encoding="utf-8")
    hf_card = HF_FRONTMATTER + readme_body

    api.upload_file(
        path_or_fileobj=hf_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload ResidualRescaledBF16 research card",
    )

    for folder in ("src", "scripts", "results", "notebooks", "paper"):
        p = ROOT / folder
        if p.exists():
            api.upload_folder(
                folder_path=str(p),
                path_in_repo=folder,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Upload {folder}/",
            )

    for single in ("RESULTS_REPORT.md", "CITATION.cff", "pyproject.toml", "requirements.txt", "LICENSE", "SHA256SUMS.txt"):
        sp = ROOT / single
        if sp.exists():
            api.upload_file(
                path_or_fileobj=str(sp),
                path_in_repo=single,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Upload {single}",
            )

    print(f"[HuggingFace] LIVE: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
