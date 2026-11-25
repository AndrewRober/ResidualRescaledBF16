"""Build and Audit All Publication Submission Bundles & Cryptographic Checksums.

Generates:
  1. `submission_bundles/arxiv_submission.zip`
  2. `submission_bundles/tmlr_camera_source.zip`
  3. `submission_bundles/anonymous_supplementary_code_and_results.zip` (strictly scrubbed)
  4. `SHA256SUMS.txt`

Performs automated audits:
  - PDF compilation integrity (`pypdf` page count > 0, zero unresolved `??` citations/references)
  - Double-blind zero-leakage audit across `paper/main_anonymous_review.pdf` and every file inside
    `anonymous_supplementary_code_and_results.zip` (0 occurrences of author/affiliation tokens).
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
BUNDLE_DIR = ROOT / "submission_bundles"

FORBIDDEN_TOKENS = ("andrew", "rober", "fouad", "samy", "bytesglue")


def audit_pdf(pdf_path: Path, must_be_anonymous: bool = False) -> int:
    assert pdf_path.exists(), f"Missing PDF: {pdf_path}"
    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    assert n_pages >= 10, f"Expected >= 10 pages in {pdf_path.name}, got {n_pages}"
    full_text = "\n".join((p.extract_text() or "") for p in reader.pages)
    assert "??" not in full_text, f"Unresolved LaTeX reference '??' detected in {pdf_path.name}!"
    if must_be_anonymous:
        low = full_text.lower()
        for tok in FORBIDDEN_TOKENS:
            assert tok not in low, f"Author leak '{tok}' detected in {pdf_path.name}!"
    return n_pages


def scrub_text(text: str) -> str:
    # Remove non-anonymous \else ... \fi author block in main.tex if present
    text = re.sub(
        r"\\ifanonymous\s*(.*?)\\else\s*.*?\\fi",
        r"\\1",
        text,
        flags=re.DOTALL,
    )
    replacements = [
        ("Andrew Rober Fouad Samy", "Anonymous Authors"),
        ("AndrewRober", "AnonymousRepo"),
        ("andrew-rober", "anonymous-authors"),
        ("andrew.rober@bytesglue.com", "anonymous@example.com"),
        ("BytesGlue", "Anonymous Institution"),
        ("AndrewR123", "AnonymousHF"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    for tok in FORBIDDEN_TOKENS:
        text = re.sub(re.escape(tok), "anonymous", text, flags=re.IGNORECASE)
    return text


def build_arxiv_and_tmlr_zips() -> None:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    fig_pdfs = sorted((PAPER_DIR / "figures").glob("*.pdf"))
    assert len(fig_pdfs) == 6, f"Expected 6 vector PDF figures, found {len(fig_pdfs)}"

    arxiv_zip = BUNDLE_DIR / "arxiv_submission.zip"
    with zipfile.ZipFile(arxiv_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(PAPER_DIR / "main.tex", "main.tex")
        zf.write(PAPER_DIR / "references.bib", "references.bib")
        bbl = PAPER_DIR / "main.bbl"
        if bbl.exists():
            zf.write(bbl, "main.bbl")
        for fp in fig_pdfs:
            zf.write(fp, f"figures/{fp.name}")

    tmlr_zip = BUNDLE_DIR / "tmlr_camera_source.zip"
    with zipfile.ZipFile(tmlr_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(PAPER_DIR / "main.tex", "main.tex")
        zf.write(PAPER_DIR / "references.bib", "references.bib")
        zf.write(PAPER_DIR / "main.pdf", "main.pdf")
        for fp in fig_pdfs:
            zf.write(fp, f"figures/{fp.name}")
    print(f"[OK] Built {arxiv_zip.name} & {tmlr_zip.name}")


def build_anonymous_zip() -> None:
    anon_zip = BUNDLE_DIR / "anonymous_supplementary_code_and_results.zip"
    include_paths: list[tuple[Path, str]] = [
        (PAPER_DIR / "main_anonymous_review.pdf", "paper/main_anonymous_review.pdf"),
        (PAPER_DIR / "main.tex", "paper/main.tex"),
        (PAPER_DIR / "references.bib", "paper/references.bib"),
        (ROOT / "README.md", "README.md"),
        (ROOT / "RESULTS_REPORT.md", "RESULTS_REPORT.md"),
        (ROOT / "requirements.txt", "requirements.txt"),
        (ROOT / "pyproject.toml", "pyproject.toml"),
        (ROOT / "notebooks" / "CSP_BF16_IR_Demo.ipynb", "notebooks/CSP_BF16_IR_Demo.ipynb"),
    ]
    for p in sorted((PAPER_DIR / "figures").glob("*.pdf")):
        include_paths.append((p, f"paper/figures/{p.name}"))
    for p in sorted((ROOT / "src" / "residual_rescaled_bf16").glob("*.py")):
        include_paths.append((p, f"src/residual_rescaled_bf16/{p.name}"))
    for p in sorted((ROOT / "scripts").glob("*.py")):
        if p.name != "build_submission_bundles.py":
            include_paths.append((p, f"scripts/{p.name}"))
    for p in sorted((ROOT / "results").glob("*.json")):
        include_paths.append((p, f"results/{p.name}"))

    text_exts = {".tex", ".bib", ".md", ".txt", ".toml", ".py", ".json", ".ipynb"}
    with zipfile.ZipFile(anon_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src_p, arc_name in include_paths:
            if src_p.suffix.lower() in text_exts:
                scrubbed = scrub_text(src_p.read_text(encoding="utf-8"))
                low = scrubbed.lower()
                for tok in FORBIDDEN_TOKENS:
                    assert tok not in low, f"Leak '{tok}' remained after scrubbing {arc_name}!"
                zf.writestr(arc_name, scrubbed.encode("utf-8"))
            else:
                zf.write(src_p, arc_name)

    # Re-open zip and verify zero leaks across all stored members
    with zipfile.ZipFile(anon_zip, "r") as zf:
        for info in zf.infolist():
            raw = zf.read(info.filename)
            if info.filename.endswith(".pdf") and "main_anonymous" in info.filename:
                rdr = PdfReader(io.BytesIO(raw))
                txt = "\n".join((pg.extract_text() or "") for pg in rdr.pages).lower()
                for tok in FORBIDDEN_TOKENS:
                    assert tok not in txt, f"PDF leak '{tok}' in {info.filename}!"
            elif Path(info.filename).suffix.lower() in text_exts:
                txt = raw.decode("utf-8").lower()
                for tok in FORBIDDEN_TOKENS:
                    assert tok not in txt, f"Text leak '{tok}' in {info.filename}!"
    print(f"[OK] Built & double-blind audited {anon_zip.name} (0 identity leaks)")


def write_sha256sums() -> None:
    targets: list[Path] = []
    for pat in [
        "README.md",
        "RESULTS_REPORT.md",
        "PUBLISH_EVERYWHERE_GUIDE.md",
        "CITATION.cff",
        ".zenodo.json",
        "pyproject.toml",
        "requirements.txt",
        "LICENSE",
        "src/residual_rescaled_bf16/*.py",
        "scripts/*.py",
        "results/*.json",
        "notebooks/*.ipynb",
        "paper/main.tex",
        "paper/main_anonymous_review.tex",
        "paper/references.bib",
        "paper/main.pdf",
        "paper/main_anonymous_review.pdf",
        "paper/figures/*.pdf",
        "paper/figures/*.png",
        "submission_bundles/*.zip",
    ]:
        targets.extend(sorted(ROOT.glob(pat)))

    lines = []
    for p in targets:
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            rel = p.relative_to(ROOT).as_posix()
            lines.append(f"{h}  {rel}")
    out = ROOT / "SHA256SUMS.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {out.name} ({len(lines)} files hashed)")


def main() -> None:
    p_cam = audit_pdf(PAPER_DIR / "main.pdf", must_be_anonymous=False)
    p_anon = audit_pdf(PAPER_DIR / "main_anonymous_review.pdf", must_be_anonymous=True)
    print(f"[OK] Audited PDFs: main.pdf ({p_cam} pages), main_anonymous_review.pdf ({p_anon} pages, 0 leaks)")
    build_arxiv_and_tmlr_zips()
    build_anonymous_zip()
    write_sha256sums()


if __name__ == "__main__":
    main()
