from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".py", ".json", ".yaml", ".yml", ".sh", ".md", ".txt", ".csv",
    ".log", ".sbatch",
}

# These patterns identify user- or account-specific filesystem/login strings.
# They intentionally do not contain any real contributor account name.
PRIVATE_PATTERNS = (
    ("macOS user-home path", re.compile(r"(?<![A-Za-z0-9])/(?:Users)/[^/\s`\"']+/")),
    ("Linux user-home path", re.compile(r"(?<![A-Za-z0-9])/(?:home)/[^/\s`\"']+/")),
    ("institutional mounted-user path", re.compile(r"(?<![A-Za-z0-9])/mnt/iusers\d*/[^/\s`\"']+/")),
    ("network scratch account path", re.compile(r"(?<![A-Za-z0-9])/net/scratch/[^/\s`\"']+/")),
    ("Windows user profile path", re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\\\s`\"']+\\\\")),
    ("CSF login identity", re.compile(r"\b[A-Za-z0-9._-]+@csf\d*\.itservices\.manchester\.ac\.uk\b")),
    ("course-account namespace", re.compile(r"\bhum-msc-data-sci-\d{4}-\d{4}\b")),
)

# Validator source necessarily contains the patterns above.
SENTINEL_FILES = {
    Path("validation/validate_public_privacy.py"),
    Path("validation/validate_static.py"),
}

issues: list[tuple[str, str]] = []


def add_issue(path: Path, label: str, match: str) -> None:
    rel = path.relative_to(ROOT)
    issues.append((str(rel), f"{label}: {match[:160]}"))


def scan_text(path: Path) -> None:
    rel = path.relative_to(ROOT)
    if rel in SENTINEL_FILES:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for label, pattern in PRIVATE_PATTERNS:
        found = pattern.search(text)
        if found:
            add_issue(path, label, found.group(0))


def scan_binary_for_path_prefixes(path: Path) -> None:
    # Check model/checkpoint-like binaries for path prefixes without parsing the
    # file format. This catches accidental pickled absolute paths.
    prefixes = (
        b"/Users/",
        b"/home/",
        b"/mnt/iusers",
        b"/net/scratch/",
        b"\\Users\\",
        b"@csf",
    )
    overlap = max(len(x) for x in prefixes) - 1
    carry = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            data = carry + chunk
            for prefix in prefixes:
                if prefix in data:
                    rel = path.relative_to(ROOT)
                    issues.append((str(rel), f"binary private-path prefix: {prefix!r}"))
                    return
            carry = data[-overlap:] if overlap > 0 else b""


# Source/public text plus any generated outputs included in a future release.
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(ROOT)

    # Provider raw data and processed intermediates are never distributable
    # package content and may be large/binary.
    if rel == Path("data/raw") or Path("data/raw") in rel.parents:
        continue
    if rel == Path("data/processed") or Path("data/processed") in rel.parents:
        continue
    if rel.parts and rel.parts[0].startswith(".venv"):
        continue
    if ".git" in rel.parts:
        continue

    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", "README.md"}:
        scan_text(path)

# If somebody later includes generated model/checkpoint files in the release,
# ensure pickled absolute paths do not silently reintroduce account information.
outputs = ROOT / "outputs"
if outputs.exists():
    for path in outputs.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".pt", ".pth", ".pkl", ".pickle"}:
            scan_binary_for_path_prefixes(path)

if issues:
    print("PUBLIC PRIVACY VALIDATION: FAIL")
    for path, message in issues:
        print(f"- {path}: {message}")
    raise SystemExit(1)

print("PUBLIC PRIVACY VALIDATION: PASS")
