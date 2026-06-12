# ==============================================================================
# tools/bump_version.py
# Usage: python tools/bump_version.py 0.2.0
# Updates version in pyproject.toml and src/floudsonnx/__init__.py
# ==============================================================================
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def bump(new_version: str) -> None:
    # Validate semver shape
    if not re.fullmatch(r"\d+\.\d+\.\d+", new_version):
        print(f"ERROR: '{new_version}' is not a valid semver (x.y.z)")
        sys.exit(1)

    # pyproject.toml
    pyproject = ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    text, n = re.subn(r'(^version\s*=\s*")[^"]+(")', rf"\g<1>{new_version}\2", text, flags=re.MULTILINE)
    if n == 0:
        print("ERROR: version field not found in pyproject.toml")
        sys.exit(1)
    pyproject.write_text(text, encoding="utf-8")
    print(f"  pyproject.toml   → {new_version}")

    # __init__.py
    init = ROOT / "src" / "floudsonnx" / "__init__.py"
    text = init.read_text(encoding="utf-8")
    text, n = re.subn(r'(__version__\s*=\s*")[^"]+(")', rf"\g<1>{new_version}\2", text)
    if n == 0:
        print("ERROR: __version__ not found in __init__.py")
        sys.exit(1)
    init.write_text(text, encoding="utf-8")
    print(f"  __init__.py      → {new_version}")

    print(f"\nDone. Version bumped to {new_version}")
    print("Next steps:")
    print("  git add pyproject.toml src/floudsonnx/__init__.py")
    print(f"  git commit -m 'chore: bump version to {new_version}'")
    print(f"  git tag v{new_version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/bump_version.py <new_version>")
        print("Example: python tools/bump_version.py 0.2.0")
        sys.exit(1)
    bump(sys.argv[1])
