"""Add cell number markers to the beginning of each cell in all notebooks."""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"

CODE_PREFIX = "# Cell {n}\n"
MD_PREFIX = "<!-- Cell {n} -->\n"


def already_numbered(source: list[str], n: int) -> bool:
    if not source:
        return False
    first = source[0]
    return first == CODE_PREFIX.format(n=n) or first == MD_PREFIX.format(n=n)


def add_numbers(nb_path: Path) -> int:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    changed = 0
    for i, cell in enumerate(nb["cells"], 1):
        src = cell.get("source", [])
        if isinstance(src, str):
            src = src.splitlines(keepends=True)
        if already_numbered(src, i):
            continue
        prefix = CODE_PREFIX if cell["cell_type"] == "code" else MD_PREFIX
        cell["source"] = [prefix.format(n=i)] + src
        changed += 1
    if changed:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main():
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    for nb_path in notebooks:
        n = add_numbers(nb_path)
        print(f"{nb_path.name}: {n} cells updated")


if __name__ == "__main__":
    main()
