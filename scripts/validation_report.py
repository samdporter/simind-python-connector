"""Render the recorded SIMIND validation results as a Markdown section."""

import argparse
import json
from pathlib import Path


BEGIN = "<!-- validation:begin -->"
END = "<!-- validation:end -->"


def _load(out_dir: Path) -> tuple[list[str], dict]:
    engines = []
    features: dict[str, dict[str, dict]] = {}
    for path in sorted(out_dir.glob("*.jsonl")):
        engines.append(path.stem)
        for line in path.read_text().splitlines():
            record = json.loads(line)
            case = features.setdefault(record["feature"], {}).setdefault(
                record["case"], {"expected": "—", "smoke": False, "engines": {}}
            )
            case["engines"][path.stem] = record
            case["smoke"] = case["smoke"] or record.get("smoke", False)
            if record.get("expected") is not None:
                case["expected"] = f"{record['expected']:.4g}"
            else:
                lower, upper = record.get("lower"), record.get("upper")
                if upper is None:
                    case["expected"] = f"≥ {lower:.4g}"
                elif lower is None:
                    case["expected"] = f"≤ {upper:.4g}"
                else:
                    case["expected"] = f"{lower:.4g}–{upper:.4g}"
    return engines, features


def _cell(record: dict | None) -> str:
    if record is None:
        return "—"
    if record["status"] != "run":
        return "not run"
    verdict = "pass" if record["passed"] else "FAIL"
    return f"{verdict} {record['measured']:.3g}"


def render(out_dir: Path) -> str:
    engines, features = _load(out_dir)
    lines = []
    for feature in sorted(features):
        lines += [f"### {feature}", ""]
        lines.append("| case | expected | " + " | ".join(engines) + " | smoke |")
        lines.append("|" + "---|" * (len(engines) + 3))
        for case, row in features[feature].items():
            cells = [_cell(row["engines"].get(engine)) for engine in engines]
            smoke = "yes" if row["smoke"] else "no"
            lines.append(
                f"| {case} | {row['expected']} | " + " | ".join(cells) + f" | {smoke} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_section(path: Path, section: str) -> None:
    text = path.read_text()
    block = f"{BEGIN}\n{section.rstrip()}\n{END}"
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        tail = rest.split(END, 1)[1]
        text = head + block + tail
    else:
        text = text.rstrip("\n") + f"\n\n## Validation results\n\n{block}\n"
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("output/validation"))
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    section = render(args.out_dir)
    if args.write:
        write_section(args.write, section)
    else:
        print(section, end="")


if __name__ == "__main__":
    main()
