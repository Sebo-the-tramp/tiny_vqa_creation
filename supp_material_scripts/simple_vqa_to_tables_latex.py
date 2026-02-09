#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path


def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def title_from_key(key: str) -> str:
    key = key.replace("_", " ")
    return re.sub(r"\b([a-z])", lambda m: m.group(1).upper(), key)


def build_tables(data: dict) -> str:
    lines = [
        "% Auto-generated from simple_vqa.json",
        r"\newcommand{\vqadetails}[3]{#1 & \texttt{#2} & \texttt{#3} \\}",
        "",
    ]

    for cat_key, cat_items in data.items():
        lines.append(r"\section*{" + latex_escape(title_from_key(cat_key)) + "}")

        sub_groups = {}
        sub_order = []
        for qid, item in cat_items.items():
            sub = item.get("sub_category", "Uncategorized")
            if sub not in sub_groups:
                sub_groups[sub] = []
                sub_order.append(sub)
            sub_groups[sub].append(qid)

        for sub in sub_order:
            lines.append(r"\subsection*{" + latex_escape(title_from_key(sub)) + "}")
            lines.append(r"\begin{table}[h]")
            lines.append(r"\centering")
            lines.append(r"\begin{tabular}{lll}")
            for i, qid in enumerate(sub_groups[sub], start=1):
                label = f"q:{qid}"
                lines.append(
                    r"\vqadetails{"
                    + latex_escape(f"Q{i}")
                    + "}{"
                    + latex_escape(qid)
                    + "}{"
                    + latex_escape(label)
                    + "}"
                )
            lines.append(r"\end{tabular}")
            lines.append(
                r"\caption{"
                + latex_escape(f"{title_from_key(cat_key)} — {title_from_key(sub)}")
                + "}"
            )
            lines.append(r"\label{tab:" + latex_escape(f"{cat_key}_{sub}") + "}")
            lines.append(r"\end{table}")
            lines.append("")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert simple_vqa.json to LaTeX tables grouped by category/sub-category."
    )
    parser.add_argument(
        "--input_json",
        default="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/simple_vqa.json",
        help="Path to simple_vqa.json",
    )
    parser.add_argument(
        "--out_latex",
        default="simple_vqa_tables.tex",
        help="Output LaTeX file path",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input_json).read_text())
    latex = build_tables(data)
    Path(args.out_latex).write_text(latex)
    print(f"Wrote LaTeX to: {args.out_latex}")


if __name__ == "__main__":
    main()
