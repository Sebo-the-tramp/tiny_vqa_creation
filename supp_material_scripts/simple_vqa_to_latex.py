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


def build_latex(data: dict) -> str:
    lines = [
        "% Auto-generated from simple_vqa.json",
        "% Requires \\usepackage{multirow}, \\usepackage{array}, \\usepackage{longtable}",
        r"\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}",
        r"\newcommand{\questionRow}[4]{%",
        r"\multirow{2}{*}{#1}",
        r"& \multirow{2}{*}{#2}",
        r"& \textbf{Single image:} #3 \\ \cline{3-3}",
        r"& & \textbf{Multi image:} #4 \\ \hline",
        r"}",
        "",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{longtable}{|l|l|L{0.6\linewidth}|}",
        r"\caption{Question templates grouped by category and sub-category} \\",
        r"\hline",
        r"\textbf{Category} & \textbf{Sub-category} & \textbf{Question} \\ \hline",
        r"\endfirsthead",
        r"\hline",
        r"\textbf{Category} & \textbf{Sub-category} & \textbf{Question} \\ \hline",
        r"\endhead",
    ]

    for cat_key, cat_items in data.items():
        sub_groups = {}
        sub_order = []
        for qid, item in cat_items.items():
            sub = item.get("sub_category", "Uncategorized")
            if sub not in sub_groups:
                sub_groups[sub] = []
                sub_order.append(sub)
            sub_groups[sub].append((qid, item))

        for sub in sub_order:
            for qid, item in sub_groups[sub]:
                single = item.get("question_single", "") or ""
                multi = item.get("question_multi", "") or ""
                lines.append(
                    r"\questionRow{"
                    + latex_escape(title_from_key(cat_key))
                    + "}{"
                    + latex_escape(title_from_key(sub))
                    + "}{"
                    + latex_escape(single)
                    + "}{"
                    + latex_escape(multi)
                    + "}"
                )
            lines.append("")

    lines.append(r"\end{longtable}")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert simple_vqa.json to a LaTeX file with grouped question templates."
    )
    parser.add_argument(
        "--input_json",
        default="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/simple_vqa.json",
        help="Path to simple_vqa.json",
    )
    parser.add_argument(
        "--out_latex",
        default="simple_vqa_questions.tex",
        help="Output LaTeX file path",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input_json).read_text())
    latex = build_latex(data)
    Path(args.out_latex).write_text(latex)
    print(f"Wrote LaTeX to: {args.out_latex}")


if __name__ == "__main__":
    main()
