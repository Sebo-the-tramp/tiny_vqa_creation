#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Iterable


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


def _fmt_params(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _iter_rows(items: Iterable[dict]) -> Iterable[dict]:
    for item in items:
        yield {
            "id": str(item.get("id", "")),
            "family": str(item.get("family", "")),
            "mode": str(item.get("mode", "")),
            "params_b": _fmt_params(item.get("params_b")),
            "release_year": str(item.get("release_year", "")),
            "license": str(item.get("license", "")),
            "release_type": str(item.get("release_type", "")),
            "source": str(item.get("source", "")),
        }


def build_latex(items: list[dict], caption: str) -> str:
    lines = [
        "% Auto-generated from metadata.json",
        r"% Requires: \usepackage{booktabs} and \usepackage{longtable}",
        r"\newcommand{\modelmetaRow}[8]{#1 & #2 & #3 & #4 & #5 & #6 & #7 & \texttt{#8} \\}",
        r"\newcommand{\printmodelmeta}{%",
        r"\small",
        r"\begin{longtable}{lll r l l l p{0.28\linewidth}}",
        r"\caption{" + latex_escape(caption) + r"}\label{tab:model_metadata}\\",
        r"\toprule",
        r"\textbf{ID} & \textbf{Family} & \textbf{Mode} & \textbf{Params(B)} & \textbf{Year} & \textbf{License} & \textbf{Release} & \textbf{Source} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{ID} & \textbf{Family} & \textbf{Mode} & \textbf{Params(B)} & \textbf{Year} & \textbf{License} & \textbf{Release} & \textbf{Source} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{8}{r}{\textit{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for row in _iter_rows(items):
        lines.append(
            r"\modelmetaRow{"
            + latex_escape(row["id"])
            + "}{"
            + latex_escape(row["family"])
            + "}{"
            + latex_escape(row["mode"])
            + "}{"
            + latex_escape(row["params_b"])
            + "}{"
            + latex_escape(row["release_year"])
            + "}{"
            + latex_escape(row["license"])
            + "}{"
            + latex_escape(row["release_type"])
            + "}{"
            + latex_escape(row["source"])
            + "}"
        )

    lines.extend(
        [
            r"\end{longtable}",
            r"}",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert metadata.json to a LaTeX table with reusable macros."
    )
    parser.add_argument(
        "--input_json",
        default="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/analysis/utils/metadata.json",
        help="Path to metadata.json",
    )
    parser.add_argument(
        "--out_latex",
        default="metadata_table.tex",
        help="Output LaTeX file path",
    )
    parser.add_argument(
        "--caption",
        default="Model metadata.",
        help="Table caption text",
    )
    args = parser.parse_args()

    items = json.loads(Path(args.input_json).read_text())
    latex = build_latex(items, args.caption)
    Path(args.out_latex).write_text(latex)
    print(f"Wrote LaTeX to: {args.out_latex}")


if __name__ == "__main__":
    main()
