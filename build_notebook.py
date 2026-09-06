#!/usr/bin/env python3
"""Собирает Jupyter/Colab-ноутбук из src/analysis.py.

analysis.py написан в percent-формате (`# %%` / `# %% [markdown]`), поэтому он
одновременно является исполняемым скриптом и источником ноутбука — код не дублируется
и не расходится между двумя артефактами.

    python build_notebook.py
"""
import json
import pathlib
import re

SRC = pathlib.Path("src/analysis.py")
OUT = pathlib.Path("notebooks/Dev_Estimations_Analysis.ipynb")

CELL_RE = re.compile(r"^# %%(?P<md> \[markdown\])?\s*$")


def parse_cells(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if body.strip():
            cells.append((kind, body))

    for line in text.splitlines():
        m = CELL_RE.match(line)
        if m:
            flush()
            kind = "markdown" if m.group("md") else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def to_nb(cells):
    out = []
    for kind, body in cells:
        if kind == "markdown":
            # снимаем "# " с начала строк комментария
            src = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.splitlines())
        else:
            src = body
        out.append({
            "cell_type": kind,
            "metadata": {},
            "source": [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]],
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
        })
    return {
        "cells": out,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    cells = parse_cells(SRC.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(to_nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    md = sum(1 for k, _ in cells if k == "markdown")
    print(f"{OUT}: {len(cells)} ячеек ({md} markdown, {len(cells) - md} code)")
