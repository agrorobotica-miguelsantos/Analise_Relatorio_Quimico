"""Gera um PDF multipagina com as analises quimicas da amostragem Carbono.

Exemplo:
    python gerar_relatorio_pdf.py relatorio.xlsx

Dependencias adicionais ao aplicativo Streamlit:
    pip install matplotlib reportlab

O relatorio inclui todos os parametros com dados e todas as combinacoes de
parametros com pares validos. Mapas nao fazem parte deste gerador.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import textwrap
import unicodedata
import zipfile
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


PAGE_SIZE = landscape(A4)
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE
MARGIN = 34

GROUP_COLUMNS = ["Talhão", "Profundidade"]
INTERNAL_FILTER_COLUMN = "Amostragem"
DEPTH_ORDER = ["00-20", "20-40", "40-60"]

PARAMETER_META = {
    "S_(mg/dm3)": {"label": "Enxofre (S)", "short": "S", "unit": "mg/dm3", "ratio_scale": True},
    "P_resina_(mg/dm3)": {"label": "Fósforo - resina", "short": "P resina", "unit": "mg/dm3", "ratio_scale": True},
    "K_(mg/dm3)": {"label": "Potássio (K)", "short": "K mg/dm3", "unit": "mg/dm3", "ratio_scale": True},
    "P_(mg/dm3)": {"label": "Fósforo (P)", "short": "P", "unit": "mg/dm3", "ratio_scale": True},
    "Al_(cmolc/dm3)": {"label": "Alumínio (Al)", "short": "Al", "unit": "cmolc/dm3", "ratio_scale": True},
    "Ca_(cmolc/dm3)": {"label": "Cálcio (Ca)", "short": "Ca", "unit": "cmolc/dm3", "ratio_scale": True},
    "K_(cmolc/dm3)": {"label": "Potássio (K)", "short": "K cmolc/dm3", "unit": "cmolc/dm3", "ratio_scale": True},
    "Mg_(cmolc/dm3)": {"label": "Magnésio (Mg)", "short": "Mg", "unit": "cmolc/dm3", "ratio_scale": True},
    "B_(mg/dm3)": {"label": "Boro (B)", "short": "B", "unit": "mg/dm3", "ratio_scale": True},
    "Cu_(mg/dm3)": {"label": "Cobre (Cu)", "short": "Cu", "unit": "mg/dm3", "ratio_scale": True},
    "Fe_(mg/dm3)": {"label": "Ferro (Fe)", "short": "Fe", "unit": "mg/dm3", "ratio_scale": True},
    "Mn_(mg/dm3)": {"label": "Manganês (Mn)", "short": "Mn", "unit": "mg/dm3", "ratio_scale": True},
    "Zn_(mg/dm3)": {"label": "Zinco (Zn)", "short": "Zn", "unit": "mg/dm3", "ratio_scale": True},
    "MO_(g/dm3)": {"label": "Matéria orgânica (MO)", "short": "MO", "unit": "g/dm3", "ratio_scale": True},
    "pH CaCl2": {"label": "pH em CaCl2", "short": "pH CaCl2", "unit": "", "ratio_scale": False},
    "pH SMP": {"label": "pH SMP", "short": "pH SMP", "unit": "", "ratio_scale": False},
    "H+Al_(cmolc/dm3)": {"label": "Acidez potencial (H+Al)", "short": "H+Al", "unit": "cmolc/dm3", "ratio_scale": True},
}
PARAM_COLUMNS = list(PARAMETER_META)

CENSOR_POLICIES = {
    "missing": "considerar como ausente",
    "half": "usar metade do limite inferior",
    "limit": "usar o limite informado",
}

MISSING_MARKERS = {
    "", "-", "--", "na", "n/a", "nd", "n/d", "nan", "none",
    "não detectado", "nao detectado", "não determinado", "nao determinado",
}

DEPTH_COLORS = ["#2563eb", "#f97316", "#16a34a", "#9333ea", "#dc2626", "#0891b2"]
ACCENT = HexColor("#176b5b")
ACCENT_LIGHT = HexColor("#dff3ee")
INK = HexColor("#17212b")
MUTED = HexColor("#5f6b76")
GRID = HexColor("#dfe5e8")
PAPER = HexColor("#f7f9fb")


class ReportValidationError(ValueError):
    pass


def parameter_label(column: str) -> str:
    meta = PARAMETER_META[column]
    return f"{meta['label']} ({meta['unit']})" if meta["unit"] else meta["label"]


def parameter_short_label(column: str) -> str:
    return PARAMETER_META[column]["short"]


def normalize_header(value) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return re.sub(r"\s+", " ", text)


def column_key(value) -> str:
    text = unicodedata.normalize("NFKD", normalize_header(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", "", text).casefold()


def canonical_column_map() -> dict[str, str]:
    expected = [INTERNAL_FILTER_COLUMN] + GROUP_COLUMNS + PARAM_COLUMNS
    aliases = {column_key(column): column for column in expected}
    aliases.update(
        {
            column_key("Talhao"): "Talhão",
            column_key("Parcela"): "Talhão",
            column_key("Amostra"): "Amostragem",
        }
    )
    return aliases


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = canonical_column_map()
    renamed = []
    for original in df.columns:
        cleaned = normalize_header(original)
        duplicate_match = re.fullmatch(r"(.+)\.(\d+)", cleaned)
        lookup_name = duplicate_match.group(1) if duplicate_match else cleaned
        renamed.append(aliases.get(column_key(lookup_name), cleaned))

    duplicates = sorted({name for name in renamed if renamed.count(name) > 1})
    if duplicates:
        raise ReportValidationError(
            "Colunas duplicadas ou equivalentes: " + ", ".join(duplicates)
        )
    result = df.copy()
    result.columns = renamed
    return result


def _parse_locale_literal(text: str) -> float:
    cleaned = text.strip().replace("\u00a0", "").replace(" ", "").replace("−", "-")
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")

    pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    if not re.fullmatch(pattern, cleaned):
        raise ValueError("formato numérico inválido")
    value = float(cleaned)
    if not math.isfinite(value):
        raise ValueError("valor não finito")
    return value


def parse_numeric_value(value, censor_policy: str) -> tuple[float, str]:
    if value is None or pd.isna(value):
        return np.nan, "missing"
    if isinstance(value, (bool, np.bool_)):
        return np.nan, "invalid"
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return (number, "numeric") if math.isfinite(number) else (np.nan, "invalid")

    text = unicodedata.normalize("NFKC", str(value)).strip()
    if text.casefold() in MISSING_MARKERS:
        return np.nan, "missing_marker"
    censored = re.fullmatch(r"([<>≤≥])\s*(.+)", text)
    if censored:
        operator, literal = censored.groups()
        try:
            limit = _parse_locale_literal(literal)
        except ValueError:
            return np.nan, "invalid"
        status = "censored_low" if operator in {"<", "≤"} else "censored_high"
        if censor_policy == "missing":
            return np.nan, status
        if censor_policy == "half" and status == "censored_low":
            return limit / 2, status
        return limit, status
    try:
        return _parse_locale_literal(text), "parsed_text"
    except ValueError:
        return np.nan, "invalid"


def normalize_numeric_columns(
    data: pd.DataFrame, censor_policy: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = data.copy()
    quality_rows = []
    for column in [item for item in PARAM_COLUMNS if item in result.columns]:
        values = []
        counts: dict[str, int] = {}
        for raw_value in result[column]:
            number, status = parse_numeric_value(raw_value, censor_policy)
            values.append(number)
            counts[status] = counts.get(status, 0) + 1
        result[column] = pd.Series(values, index=result.index, dtype="float64")
        quality_rows.append(
            {
                "Parâmetro": parameter_short_label(column),
                "Válidos": int(result[column].notna().sum()),
                "Inválidos": counts.get("invalid", 0),
                "Limite inf.": counts.get("censored_low", 0),
                "Limite sup.": counts.get("censored_high", 0),
                "Ausentes marcados": counts.get("missing_marker", 0),
            }
        )
    return result, pd.DataFrame(quality_rows)


def text_filter(data: pd.DataFrame, column: str, selected: list[str] | None) -> pd.DataFrame:
    if not selected:
        return data
    if column not in data.columns:
        raise ReportValidationError(f"Não é possível filtrar: coluna {column} ausente.")
    normalized_selected = {str(value).strip().casefold() for value in selected}
    values = data[column].astype("string").str.strip().str.casefold()
    return data.loc[values.isin(normalized_selected)].copy()


def load_report(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not args.input.exists():
        raise ReportValidationError(f"Arquivo não encontrado: {args.input}")
    if args.input.suffix.casefold() not in {".xlsx", ".xlsm"}:
        raise ReportValidationError("Use um arquivo .xlsx ou .xlsm.")

    try:
        with pd.ExcelFile(args.input, engine="openpyxl") as workbook:
            sheets = workbook.sheet_names
        if not sheets:
            raise ReportValidationError("O arquivo não possui planilhas.")
        sheet_name = args.sheet or sheets[0]
        if sheet_name not in sheets:
            raise ReportValidationError(
                f"Aba '{sheet_name}' não encontrada. Disponíveis: {', '.join(sheets)}"
            )
        raw = pd.read_excel(args.input, sheet_name=sheet_name, engine="openpyxl")
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        raise ReportValidationError(f"Não foi possível abrir o Excel: {exc}") from exc

    normalized = normalize_columns(raw)
    if INTERNAL_FILTER_COLUMN not in normalized.columns:
        raise ReportValidationError("A coluna 'Amostragem' não foi encontrada.")
    mask = (
        normalized[INTERNAL_FILTER_COLUMN]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq(args.amostragem.strip().casefold())
        .fillna(False)
    )
    filtered = normalized.loc[mask].copy()
    if filtered.empty:
        raise ReportValidationError(
            f"Nenhuma linha encontrada para Amostragem = {args.amostragem}."
        )

    existing_params = [column for column in PARAM_COLUMNS if column in filtered.columns]
    if not existing_params:
        raise ReportValidationError("Nenhuma coluna de parâmetro químico foi encontrada.")
    keep = [column for column in GROUP_COLUMNS + existing_params if column in filtered.columns]
    filtered = filtered[keep].copy()
    filtered, quality = normalize_numeric_columns(filtered, args.censor_policy)
    filtered = text_filter(filtered, "Talhão", args.talhao)
    filtered = text_filter(filtered, "Profundidade", args.profundidade)
    if filtered.empty:
        raise ReportValidationError("Nenhuma linha permaneceu após os filtros.")
    return filtered, quality, sheet_name


def available_parameters(data: pd.DataFrame) -> list[str]:
    return [
        column
        for column in PARAM_COLUMNS
        if column in data.columns and data[column].notna().any()
    ]


def format_number(value, decimals: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return (
        f"{value:,.{decimals}f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def parameter_statistics(data: pd.DataFrame, column: str) -> dict[str, float | int]:
    values = data[column].dropna()
    mean = values.mean()
    std = values.std(ddof=1)
    cv = np.nan
    if PARAMETER_META[column]["ratio_scale"] and pd.notna(mean) and mean != 0:
        cv = std / mean * 100
    return {
        "n": int(values.count()),
        "mean": mean,
        "median": values.median(),
        "min": values.min(),
        "max": values.max(),
        "std": std,
        "cv": cv,
        "coverage": values.count() / len(data) * 100,
    }


def correlation_statistics(
    data: pd.DataFrame, x_column: str, y_column: str, min_pairs: int
) -> dict[str, float | int | bool | pd.DataFrame]:
    pair = data[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()
    result: dict[str, float | int | bool | pd.DataFrame] = {
        "pair": pair,
        "n": len(pair),
        "pearson": np.nan,
        "spearman": np.nan,
        "eligible": False,
    }
    if (
        len(pair) >= min_pairs
        and pair[x_column].nunique() >= 2
        and pair[y_column].nunique() >= 2
    ):
        result["eligible"] = True
        result["pearson"] = pair[x_column].corr(pair[y_column])
        result["spearman"] = pair[x_column].rank(method="average").corr(
            pair[y_column].rank(method="average")
        )
    return result


def ordered_depths(series: pd.Series) -> list[str]:
    values = sorted({str(value).strip() for value in series.dropna()})
    ordered = [value for value in DEPTH_ORDER if value in values]
    return ordered + [value for value in values if value not in ordered]


def depth_color_map(data: pd.DataFrame) -> dict[str, str]:
    if "Profundidade" not in data.columns:
        return {}
    depths = ordered_depths(data["Profundidade"])
    return {depth: DEPTH_COLORS[index % len(DEPTH_COLORS)] for index, depth in enumerate(depths)}


def apply_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": "#9aa5ad",
            "axes.grid": True,
            "grid.color": "#e4e9ec",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


class PdfReport:
    def __init__(self, output_path: Path, title: str):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.canvas = canvas.Canvas(str(output_path), pagesize=PAGE_SIZE, pageCompression=1)
        self.canvas.setTitle(title)
        self.canvas.setAuthor("Gerador de relatório químico")
        self.title = title
        self.page_number = 0
        self.page_open = False

    def _finish_page(self):
        if not self.page_open:
            return
        self.canvas.setStrokeColor(GRID)
        self.canvas.line(MARGIN, 27, PAGE_WIDTH - MARGIN, 27)
        self.canvas.setFont("Helvetica", 7.5)
        self.canvas.setFillColor(MUTED)
        self.canvas.drawString(MARGIN, 15, self.title)
        self.canvas.drawRightString(PAGE_WIDTH - MARGIN, 15, f"Página {self.page_number}")
        self.canvas.showPage()
        self.page_open = False

    def start_page(self, heading: str, section: str = "", bookmark: str | None = None):
        self._finish_page()
        self.page_number += 1
        self.page_open = True
        key = bookmark or f"page-{self.page_number}"
        self.canvas.bookmarkPage(key)
        self.canvas.addOutlineEntry(heading, key, level=0, closed=False)
        self.canvas.setFillColor(PAPER)
        self.canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
        self.canvas.setFillColor(ACCENT)
        self.canvas.rect(0, PAGE_HEIGHT - 58, PAGE_WIDTH, 58, fill=1, stroke=0)
        self.canvas.setFillColor(white)
        self.canvas.setFont("Helvetica-Bold", 17)
        self.canvas.drawString(MARGIN, PAGE_HEIGHT - 36, heading[:92])
        if section:
            self.canvas.setFont("Helvetica", 8)
            self.canvas.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 34, section)

    def draw_wrapped_text(
        self, text: str, x: float, y: float, width_chars: int, leading: float = 13
    ) -> float:
        self.canvas.setFont("Helvetica", 9)
        self.canvas.setFillColor(INK)
        for line in textwrap.wrap(text, width=width_chars):
            self.canvas.drawString(x, y, line)
            y -= leading
        return y

    def draw_figure(self, figure, x: float, y: float, width: float, height: float):
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)
        buffer.seek(0)
        self.canvas.drawImage(
            ImageReader(buffer),
            x,
            y,
            width=width,
            height=height,
            preserveAspectRatio=True,
            anchor="c",
            mask="auto",
        )

    def save(self):
        self._finish_page()
        self.canvas.save()


def draw_metric_cards(report: PdfReport, statistics: dict[str, float | int], y: float):
    cards = [
        ("Resultados", str(statistics["n"])),
        ("Média", format_number(statistics["mean"])),
        ("Mediana", format_number(statistics["median"])),
        ("Mínimo", format_number(statistics["min"])),
        ("Máximo", format_number(statistics["max"])),
        ("Desvio-padrão", format_number(statistics["std"])),
        ("CV", f"{format_number(statistics['cv'], 1)}%" if pd.notna(statistics["cv"]) else "não aplicável"),
        ("Preenchimento", f"{format_number(statistics['coverage'], 1)}%"),
    ]
    gap = 6
    total_width = PAGE_WIDTH - 2 * MARGIN
    card_width = (total_width - gap * (len(cards) - 1)) / len(cards)
    for index, (label, value) in enumerate(cards):
        x = MARGIN + index * (card_width + gap)
        report.canvas.setFillColor(white)
        report.canvas.setStrokeColor(GRID)
        report.canvas.roundRect(x, y, card_width, 44, 5, fill=1, stroke=1)
        report.canvas.setFillColor(MUTED)
        report.canvas.setFont("Helvetica", 6.8)
        report.canvas.drawString(x + 6, y + 29, label)
        report.canvas.setFillColor(INK)
        report.canvas.setFont("Helvetica-Bold", 9.5)
        report.canvas.drawString(x + 6, y + 12, value)


def create_behavior_figure(data: pd.DataFrame, column: str):
    figure, axis = plt.subplots(figsize=(10.8, 2.65))
    valid = data.dropna(subset=[column]).reset_index(drop=True)
    sequence = np.arange(1, len(valid) + 1)
    colors = depth_color_map(valid)
    if colors:
        depth_text = valid["Profundidade"].astype("string").str.strip()
        plotted = pd.Series(False, index=valid.index)
        for depth, color in colors.items():
            mask = depth_text.eq(depth).fillna(False)
            plotted |= mask
            axis.scatter(sequence[mask], valid.loc[mask, column], s=23, alpha=0.82, color=color, label=depth)
        if (~plotted).any():
            axis.scatter(
                sequence[~plotted],
                valid.loc[~plotted, column],
                s=23,
                alpha=0.72,
                color="#64748b",
                label="Sem profundidade",
            )
        axis.legend(title="Profundidade (cm)", fontsize=7, title_fontsize=7, ncol=min(4, len(colors)))
    else:
        axis.scatter(sequence, valid[column], s=23, alpha=0.82, color="#176b5b")
    axis.set_title(f"Comportamento de {parameter_label(column)} nas amostras")
    axis.set_xlabel("Sequência das amostras")
    axis.set_ylabel(parameter_label(column))
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    figure.tight_layout()
    return figure


def create_distribution_figure(data: pd.DataFrame, column: str):
    values = data[column].dropna().to_numpy()
    figure = plt.figure(figsize=(5.25, 2.8), layout="constrained")
    grid = figure.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.06)
    histogram = figure.add_subplot(grid[0])
    marginal = figure.add_subplot(grid[1], sharex=histogram)
    bins = min(20, max(5, int(np.sqrt(len(values)))))
    histogram.hist(values, bins=bins, color="#3a8878", edgecolor="white", linewidth=0.7)
    histogram.set_title(f"Distribuição de {parameter_short_label(column)}")
    histogram.set_ylabel("Frequência")
    histogram.tick_params(axis="x", labelbottom=False)
    marginal.boxplot(
        values,
        vert=False,
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#dff3ee", "edgecolor": "#176b5b"},
        medianprops={"color": "#b42318", "linewidth": 1.5},
        whiskerprops={"color": "#176b5b"},
        capprops={"color": "#176b5b"},
        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.45},
    )
    marginal.set_yticks([])
    marginal.set_xlabel(parameter_label(column))
    marginal.grid(axis="x")
    return figure


def create_depth_boxplot_figure(data: pd.DataFrame, column: str):
    figure, axis = plt.subplots(figsize=(5.25, 2.8))
    if "Profundidade" not in data.columns:
        axis.axis("off")
        axis.text(0.5, 0.5, "Profundidade não disponível", ha="center", va="center", color="#5f6b76")
        return figure

    source = data.dropna(subset=[column, "Profundidade"]).copy()
    source["_depth"] = source["Profundidade"].astype("string").str.strip()
    depths = ordered_depths(source["_depth"])
    groups = [source.loc[source["_depth"].eq(depth), column].to_numpy() for depth in depths]
    if not groups:
        axis.axis("off")
        axis.text(0.5, 0.5, "Sem dados por profundidade", ha="center", va="center", color="#5f6b76")
        return figure

    boxplot_options = {
        "patch_artist": True,
        "widths": 0.55,
        "medianprops": {"color": "#b42318", "linewidth": 1.5},
        "flierprops": {"marker": "", "markersize": 0},
    }
    try:
        boxes = axis.boxplot(groups, tick_labels=depths, **boxplot_options)
    except TypeError:  # Compatibilidade com Matplotlib anterior a 3.9.
        boxes = axis.boxplot(groups, labels=depths, **boxplot_options)
    for patch, color in zip(boxes["boxes"], DEPTH_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    random = np.random.default_rng(42)
    for position, values in enumerate(groups, start=1):
        jitter = random.normal(0, 0.045, size=len(values))
        axis.scatter(np.full(len(values), position) + jitter, values, s=14, alpha=0.62, color="#334155")
    axis.set_title(f"{parameter_short_label(column)} por profundidade")
    axis.set_xlabel("Profundidade (cm)")
    axis.set_ylabel(parameter_label(column))
    axis.set_ylim(bottom=0)
    figure.tight_layout()
    return figure


def create_pair_grid_figure(
    data: pd.DataFrame,
    pairs: list[tuple[str, str]],
    min_pairs: int,
    depth_colors: dict[str, str],
):
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 6.2))
    axes_flat = axes.ravel()
    for axis, (x_column, y_column) in zip(axes_flat, pairs):
        stats = correlation_statistics(data, x_column, y_column, min_pairs)
        pair = stats["pair"]
        if depth_colors and "Profundidade" in data.columns:
            source_columns = [x_column, y_column, "Profundidade"]
            source = data[source_columns].dropna(subset=[x_column, y_column]).copy()
            depth_text = source["Profundidade"].astype("string").str.strip()
            plotted = pd.Series(False, index=source.index)
            for depth, color in depth_colors.items():
                mask = depth_text.eq(depth).fillna(False)
                plotted |= mask
                axis.scatter(source.loc[mask, x_column], source.loc[mask, y_column], s=17, alpha=0.7, color=color)
            if (~plotted).any():
                axis.scatter(
                    source.loc[~plotted, x_column],
                    source.loc[~plotted, y_column],
                    s=17,
                    alpha=0.65,
                    color="#64748b",
                )
        else:
            axis.scatter(pair[x_column], pair[y_column], s=17, alpha=0.72, color="#176b5b")
        axis.set_xlabel(parameter_short_label(x_column))
        axis.set_ylabel(parameter_short_label(y_column))
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
        if stats["eligible"]:
            detail = (
                f"n={stats['n']} | Pearson={format_number(stats['pearson'], 3)} | "
                f"Spearman={format_number(stats['spearman'], 3)}"
            )
        else:
            detail = f"n={stats['n']} | correlação não calculada"
        axis.set_title(
            f"{parameter_short_label(y_column)} x {parameter_short_label(x_column)}\n{detail}",
            fontsize=8.5,
        )
    for axis in axes_flat[len(pairs):]:
        axis.axis("off")
    if depth_colors:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=color, label=depth, markersize=5)
            for depth, color in depth_colors.items()
        ]
        figure.legend(
            handles=handles,
            title="Profundidade (cm)",
            loc="lower center",
            ncol=min(6, len(handles)),
            fontsize=7,
            title_fontsize=7,
        )
        figure.tight_layout(rect=[0, 0.06, 1, 1])
    else:
        figure.tight_layout()
    return figure


def create_matrix_figure(matrix: pd.DataFrame, title: str, counts: bool = False):
    size = max(8.5, 0.52 * len(matrix))
    figure, axis = plt.subplots(figsize=(size, 7.0))
    numeric_values = matrix.to_numpy(dtype=float)
    if counts:
        image = axis.imshow(numeric_values, cmap="Blues", aspect="auto")
        finite_values = numeric_values[np.isfinite(numeric_values)]
        color_midpoint = (
            (finite_values.min() + finite_values.max()) / 2
            if finite_values.size
            else 0
        )
    else:
        image = axis.imshow(numeric_values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        color_midpoint = 0.55
    labels = list(matrix.columns)
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=50, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_title(title)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix.iat[row, column]
            if pd.isna(value):
                text = "-"
            elif counts:
                text = str(int(value))
            else:
                text = f"{value:.2f}"
            if pd.isna(value):
                text_color = "#5f6b76"
            elif counts:
                text_color = "white" if value >= color_midpoint else "#17212b"
            else:
                text_color = "white" if abs(value) >= color_midpoint else "#17212b"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=5.7,
                color=text_color,
            )
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    figure.tight_layout()
    return figure


def draw_cover_page(
    report: PdfReport,
    args: argparse.Namespace,
    data: pd.DataFrame,
    parameters: list[str],
    sheet_name: str,
):
    report.start_page(args.title, "Relatório completo", bookmark="capa")
    c = report.canvas
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 21)
    c.drawString(MARGIN, PAGE_HEIGHT - 112, "Relatório de análises químicas")
    c.setFont("Helvetica", 11)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, PAGE_HEIGHT - 136, f"Amostragem: {args.amostragem}")

    details = [
        ("Arquivo", args.input.name),
        ("Aba", sheet_name),
        ("Gerado em", datetime.now().strftime("%d/%m/%Y %H:%M")),
        ("Tratamento dos limites", CENSOR_POLICIES[args.censor_policy]),
        ("Amostras analisadas", str(len(data))),
        ("Parâmetros com dados", str(len(parameters))),
        ("Talhões", str(data["Talhão"].nunique()) if "Talhão" in data.columns else "não disponível"),
        ("Profundidades", str(data["Profundidade"].nunique()) if "Profundidade" in data.columns else "não disponível"),
    ]
    y = PAGE_HEIGHT - 190
    for label, value in details:
        c.setFillColor(ACCENT_LIGHT)
        c.roundRect(MARGIN, y - 6, 150, 24, 4, fill=1, stroke=0)
        c.setFillColor(ACCENT)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(MARGIN + 8, y + 2, label)
        c.setFillColor(INK)
        c.setFont("Helvetica", 9)
        c.drawString(MARGIN + 166, y + 2, str(value)[:92])
        y -= 33

    c.setFillColor(white)
    c.setStrokeColor(GRID)
    c.roundRect(PAGE_WIDTH - 290, PAGE_HEIGHT - 460, 250, 292, 8, fill=1, stroke=1)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(PAGE_WIDTH - 270, PAGE_HEIGHT - 194, "Conteúdo")
    items = [
        "Visão geral e qualidade dos dados",
        "Estatísticas de todos os parâmetros",
        "Dispersão por sequência de amostras",
        "Histogramas com boxplot marginal",
        "Boxplots por profundidade",
        "Relações entre todos os parâmetros",
        "Matriz de correlação de Pearson",
        "Matriz de pares válidos",
    ]
    y = PAGE_HEIGHT - 222
    for index, item in enumerate(items, start=1):
        c.setFillColor(ACCENT)
        c.circle(PAGE_WIDTH - 270, y + 3, 8, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(PAGE_WIDTH - 270, y, str(index))
        c.setFillColor(INK)
        c.setFont("Helvetica", 8.5)
        c.drawString(PAGE_WIDTH - 253, y, item)
        y -= 29
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(MARGIN, 47, "Mapas foram excluídos deste relatório conforme solicitado.")


def draw_overview_page(
    report: PdfReport,
    data: pd.DataFrame,
    quality: pd.DataFrame,
    parameters: list[str],
):
    report.start_page("Visão geral", "Resumo estatístico", bookmark="visao-geral")
    c = report.canvas
    card_values = [
        ("Amostras", len(data)),
        ("Talhões", data["Talhão"].nunique() if "Talhão" in data.columns else "-"),
        ("Profundidades", data["Profundidade"].nunique() if "Profundidade" in data.columns else "-"),
        ("Parâmetros", len(parameters)),
    ]
    card_width = 155
    for index, (label, value) in enumerate(card_values):
        x = MARGIN + index * (card_width + 13)
        c.setFillColor(white)
        c.setStrokeColor(GRID)
        c.roundRect(x, PAGE_HEIGHT - 130, card_width, 48, 6, fill=1, stroke=1)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7.5)
        c.drawString(x + 10, PAGE_HEIGHT - 101, label)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 15)
        c.drawString(x + 10, PAGE_HEIGHT - 120, str(value))

    headers = ["Parâmetro", "n", "Média", "Mediana", "Mínimo", "Máximo", "CV (%)", "Preench. (%)"]
    widths = [224, 42, 72, 72, 66, 66, 68, 78]
    x_positions = [MARGIN]
    for width in widths[:-1]:
        x_positions.append(x_positions[-1] + width)
    top = PAGE_HEIGHT - 158
    row_height = min(21, 355 / max(1, len(parameters)))
    c.setFillColor(ACCENT)
    c.rect(MARGIN, top - 18, sum(widths), 20, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 7)
    for x, header in zip(x_positions, headers):
        c.drawString(x + 4, top - 12, header)
    y = top - 18
    for index, column in enumerate(parameters):
        stats = parameter_statistics(data, column)
        y -= row_height
        c.setFillColor(white if index % 2 == 0 else PAPER)
        c.rect(MARGIN, y, sum(widths), row_height, fill=1, stroke=0)
        values = [
            parameter_label(column),
            str(stats["n"]),
            format_number(stats["mean"]),
            format_number(stats["median"]),
            format_number(stats["min"]),
            format_number(stats["max"]),
            format_number(stats["cv"], 1),
            format_number(stats["coverage"], 1),
        ]
        c.setFillColor(INK)
        c.setFont("Helvetica", 6.8)
        for x, value, width in zip(x_positions, values, widths):
            c.drawString(x + 4, y + max(5, (row_height - 7) / 2), str(value)[:36])

    issues = quality[
        quality[["Inválidos", "Limite inf.", "Limite sup.", "Ausentes marcados"]]
        .sum(axis=1)
        .gt(0)
    ]
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7.5)
    issue_text = (
        f"Diagnóstico: {len(issues)} parâmetro(s) possuem células inválidas, censuradas "
        "ou marcadores explícitos de ausência."
    )
    c.drawString(MARGIN, 42, issue_text)


def draw_quality_page(report: PdfReport, quality: pd.DataFrame):
    report.start_page("Qualidade dos dados", "Conversão numérica", bookmark="qualidade")
    c = report.canvas
    c.setFillColor(INK)
    c.setFont("Helvetica", 9)
    c.drawString(
        MARGIN,
        PAGE_HEIGHT - 86,
        "Valores inválidos não entram nos cálculos; limites seguem a política registrada na capa.",
    )
    headers = list(quality.columns)
    widths = [210, 80, 80, 100, 100, 130]
    x_positions = [MARGIN]
    for width in widths[:-1]:
        x_positions.append(x_positions[-1] + width)
    top = PAGE_HEIGHT - 114
    c.setFillColor(ACCENT)
    c.rect(MARGIN, top - 20, sum(widths), 22, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 7.5)
    for x, header in zip(x_positions, headers):
        c.drawString(x + 5, top - 14, header)
    y = top - 20
    for index, row in quality.iterrows():
        y -= 22
        c.setFillColor(white if index % 2 == 0 else PAPER)
        c.rect(MARGIN, y, sum(widths), 22, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica", 7.2)
        for x, header in zip(x_positions, headers):
            c.drawString(x + 5, y + 8, str(row[header])[:34])


def draw_parameter_page(report: PdfReport, data: pd.DataFrame, column: str, index: int, total: int):
    report.start_page(
        parameter_label(column),
        f"Parâmetro {index} de {total}",
        bookmark=f"parametro-{index}",
    )
    stats = parameter_statistics(data, column)
    draw_metric_cards(report, stats, PAGE_HEIGHT - 119)
    behavior = create_behavior_figure(data, column)
    report.draw_figure(behavior, MARGIN, 274, PAGE_WIDTH - 2 * MARGIN, 192)
    distribution = create_distribution_figure(data, column)
    report.draw_figure(distribution, MARGIN, 41, 372, 221)
    depth_boxplot = create_depth_boxplot_figure(data, column)
    report.draw_figure(depth_boxplot, PAGE_WIDTH - MARGIN - 372, 41, 372, 221)


def draw_relationship_pages(
    report: PdfReport,
    data: pd.DataFrame,
    parameters: list[str],
    min_pairs: int,
):
    pairs = list(combinations(parameters, 2))
    if not pairs:
        return 0
    depth_colors = depth_color_map(data)
    chunks = [pairs[index:index + 4] for index in range(0, len(pairs), 4)]
    for page_index, pair_chunk in enumerate(chunks, start=1):
        report.start_page(
            "Relações entre parâmetros",
            f"Página {page_index} de {len(chunks)}",
            bookmark=f"relacoes-{page_index}",
        )
        figure = create_pair_grid_figure(data, pair_chunk, min_pairs, depth_colors)
        report.draw_figure(
            figure,
            MARGIN,
            38,
            PAGE_WIDTH - 2 * MARGIN,
            PAGE_HEIGHT - 105,
        )
    return len(pairs)


def draw_correlation_pages(
    report: PdfReport,
    data: pd.DataFrame,
    parameters: list[str],
    min_pairs: int,
):
    source = data[parameters]
    correlation = source.corr(method="pearson", min_periods=min_pairs)
    labels = [parameter_short_label(column) for column in parameters]
    correlation.index = labels
    correlation.columns = labels
    report.start_page(
        "Matriz de correlação de Pearson",
        f"Mínimo de {min_pairs} pares",
        bookmark="matriz-correlacao",
    )
    figure = create_matrix_figure(correlation, "Correlação de Pearson")
    report.draw_figure(figure, MARGIN, 38, PAGE_WIDTH - 2 * MARGIN, PAGE_HEIGHT - 105)

    presence = source.notna().astype(int)
    counts = presence.T.dot(presence)
    counts.index = labels
    counts.columns = labels
    report.start_page(
        "Matriz de pares válidos",
        "Quantidade usada em cada correlação",
        bookmark="matriz-pares",
    )
    figure = create_matrix_figure(counts, "Quantidade de pares válidos", counts=True)
    report.draw_figure(figure, MARGIN, 38, PAGE_WIDTH - 2 * MARGIN, PAGE_HEIGHT - 105)


def generate_report(
    args: argparse.Namespace,
    data: pd.DataFrame,
    quality: pd.DataFrame,
    sheet_name: str,
) -> tuple[int, int]:
    parameters = available_parameters(data)
    if not parameters:
        raise ReportValidationError("Nenhum parâmetro possui valores numéricos.")
    apply_plot_style()
    report = PdfReport(args.output, args.title)
    draw_cover_page(report, args, data, parameters, sheet_name)
    draw_overview_page(report, data, quality, parameters)
    draw_quality_page(report, quality)
    for index, column in enumerate(parameters, start=1):
        draw_parameter_page(report, data, column, index, len(parameters))
    pair_count = 0
    if not args.skip_pair_plots:
        pair_count = draw_relationship_pages(report, data, parameters, args.min_pairs)
    draw_correlation_pages(report, data, parameters, args.min_pairs)
    report.save()
    return report.page_number, pair_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gera um PDF multipágina com todas as análises químicas e todos os "
            "parâmetros disponíveis, sem mapas."
        )
    )
    parser.add_argument("input", type=Path, help="Arquivo Excel .xlsx ou .xlsm")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/pdf/relatorio_analise_quimica.pdf"),
        help="PDF de saída (padrão: output/pdf/relatorio_analise_quimica.pdf)",
    )
    parser.add_argument("--sheet", help="Nome da aba; por padrão usa a primeira")
    parser.add_argument("--amostragem", default="Carbono", help="Valor da coluna Amostragem")
    parser.add_argument("--talhao", action="append", help="Talhão a incluir; pode ser repetido")
    parser.add_argument(
        "--profundidade", action="append", help="Profundidade a incluir; pode ser repetida"
    )
    parser.add_argument(
        "--censor-policy",
        choices=sorted(CENSOR_POLICIES),
        default="missing",
        help="Tratamento de valores como <0,01 (missing, half ou limit)",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=5,
        help="Mínimo de pares para calcular correlações (padrão: 5)",
    )
    parser.add_argument(
        "--skip-pair-plots",
        action="store_true",
        help="Não gera as páginas com todas as combinações de parâmetros",
    )
    parser.add_argument(
        "--title",
        default="Análise laboratorial - Carbono",
        help="Título exibido no relatório",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.min_pairs < 3:
        parser.error("--min-pairs deve ser pelo menos 3")
    try:
        data, quality, sheet_name = load_report(args)
        page_count, pair_count = generate_report(args, data, quality, sheet_name)
    except (ReportValidationError, ImportError) as exc:
        parser.exit(2, f"Erro: {exc}\n")
    print(f"PDF criado: {args.output.resolve()}")
    print(f"Páginas: {page_count} | Relações entre parâmetros: {pair_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
