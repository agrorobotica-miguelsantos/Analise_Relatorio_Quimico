import io
import logging
import math
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from pyproj import Transformer
except ImportError:  # Coordenadas geográficas continuam disponíveis sem pyproj.
    Transformer = None


# Precisa ser o primeiro comando Streamlit, antes inclusive dos decoradores de cache.
st.set_page_config(
    page_title="Análise Laboratorial - Carbono", layout="wide"
)


LOGGER = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
COORD_COLUMNS = ["Coord-X", "Coord-Y"]
GROUP_COLUMNS = ["Talhão", "Profundidade"]
INTERNAL_FILTER_COLUMN = "Amostragem"
DEPTH_ORDER = ["00-20", "20-40", "40-60"]
MISSING_OPTION = "(Sem informação)"
MIN_CORRELATION_PAIRS = 5

PARAMETER_META = {
    "S_(mg/dm3)": {"label": "Enxofre (S)", "unit": "mg/dm³", "ratio_scale": True},
    "P_resina_(mg/dm3)": {"label": "Fósforo — resina", "unit": "mg/dm³", "ratio_scale": True},
    "K_(mg/dm3)": {"label": "Potássio (K)", "unit": "mg/dm³", "ratio_scale": True},
    "P_(mg/dm3)": {"label": "Fósforo (P)", "unit": "mg/dm³", "ratio_scale": True},
    "Al_(cmolc/dm3)": {"label": "Alumínio (Al)", "unit": "cmolc/dm³", "ratio_scale": True},
    "Ca_(cmolc/dm3)": {"label": "Cálcio (Ca)", "unit": "cmolc/dm³", "ratio_scale": True},
    "K_(cmolc/dm3)": {"label": "Potássio (K)", "unit": "cmolc/dm³", "ratio_scale": True},
    "Mg_(cmolc/dm3)": {"label": "Magnésio (Mg)", "unit": "cmolc/dm³", "ratio_scale": True},
    "B_(mg/dm3)": {"label": "Boro (B)", "unit": "mg/dm³", "ratio_scale": True},
    "Cu_(mg/dm3)": {"label": "Cobre (Cu)", "unit": "mg/dm³", "ratio_scale": True},
    "Fe_(mg/dm3)": {"label": "Ferro (Fe)", "unit": "mg/dm³", "ratio_scale": True},
    "Mn_(mg/dm3)": {"label": "Manganês (Mn)", "unit": "mg/dm³", "ratio_scale": True},
    "Zn_(mg/dm3)": {"label": "Zinco (Zn)", "unit": "mg/dm³", "ratio_scale": True},
    "MO_(g/dm3)": {"label": "Matéria orgânica (MO)", "unit": "g/dm³", "ratio_scale": True},
    "pH CaCl2": {"label": "pH em CaCl₂", "unit": "", "ratio_scale": False},
    "pH SMP": {"label": "pH SMP", "unit": "", "ratio_scale": False},
    "H+Al_(cmolc/dm3)": {"label": "Acidez potencial (H+Al)", "unit": "cmolc/dm³", "ratio_scale": True},
}
PARAM_COLUMNS = list(PARAMETER_META)

ESRI_WORLD_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

MISSING_MARKERS = {
    "", "-", "--", "na", "n/a", "nd", "n/d", "nan", "none",
    "não detectado", "nao detectado", "não determinado", "nao determinado",
}

CENSOR_POLICIES = {
    "missing": "Considerar como ausente",
    "half": "Usar metade do limite inferior",
    "limit": "Usar o limite informado",
}


class ReportValidationError(ValueError):
    """Erro esperado na estrutura ou no conteúdo do relatório."""


class CoordinateTransformError(ValueError):
    """Erro esperado na validação ou transformação de coordenadas."""


def parameter_label(column: str) -> str:
    meta = PARAMETER_META.get(column)
    if not meta:
        return column
    return f"{meta['label']} ({meta['unit']})" if meta["unit"] else meta["label"]


def normalize_header(value) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return re.sub(r"\s+", " ", text)


def column_key(value) -> str:
    text = unicodedata.normalize("NFKD", normalize_header(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", "", text).casefold()


def canonical_column_map() -> dict[str, str]:
    expected = [INTERNAL_FILTER_COLUMN] + GROUP_COLUMNS + COORD_COLUMNS + PARAM_COLUMNS
    aliases = {column_key(column): column for column in expected}
    aliases.update(
        {
            column_key("Talhao"): "Talhão",
            column_key("Parcela"): "Talhão",
            column_key("Coord X"): "Coord-X",
            column_key("Coord_X"): "Coord-X",
            column_key("Longitude"): "Coord-X",
            column_key("Coord Y"): "Coord-Y",
            column_key("Coord_Y"): "Coord-Y",
            column_key("Latitude"): "Coord-Y",
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
            "Existem colunas duplicadas ou equivalentes: " + ", ".join(duplicates)
        )

    result = df.copy()
    result.columns = renamed
    return result


@st.cache_data(show_spinner=False)
def read_sheet_names(file_bytes: bytes) -> list[str]:
    with pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl") as workbook:
        return workbook.sheet_names


@st.cache_data(show_spinner=False)
def read_report(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, engine="openpyxl")


def carbon_only(df: pd.DataFrame) -> pd.DataFrame:
    if INTERNAL_FILTER_COLUMN not in df.columns:
        raise ReportValidationError(
            "A coluna 'Amostragem' não foi encontrada. Verifique a aba e o cabeçalho."
        )
    mask = (
        df[INTERNAL_FILTER_COLUMN]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("carbono")
        .fillna(False)
    )
    carbon = df.loc[mask].copy()
    if carbon.empty:
        raise ReportValidationError(
            "A aba selecionada não possui linhas com Amostragem = Carbono."
        )
    return carbon


def validate_report_structure(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    normalized = normalize_columns(df)
    carbon = carbon_only(normalized)
    existing_params = [column for column in PARAM_COLUMNS if column in carbon.columns]
    if not existing_params:
        raise ReportValidationError(
            "Nenhuma das colunas de parâmetros químicos esperadas foi encontrada."
        )
    optional_missing = [
        column for column in GROUP_COLUMNS + COORD_COLUMNS if column not in carbon.columns
    ]
    keep = [
        column
        for column in GROUP_COLUMNS + COORD_COLUMNS + PARAM_COLUMNS
        if column in carbon.columns
    ]
    return carbon[keep].copy(), optional_missing


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
        numeric = float(value)
        return (numeric, "numeric") if math.isfinite(numeric) else (np.nan, "invalid")

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
        if operator in {"<", "≤"}:
            if censor_policy == "missing":
                return np.nan, "censored_low"
            if censor_policy == "half":
                return limit / 2, "censored_low"
            return limit, "censored_low"
        if censor_policy == "missing":
            return np.nan, "censored_high"
        return limit, "censored_high"

    try:
        return _parse_locale_literal(text), "parsed_text"
    except ValueError:
        return np.nan, "invalid"


def normalize_numeric_series(
    series: pd.Series, censor_policy: str
) -> tuple[pd.Series, dict[str, int]]:
    values = []
    status_counts: dict[str, int] = {}
    for value in series:
        numeric, status = parse_numeric_value(value, censor_policy)
        values.append(numeric)
        status_counts[status] = status_counts.get(status, 0) + 1
    return pd.Series(values, index=series.index, dtype="float64"), status_counts


def normalize_numeric_columns(
    df: pd.DataFrame, censor_policy: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()
    quality_rows = []
    numeric_columns = [
        column for column in COORD_COLUMNS + PARAM_COLUMNS if column in result.columns
    ]
    for column in numeric_columns:
        result[column], counts = normalize_numeric_series(result[column], censor_policy)
        quality_rows.append(
            {
                "Coluna": parameter_label(column),
                "Válidos": int(result[column].notna().sum()),
                "Inválidos": counts.get("invalid", 0),
                "Limite inferior": counts.get("censored_low", 0),
                "Limite superior": counts.get("censored_high", 0),
                "Marcadores ausentes": counts.get("missing_marker", 0),
            }
        )
    return result, pd.DataFrame(quality_rows)


def sorted_text_values(series: pd.Series) -> tuple[list[str], pd.Series]:
    as_text = series.astype("string").str.strip()
    values = sorted(
        {value for value in as_text.dropna().tolist() if value},
        key=lambda value: value.casefold(),
    )
    return values, as_text


def apply_categorical_filter(
    data: pd.DataFrame,
    column: str,
    label: str,
    preferred_order: list[str] | None = None,
) -> pd.DataFrame:
    if column not in data.columns:
        return data
    values, as_text = sorted_text_values(data[column])
    if preferred_order:
        ordered = [value for value in preferred_order if value in values]
        values = ordered + [value for value in values if value not in ordered]
    missing_mask = as_text.isna() | as_text.eq("")
    options = values + ([MISSING_OPTION] if missing_mask.any() else [])
    selected = st.sidebar.multiselect(label, options, default=options)
    include_missing = MISSING_OPTION in selected
    selected_values = [value for value in selected if value != MISSING_OPTION]
    mask = as_text.isin(selected_values)
    if include_missing:
        mask |= missing_mask
    return data.loc[mask].copy()


def existing_parameter_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in PARAM_COLUMNS if column in df.columns]


def parameters_with_data(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in existing_parameter_columns(df)
        if df[column].notna().any()
    ]


def format_number(value, decimals: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return (
        f"{value:,.{decimals}f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def correlation_stats(
    data: pd.DataFrame, x_col: str, y_col: str, min_pairs: int = MIN_CORRELATION_PAIRS
) -> dict:
    pair = data[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    stats = {
        "pair": pair,
        "n": len(pair),
        "pearson": np.nan,
        "spearman": np.nan,
        "eligible": False,
        "reason": "",
    }
    if len(pair) < min_pairs:
        stats["reason"] = f"São necessários pelo menos {min_pairs} pares válidos."
        return stats
    if pair[x_col].nunique() < 2 or pair[y_col].nunique() < 2:
        stats["reason"] = "Uma das variáveis é constante; a correlação não é definida."
        return stats

    stats["eligible"] = True
    stats["pearson"] = pair[x_col].corr(pair[y_col])
    stats["spearman"] = pair[x_col].rank(method="average").corr(
        pair[y_col].rank(method="average")
    )
    return stats


def prepare_coordinates(
    data: pd.DataFrame,
    mode: str,
    swap_xy: bool,
    utm_zone: int,
    hemisphere: str,
) -> tuple[pd.DataFrame, int]:
    mapped = data.dropna(subset=COORD_COLUMNS).copy()
    original_count = len(mapped)
    x = mapped["Coord-X"].to_numpy(dtype=float)
    y = mapped["Coord-Y"].to_numpy(dtype=float)
    if swap_xy:
        x, y = y, x

    if mode == "Geográficas (WGS84)":
        valid = (
            np.isfinite(x) & np.isfinite(y)
            & (x >= -180) & (x <= 180) & (y >= -90) & (y <= 90)
        )
        mapped = mapped.loc[valid].copy()
        mapped["_Longitude"] = x[valid]
        mapped["_Latitude"] = y[valid]
    else:
        if Transformer is None:
            raise CoordinateTransformError(
                "Para converter coordenadas UTM, instale a dependência 'pyproj'."
            )
        valid = (
            np.isfinite(x) & np.isfinite(y)
            & (x >= 100_000) & (x <= 1_000_000)
            & (y >= 0) & (y <= 10_000_000)
        )
        mapped = mapped.loc[valid].copy()
        epsg = (32700 if hemisphere == "Sul" else 32600) + int(utm_zone)
        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        longitude, latitude = transformer.transform(x[valid], y[valid])
        geo_valid = (
            np.isfinite(longitude) & np.isfinite(latitude)
            & (longitude >= -180) & (longitude <= 180)
            & (latitude >= -90) & (latitude <= 90)
        )
        mapped = mapped.loc[geo_valid].copy()
        mapped["_Longitude"] = np.asarray(longitude)[geo_valid]
        mapped["_Latitude"] = np.asarray(latitude)[geo_valid]
    return mapped, original_count - len(mapped)


def estimate_map_zoom(data: pd.DataFrame) -> int:
    if len(data) <= 1:
        return 15
    span = max(
        data["_Longitude"].max() - data["_Longitude"].min(),
        data["_Latitude"].max() - data["_Latitude"].min(),
    )
    for limit, zoom in [
        (0.005, 16), (0.01, 15), (0.03, 14), (0.08, 13), (0.2, 11),
        (0.5, 9), (1, 8), (5, 6), (20, 4),
    ]:
        if span <= limit:
            return zoom
    return 2


def add_satellite_layer(fig, data: pd.DataFrame):
    fig.update_layout(
        map={
            "style": "white-bg",
            "center": {
                "lon": float(data["_Longitude"].mean()),
                "lat": float(data["_Latitude"].mean()),
            },
            "zoom": estimate_map_zoom(data),
            "layers": [
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "sourceattribution": "Esri World Imagery",
                    "source": [ESRI_WORLD_IMAGERY],
                }
            ],
        },
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


def render_theme():
    st.markdown(
        """
        <style>
            .stApp { background-color: #f7f9fb; color: #17212b; }
            [data-testid="stSidebar"] { background-color: #ffffff; }
            [data-testid="stMetric"] {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                padding: 12px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_excel_error(exc: Exception):
    if isinstance(exc, ImportError):
        st.error("Não foi possível abrir o Excel: instale a dependência 'openpyxl'.")
    elif isinstance(exc, zipfile.BadZipFile):
        st.error("O arquivo não é um Excel válido ou está corrompido.")
    elif isinstance(exc, (ValueError, OSError)):
        st.error(f"Não foi possível abrir o Excel: {exc}")
    else:
        LOGGER.exception("Erro inesperado ao ler o relatório")
        st.error("Ocorreu um erro inesperado ao ler o relatório.")


def render_quality_report(quality: pd.DataFrame, optional_missing: list[str]):
    issue_columns = [
        "Inválidos", "Limite inferior", "Limite superior", "Marcadores ausentes"
    ]
    problems = quality[quality[issue_columns].sum(axis=1).gt(0)]
    if optional_missing:
        st.warning("Recursos limitados pela ausência de: " + ", ".join(optional_missing))
    if not problems.empty:
        st.warning(
            "Algumas células exigiram tratamento ou não puderam ser convertidas. "
            "Consulte o diagnóstico abaixo."
        )
    with st.expander("Diagnóstico de qualidade dos dados"):
        st.dataframe(quality, width="stretch", hide_index=True)
        st.caption(
            "Valores inválidos nunca entram nos cálculos. Valores com limite de detecção "
            "seguem a opção escolhida na barra lateral."
        )


def render_overview(
    filtered: pd.DataFrame, available_params: list[str], all_existing_params: list[str]
):
    st.subheader("1. Visão geral")
    n_talhoes = filtered["Talhão"].nunique() if "Talhão" in filtered.columns else np.nan
    n_depths = filtered["Profundidade"].nunique() if "Profundidade" in filtered.columns else np.nan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Amostras", str(len(filtered)))
    c2.metric("Talhões", "—" if pd.isna(n_talhoes) else str(n_talhoes))
    c3.metric("Profundidades", "—" if pd.isna(n_depths) else str(n_depths))
    c4.metric("Parâmetros com dados", f"{len(available_params)}/{len(all_existing_params)}")
    empty_params = [column for column in all_existing_params if column not in available_params]
    if empty_params:
        st.info(
            "Parâmetros sem resultados no recorte atual: "
            + ", ".join(parameter_label(column) for column in empty_params)
        )


def render_parameter_analysis(filtered: pd.DataFrame, available_params: list[str]) -> str:
    st.subheader("2. Análise de um parâmetro")
    preferred = ["MO_(g/dm3)", "pH CaCl2", "Ca_(cmolc/dm3)", "P_(mg/dm3)"]
    default = next((column for column in preferred if column in available_params), available_params[0])
    indicator = st.selectbox(
        "Parâmetro para análise",
        available_params,
        index=available_params.index(default),
        format_func=parameter_label,
        key="main_indicator",
    )
    valid = filtered.dropna(subset=[indicator]).copy()
    values = valid[indicator]
    mean = values.mean()
    median = values.median()
    std = values.std(ddof=1)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Resultados", str(values.count()))
    m2.metric("Média", format_number(mean))
    m3.metric("Mediana", format_number(median))
    m4.metric("Mínimo", format_number(values.min()))
    m5.metric("Máximo", format_number(values.max()))

    caption_parts = [f"Desvio-padrão: **{format_number(std)}**"]
    if PARAMETER_META[indicator]["ratio_scale"]:
        cv = std / mean * 100 if pd.notna(mean) and mean != 0 else np.nan
        caption_parts.append(f"CV: **{format_number(cv, 1)}%**")
    else:
        caption_parts.append("CV: **não aplicável à escala de pH**")
    caption_parts.append(f"Preenchimento: **{values.count() / len(filtered) * 100:.1f}%**")
    st.caption(" | ".join(caption_parts))

    behavior_df = valid.reset_index(drop=True).copy()
    behavior_df["_Amostra"] = np.arange(1, len(behavior_df) + 1)
    color_by_depth = (
        "Profundidade" in behavior_df.columns
        and behavior_df["Profundidade"].nunique() > 1
    )
    hover_columns = [
        column
        for column in GROUP_COLUMNS + COORD_COLUMNS
        if column in behavior_df.columns
    ]
    fig_behavior = px.scatter(
        behavior_df,
        x="_Amostra",
        y=indicator,
        color="Profundidade" if color_by_depth else None,
        hover_data=hover_columns,
        labels={
            "_Amostra": "Sequência das amostras",
            indicator: parameter_label(indicator),
            "Profundidade": "Profundidade (cm)",
        },
        category_orders={"Profundidade": DEPTH_ORDER},
        title=f"Comportamento de {parameter_label(indicator)} nas amostras",
        template="plotly_white",
    )
    fig_behavior.update_traces(marker={"size": 10, "opacity": 0.85})
    fig_behavior.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(rangemode="tozero"),
        yaxis=dict(rangemode="tozero"),
    )
    st.plotly_chart(fig_behavior, width="stretch")
    st.caption(
        "A sequência no eixo X segue a ordem das linhas válidas no relatório; "
        "ela não representa uma escala temporal."
    )

    left, right = st.columns(2)
    labels = {indicator: parameter_label(indicator), "Profundidade": "Profundidade (cm)"}
    with left:
        fig_hist = px.histogram(
            valid,
            x=indicator,
            nbins=min(20, max(5, int(np.sqrt(len(valid))))),
            marginal="box",
            labels=labels,
            title=f"Distribuição de {parameter_label(indicator)}",
            template="plotly_white",
        )
        fig_hist.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_hist, width="stretch")
    with right:
        if "Profundidade" in valid.columns:
            fig_box = px.box(
                valid,
                x="Profundidade",
                y=indicator,
                points="all",
                labels=labels,
                category_orders={"Profundidade": DEPTH_ORDER},
                title=f"{parameter_label(indicator)} por profundidade",
                template="plotly_white",
            )
            fig_box.update_layout(
                margin=dict(l=10, r=10, t=50, b=10),
                yaxis=dict(rangemode="tozero"),
            )
            st.plotly_chart(fig_box, width="stretch")
        else:
            st.info("A coluna Profundidade não está disponível para o gráfico comparativo.")

    return indicator


def render_map(filtered: pd.DataFrame, indicator: str):
    st.subheader("4. Distribuição espacial")
    if not set(COORD_COLUMNS + [indicator]).issubset(filtered.columns):
        st.info("O arquivo não possui todas as colunas necessárias para o mapa.")
        return

    st.sidebar.subheader("Coordenadas")
    coordinate_mode = st.sidebar.selectbox(
        "Sistema de coordenadas",
        ["Geográficas (WGS84)", "UTM"],
        help="O mapa precisa de longitude e latitude. Se o relatório usa UTM, informe a zona.",
    )
    swap_xy = st.sidebar.checkbox("Trocar Coord-X e Coord-Y", value=False)
    utm_zone = 22
    hemisphere = "Sul"
    if coordinate_mode == "UTM":
        utm_zone = int(st.sidebar.number_input("Zona UTM", 1, 60, 22, 1))
        hemisphere = st.sidebar.selectbox("Hemisfério", ["Sul", "Norte"])

    source = filtered.dropna(subset=[indicator]).copy()
    try:
        map_df, invalid_count = prepare_coordinates(
            source, coordinate_mode, swap_xy, utm_zone, hemisphere
        )
    except CoordinateTransformError as exc:
        st.error(str(exc))
        return
    if invalid_count:
        st.warning(f"{invalid_count} ponto(s) foram ignorados por coordenadas fora do intervalo esperado.")
    if map_df.empty:
        st.info("Não há coordenadas válidas e resultados simultaneamente disponíveis.")
        return

    hover_cols = [column for column in GROUP_COLUMNS if column in map_df.columns]
    hover_data = {column: True for column in hover_cols}
    hover_data.update({"_Latitude": ":.6f", "_Longitude": ":.6f"})
    fig_map = px.scatter_map(
        map_df,
        lat="_Latitude",
        lon="_Longitude",
        color=indicator,
        color_continuous_scale="Plasma",
        hover_data=hover_data,
        labels={
            indicator: parameter_label(indicator),
            "_Latitude": "Latitude",
            "_Longitude": "Longitude",
        },
        height=540,
        title=f"Distribuição espacial de {parameter_label(indicator)}",
    )
    fig_map.update_traces(marker={"size": 8, "opacity": 0.95})
    add_satellite_layer(fig_map, map_df)
    st.plotly_chart(fig_map, width="stretch")
    st.caption(
        "Fundo: Esri World Imagery. O navegador solicita imagens da área exibida a um "
        "serviço externo; avalie esse uso caso a localização seja sensível. A imagem não é em tempo real."
    )


def remember_relation_y():
    """Preserva o eixo Y quando uma alteração no eixo X recria suas opções."""
    selected = st.session_state.get("relation_y_widget")
    if selected is not None:
        st.session_state["relation_y_saved"] = selected


def render_relationships(filtered: pd.DataFrame, available_params: list[str]):
    st.subheader("3. Relações entre parâmetros")
    st.caption(
        "Os cálculos usam apenas linhas em que X e Y possuem resultado simultaneamente. "
        "Correlação não implica causalidade."
    )
    if len(available_params) < 2:
        st.info("São necessários pelo menos dois parâmetros preenchidos.")
        return

    min_pairs = int(
        st.sidebar.number_input(
            "Mínimo de pares para correlação",
            min_value=3,
            max_value=100,
            value=MIN_CORRELATION_PAIRS,
            step=1,
        )
    )
    rel_c1, rel_c2 = st.columns(2)
    default_x = "pH CaCl2" if "pH CaCl2" in available_params else available_params[0]
    with rel_c1:
        x_param = st.selectbox(
            "Eixo X",
            available_params,
            index=available_params.index(default_x),
            format_func=parameter_label,
            key="relation_x",
        )
    y_options = [column for column in available_params if column != x_param]
    preferred_y = "Al_(cmolc/dm3)" if "Al_(cmolc/dm3)" in y_options else y_options[0]
    saved_y = st.session_state.get("relation_y_saved")
    if saved_y not in y_options:
        saved_y = preferred_y
    st.session_state["relation_y_widget"] = saved_y
    with rel_c2:
        y_param = st.selectbox(
            "Eixo Y",
            y_options,
            format_func=parameter_label,
            key="relation_y_widget",
            on_change=remember_relation_y,
        )
    st.session_state["relation_y_saved"] = y_param

    stats = correlation_stats(filtered, x_param, y_param, min_pairs)
    pair = stats["pair"]
    if pair.empty:
        st.info("Não há pares de dados para essa combinação.")
        return
    pairs_col, pearson_col, spearman_col = st.columns(3)
    pairs_col.metric("Pares válidos (n)", str(stats["n"]))
    pearson_col.metric("Pearson (r)", format_number(stats["pearson"], 3))
    spearman_col.metric("Spearman (ρ)", format_number(stats["spearman"], 3))
    if not stats["eligible"]:
        st.warning(stats["reason"])

    plot_cols = list(
        dict.fromkeys([x_param, y_param] + [c for c in GROUP_COLUMNS if c in filtered.columns])
    )
    rel_df = filtered[plot_cols].dropna(subset=[x_param, y_param]).copy()
    color_by_depth = "Profundidade" in rel_df.columns and rel_df["Profundidade"].nunique() > 1
    fig_rel = px.scatter(
        rel_df,
        x=x_param,
        y=y_param,
        color="Profundidade" if color_by_depth else None,
        hover_data=[column for column in GROUP_COLUMNS if column in rel_df.columns],
        labels={x_param: parameter_label(x_param), y_param: parameter_label(y_param)},
        category_orders={"Profundidade": DEPTH_ORDER},
        title=f"{parameter_label(y_param)} × {parameter_label(x_param)}",
        template="plotly_white",
    )
    fig_rel.update_traces(marker=dict(size=9), selector=dict(mode="markers"))
    fig_rel.update_layout(
        height=540,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(rangemode="tozero"),
        yaxis=dict(rangemode="tozero"),
    )
    st.plotly_chart(fig_rel, width="stretch")

    with st.expander("Ver matriz de correlação entre todos os parâmetros"):
        corr_cols = [
            column for column in available_params if filtered[column].notna().sum() >= min_pairs
        ]
        if len(corr_cols) < 2:
            st.info("Não há parâmetros suficientes para formar a matriz.")
            return
        corr_source = filtered[corr_cols]
        corr_df = corr_source.corr(method="pearson", min_periods=min_pairs)
        labels = [parameter_label(column) for column in corr_cols]
        corr_df.index = labels
        corr_df.columns = labels
        fig_corr = px.imshow(
            corr_df,
            text_auto=".2f",
            aspect="auto",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            title="Matriz de correlação de Pearson",
            template="plotly_white",
        )
        fig_corr.update_layout(
            height=max(520, 36 * len(corr_cols)),
            margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(fig_corr, width="stretch")
        presence = corr_source.notna().astype(int)
        count_df = presence.T.dot(presence)
        count_df.index = labels
        count_df.columns = labels
        st.caption("Quantidade de pares utilizada em cada célula:")
        st.dataframe(count_df, width="stretch")


def main():
    render_theme()
    st.title("Análise do relatório químico")
    st.caption(
        "O aplicativo considera apenas as linhas em que **Amostragem = Carbono** e "
        "valida os dados antes de executar os cálculos."
    )
    uploaded_file = st.file_uploader(
        "Carregue o relatório do laboratório",
        type=["xlsx", "xlsm"],
        help=f"Excel com até {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
    )
    if uploaded_file is None:
        st.info("Carregue um arquivo Excel para iniciar a análise.")
        st.stop()
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        st.error(f"O arquivo excede o limite de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        st.stop()

    file_bytes = uploaded_file.getvalue()
    try:
        sheet_names = read_sheet_names(file_bytes)
        if not sheet_names:
            raise ReportValidationError("O arquivo não possui planilhas visíveis.")
        sheet_name = st.selectbox("Aba do relatório", sheet_names)
        raw = read_report(file_bytes, sheet_name)
        report, optional_missing = validate_report_structure(raw)
    except ReportValidationError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        show_excel_error(exc)
        st.stop()

    st.sidebar.header("Tratamento e filtros")
    censor_policy = st.sidebar.selectbox(
        "Valores abaixo/acima do limite de detecção",
        list(CENSOR_POLICIES),
        format_func=CENSOR_POLICIES.get,
        help="A escolha é aplicada a valores textuais como <0,01 ou >100.",
    )
    report, quality = normalize_numeric_columns(report, censor_policy)
    filtered = apply_categorical_filter(report, "Talhão", "Talhão")
    filtered = apply_categorical_filter(
        filtered, "Profundidade", "Profundidade (cm)", DEPTH_ORDER
    )
    if filtered.empty:
        st.warning("Nenhuma linha permaneceu após os filtros selecionados.")
        st.stop()

    available_params = parameters_with_data(filtered)
    all_existing_params = existing_parameter_columns(filtered)
    render_quality_report(quality, optional_missing)
    render_overview(filtered, available_params, all_existing_params)
    if not available_params:
        st.warning("Não foram encontrados resultados numéricos para os filtros atuais.")
        st.stop()

    indicator = render_parameter_analysis(filtered, available_params)
    render_relationships(filtered, available_params)
    render_map(filtered, indicator)


if __name__ == "__main__":
    main()
