"""
smart_plot.py
─────────────
Auto-selects and renders the best matplotlib chart for a SQL result DataFrame.

Public API
──────────
    smart_plot(df, title="Query Result")   → renders chart via plt.show()

Internal modules (all prefixed with _)
───────────────────────────────────────
    Column classification  : _classify_columns, _is_id_like, _rescue_measures
    Label utilities        : _pick_label_col, _clean_labels, _draw_legend_table,
                             _truncate
    Chart helpers          : _scales_differ
    Chart renderers        : _plot_rank_bar, _plot_area, _plot_dual_line,
                             _plot_line, _plot_single_bar, _plot_dual_subplot,
                             _plot_grouped_bar, _plot_heatmap, _plot_table
"""

import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import uuid

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_ID_NAME_HINTS = ['_id', 'id_', '_code', 'code_', '_index', '_seq', '_no', 'no_']

_MEASURE_HINTS = [
    'total', 'sum', 'count', 'avg', 'average', 'amount',
    'consumption', 'qty', 'quantity', 'revenue', 'cost',
    'sales', 'profit', 'score', 'rate', 'value',
]

_LABEL_HINTS = [
    'supplier', 'name', 'item', 'product',
    'employee', 'store', 'region', 'vendor', 'desc',
]

_WIN_RANK_HINTS    = ["rank", "row_number", "dense_rank"]
_WIN_RUNNING_HINTS = ["running", "cumsum", "cumulative"]
_WIN_DIFF_HINTS    = ["lag", "lead", "diff", "pct_change"]
_WIN_ALL_HINTS     = _WIN_RANK_HINTS + _WIN_RUNNING_HINTS + _WIN_DIFF_HINTS

_BAR_COLORS = [
    "#1E1B4B",  # Deep Indigo (darker base)
    "#2C1A67",  # Your Indigo
    "#312E81",  # Indigo-800
    "#3730A3",  # Indigo-700
    "#4338CA",  # Indigo-600
    "#6366F1",  # Indigo-500 (highlight)
]


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _is_id_like(series: pd.Series, col_name: str = "") -> bool:
    """True if a numeric column is an identifier/year/rank, not a measure."""
    col_lower = col_name.lower()

    if any(col_lower.startswith(h) or col_lower.endswith(h) for h in _ID_NAME_HINTS):
        return True

    if pd.api.types.is_integer_dtype(series):
        if series.between(1900, 2100).all():
            return True

    if pd.api.types.is_integer_dtype(series) and len(series) >= 3:
        sorted_vals = series.sort_values().values
        diffs = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
        if len(set(diffs)) == 1 and diffs[0] == 1:
            return True

    if series.max() > 2100:
        return False

    return False

def _rescue_measures(id_cols: list, num_cols: list) -> tuple[list, list]:
    """
    If _is_id_like misfired and emptied num_cols, promote obvious measure
    columns back from id_cols.
    Returns updated (id_cols, num_cols).
    """
    if num_cols:
        return id_cols, num_cols

    rescued = [c for c in id_cols if any(h in c.lower() for h in _MEASURE_HINTS)]
    remaining_ids = [c for c in id_cols if c not in rescued]
    return remaining_ids, rescued

def _classify_columns(df: pd.DataFrame) -> dict:
    """
    Inspect df and return a dict with keys:
        all_num, id_cols, num_cols, date_cols, cat_cols, win_cols
    Also mutates df in-place for parsed date columns.
    """
    all_num  = df.select_dtypes(include="number").columns.tolist()
    id_cols  = [c for c in all_num if _is_id_like(df[c], c)]
    num_cols = [c for c in all_num if c not in id_cols]
    id_cols, num_cols = _rescue_measures(id_cols, num_cols)

    date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # attempt date parsing on string columns
    if not date_cols:
        for c in cat_cols[:]:
            try:
                df[c] = pd.to_datetime(df[c], format="mixed", errors="raise")
                date_cols.append(c)
                cat_cols.remove(c)
                break
            except Exception:
                pass

    win_cols = [c for c in df.columns if any(h in c.lower() for h in _WIN_ALL_HINTS)]

    return dict(
        all_num=all_num,
        id_cols=id_cols,
        num_cols=num_cols,
        date_cols=date_cols,
        cat_cols=cat_cols,
        win_cols=win_cols,
    )


# ══════════════════════════════════════════════════════════════════════════════
# LABEL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _truncate(s: str, n: int) -> str:
    return s[:n] + "…" if len(s) > n else s


def _pick_label_col(cat_cols: list, df: pd.DataFrame) -> str:
    """Choose the most meaningful category column to use as bar labels."""
    if len(cat_cols) == 1:
        return cat_cols[0]

    variable_cols = [c for c in cat_cols if df[c].nunique() > 1]
    if not variable_cols:
        return cat_cols[0]
    if len(variable_cols) == 1:
        return variable_cols[0]

    for hint in _LABEL_HINTS:
        for col in variable_cols:
            if hint in col.lower():
                return col

    return max(variable_cols, key=lambda c: df[c].nunique())


# def _clean_labels(labels: list, max_chars: int = 18,
#                   max_bars_before_number: int = 10) -> tuple[list, dict | None]:
#     """
#     Returns (display_labels, legend_map_or_None).
#     Strategies:
#       - Too many bars  → number them + return lookup dict
#       - Long avg name  → textwrap
#       - Very long name → truncate with ellipsis
#     """
#     n = len(labels)
#     avg_len = sum(len(str(l)) for l in labels) / max(n, 1)

#     if n > max_bars_before_number:
#         legend = {str(i + 1): str(l) for i, l in enumerate(labels)}
#         return [str(i + 1) for i in range(n)], legend

#     if avg_len > 30:
#         return [_truncate(str(l), max_chars) for l in labels], None

#     if avg_len > 20:
#         return [textwrap.fill(str(l), width=15) for l in labels], None

#     return [str(l) for l in labels], None

import textwrap

def _clean_labels(labels: list, max_chars: int = 20,
                  max_bars_before_number: int = 10) -> tuple[list, dict | None]:

    n = len(labels)
    avg_len = sum(len(str(l)) for l in labels) / max(n, 1)

    # too many bars → number mapping
    if n > max_bars_before_number:
        legend = {str(i + 1): str(l) for i, l in enumerate(labels)}
        return [str(i + 1) for i in range(n)], legend

    # 👇 FORCE wrapping for long labels
    wrapped = [textwrap.fill(str(l), width=20) for l in labels]
    return wrapped, None

def _draw_legend_table(ax: plt.Axes, legend: dict) -> None:
    """Render a number → full-name lookup table below the chart."""
    lines = [f"{k}: {v}" for k, v in legend.items()]
    half  = (len(lines) + 1) // 2
    ax.figure.text(0.1,  -0.02, "\n".join(lines[:half]),
                   fontsize=7, va="top", family="monospace")
    ax.figure.text(0.55, -0.02, "\n".join(lines[half:]),
                   fontsize=7, va="top", family="monospace")


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _scales_differ(s1: pd.Series, s2: pd.Series, threshold: float = 10.0) -> bool:
    """True when two series have very different magnitudes."""

    max1, max2 = s1.max(), s2.max()

    # handle constant column case
    if max1 == 0 or max2 == 0:
        return True

    ratio = max(max1, max2) / max(min(max1, max2), 1e-9)

    return ratio > threshold

def _bar_labels(ax: plt.Axes, x: list, values, fmt: str = "{:,.1f}") -> None:
    """Annotate bar tops with their values."""
    for xi, val in zip(x, values):
        ax.text(xi, val * 1.01, fmt.format(val),
                ha="center", va="bottom", fontsize=7)


# ══════════════════════════════════════════════════════════════════════════════
# CHART RENDERERS  (each takes ax + data slice; returns nothing)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_rank_bar(ax: plt.Axes, df: pd.DataFrame,
                   rank_col: str, val_col: str, label_col: str) -> None:
    plot_df    = df.sort_values(rank_col).head(20)
    raw_labels = plot_df[label_col].astype(str).tolist()
    cleaned, legend = _clean_labels(raw_labels)
    ax.barh(cleaned[::-1], plot_df[val_col].values[::-1],
            color=_BAR_COLORS, alpha=0.85)
    ax.set_xlabel(val_col)
    if legend:
        _draw_legend_table(ax, legend)


def _plot_area(ax: plt.Axes, df: pd.DataFrame,
               x_col, win_col: str) -> None:
    x = df[x_col] if isinstance(x_col, str) else df.index
    ax.fill_between(x, df[win_col], alpha=0.4, color=_BAR_COLORS[4])
    ax.plot(x, df[win_col], color=_BAR_COLORS[5])


def _plot_dual_line(ax: plt.Axes, df: pd.DataFrame,
                    x_col, base_cols: list, win_col: str) -> None:
    x = df[x_col] if isinstance(x_col, str) else df.index
    if base_cols:
        ax.plot(x, df[base_cols[0]], label=base_cols[0], color=_BAR_COLORS)
    ax.plot(x, df[win_col], linestyle="--", label=win_col, color=_BAR_COLORS)
    ax.legend()


def _plot_line(ax: plt.Axes, df: pd.DataFrame,
               x_col: str, y_col: str) -> None:
    df_sorted = df.sort_values(x_col)
    ax.plot(df_sorted[x_col], df_sorted[y_col], color=_BAR_COLORS[1])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)


def _plot_single_bar(ax: plt.Axes, df: pd.DataFrame,
                     label_col: str, val_col: str) -> None:
    plot_df    = df[[label_col, val_col]].dropna()
    plot_df    = plot_df.sort_values(val_col, ascending=False).head(25)
    raw_labels = plot_df[label_col].astype(str).tolist()
    avg_len    = sum(len(l) for l in raw_labels) / max(len(raw_labels), 1)
    cleaned, legend = _clean_labels(raw_labels)

    if avg_len > 15:
        ax.barh(cleaned[::-1], plot_df[val_col].values[::-1],
                color=_BAR_COLORS, alpha=0.85)
        ax.set_xlabel(val_col)
    else:
        x = range(len(cleaned))
        ax.bar(x, plot_df[val_col].values, color=_BAR_COLORS, alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(cleaned, ha="center", fontsize=8)
        ax.set_ylabel(val_col)

    if legend:
        _draw_legend_table(ax, legend)




def _plot_dual_subplot(fig: plt.Figure, df: pd.DataFrame,
                       label_col: str, num_cols: list,
                       cleaned_labels: list, legend: dict | None) -> None:
    """Replace fig with 2-row subplot figure (different scales)."""
    plt.close(fig)
    fig2, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x = list(range(len(df)))

    for i, col in enumerate(num_cols):
        axes[i].bar(x, df[col], color=_BAR_COLORS[i], alpha=0.85, width=0.6)
        axes[i].set_ylabel(col, fontsize=9)
        axes[i].set_title(col, fontsize=10, pad=4)
        axes[i].grid(axis="y", alpha=0.3, linewidth=0.5)
        _bar_labels(axes[i], x, df[col])

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(cleaned_labels, ha="center", fontsize=8)
    axes[-1].set_xlabel(label_col)
    if legend:
        _draw_legend_table(axes[-1], legend)

    return fig2


def _plot_same_scale_bar(ax: plt.Axes, df: pd.DataFrame,
                         label_col: str, num_cols: list,
                         cleaned_labels: list, legend: dict | None) -> None:
    """Grouped bar for two numeric columns on the same scale."""
    x, w = list(range(len(df))), 0.35
    ax.bar([i - w / 2 for i in x], df[num_cols[0]], width=w,
           label=num_cols[0], color=_BAR_COLORS, alpha=0.85)
    ax.bar([i + w / 2 for i in x], df[num_cols[1]], width=w,
           label=num_cols[1], color=_BAR_COLORS, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cleaned_labels, ha="center", fontsize=8)
    ax.set_xlabel(label_col)
    ax.legend(fontsize=8)
    if legend:
        _draw_legend_table(ax, legend)



def compress_label(s: str) -> str:
    if not isinstance(s, str):
        return s

    words = s.split()

    # If <= 3 words → keep as is
    if len(words) <= 3:
        return s

    new_words = []

    for i, w in enumerate(words):
        # Keep first 2 words
        if i < 2:
            new_words.append(w)

        else:
            # Keep numbers as is
            if any(char.isdigit() for char in w):
                new_words.append(w)
            else:
                new_words.append(w[0].upper())

    return " ".join(new_words)

# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def smart_plot(df: pd.DataFrame, output_dir: str,file_prefix: str, title: str = "Query Result") -> None:
    """
    Auto-detect the shape of a SQL result DataFrame and render
    the most meaningful matplotlib chart.
    """
    df.columns = df.columns.str.replace("_", " ").str.title()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.title()
        df[col] = df[col].apply(compress_label)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []
    
    cols   = _classify_columns(df)
    num    = cols["num_cols"]
    cat    = cols["cat_cols"]
    dates  = cols["date_cols"]
    wins   = cols["win_cols"]
    all_n  = cols["all_num"]
    n_rows = len(df)


    if n_rows < 3:
        return plot_paths
    print(f"id_cols={cols['id_cols']} | num_cols={num} | cat_cols={cat}")

    fig, ax = plt.subplots(figsize=(10, 5))

    # ── window: rank ──────────────────────────────────────────────────
    if any(h in c.lower() for c in wins for h in _WIN_RANK_HINTS):
        print("Scenario 1: Window rank Diagram")
        rank_col    = next(c for c in wins if any(h in c.lower() for h in _WIN_RANK_HINTS))
        val_col     = next((c for c in all_n if c != rank_col), rank_col)
        label_col   = _pick_label_col(cat, df)
        _plot_rank_bar(ax, df, rank_col, val_col, label_col)
        ax.set_title(title)

    # ── window: running total ─────────────────────────────────────────
    elif any(h in c.lower() for c in wins for h in _WIN_RUNNING_HINTS):
        print("Scenario 2: Window hints Diagram")
        x_col = dates[0] if dates else None
        _plot_area(ax, df, x_col, wins[0])
        ax.set_title(title)

    # ── window: lag / lead / diff ─────────────────────────────────────
    elif any(h in c.lower() for c in wins for h in _WIN_DIFF_HINTS):
        print("Scenario 3: Window diff hints Diagram")
        x_col      = dates[0] if dates else None
        base_cols  = [c for c in num if c not in wins]
        _plot_dual_line(ax, df, x_col, base_cols, wins[0])
        ax.set_title(title)

    # ── 1 numeric + date → line ───────────────────────────────────────
    elif len(num) == 1 and dates:
        print("Scenario 4: num_1 and date Diagram")
        _plot_line(ax, df, dates[0], num[0])
        ax.set_title(title)


    elif len(num) == 1 and cat:
        print("Scenario 5: num_1 and category Diagram")

        # 👉 Detect Year + Month case
        if "Year" in df.columns and "Month" in df.columns:
            print("→ Detected Year + Month → using Time Series")

            # Create proper date
            df["Date"] = pd.to_datetime(
                df["Year"].astype(str) + "-" + df["Month"],
                format="%Y-%b",
                errors="coerce"
            ).dt.date

            df = df.sort_values("Date")

            ax.plot(df["Date"], df[num[0]],
                    marker="o", color=_BAR_COLORS[3])

            ax.set_xlabel("Date")
            ax.set_ylabel(num[0])

        else:
            # fallback to your original logic
            label_col = _pick_label_col(cat, df)

            plot_df = df.groupby(label_col)[num[0]].sum().reset_index()
            n_unique = plot_df[label_col].nunique()

            if 2 <= n_unique <= 6:
                print("→ Using PIE chart")

                ax.pie(
                    plot_df[num[0]],
                    labels=plot_df[label_col],
                    autopct="%1.1f%%",
                    colors=_BAR_COLORS[:n_unique],
                    startangle=90
                )
                ax.axis("equal")

            else:
                print("→ Using BAR chart")
                _plot_single_bar(ax, df, label_col, num[0])

        ax.set_title(title)


    elif len(num) == 2 and n_rows > 50:
        print("Scenario 6: num_2 and many rows Diagram")
        ax.scatter(df[num[0]], df[num[1]], alpha=0.5, s=20, color=_BAR_COLORS[3])
        ax.set_xlabel(num[0])
        ax.set_ylabel(num[1])
        ax.set_title(title)

    # ── 2 numeric + category ─────────────────────────────────────────
    elif len(num) == 2 and cat:
        print("Scenario 7: num_2 and category Diagram")
        label_col = _pick_label_col(cat, df)
        df        = df.sort_values(num[0], ascending=False).reset_index(drop=True)
        cleaned, legend = _clean_labels(df[label_col].astype(str).tolist())

        if _scales_differ(df[num[0]], df[num[1]]):
            fig = _plot_dual_subplot(fig, df, label_col, num, cleaned, legend)
        else:
            _plot_same_scale_bar(ax, df, label_col, num, cleaned, legend)
        fig.suptitle(title, fontsize=12)

    elif len(num) >= 2 and (dates or ("Month" in df.columns and "Year" in df.columns)):
        print("Scenario NEW: Time series with multiple metrics")

        # Create proper datetime column
        if "Year" in df.columns and "Month" in df.columns:
            df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"], errors="coerce")
            df = df.sort_values("Date")
            x_col = "Date"
        else:
            x_col = dates[0]

        main_col = num[0]

        ax.plot(df[x_col], df[main_col],
                marker="o", color=_BAR_COLORS[3], label=main_col)

        # Optional: plot % change if exists
        pct_cols = [c for c in num if "pct" in c.lower() or "change" in c.lower()]
        if pct_cols:
            ax2 = ax.twinx()
            ax2.plot(df[x_col], df[pct_cols[0]],
                    linestyle="--", color=_BAR_COLORS[5], label=pct_cols[0])
            ax2.set_ylabel(pct_cols[0])
            ax2.legend(loc="upper right")

        ax.set_xlabel(x_col)
        ax.set_ylabel(main_col)
        ax.legend(loc="upper left")
        ax.set_title(title)
    else:
        print("Scenario 9: leftover table Diagram")
        plt.close(fig)
        return []

    # _add_subtitle(fig, n_rows, len(df.columns), num, cat)
    plt.tight_layout()
    path = os.path.join(output_dir, f"_{file_prefix}_{timestamp}_{unique_id}_bar.png")
    # plt.show()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    plot_paths.append(path)
    return plot_paths

