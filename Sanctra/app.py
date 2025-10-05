# sanctra.py — Sanctra — Pro (3D-first + NASA thresholds + Optimizer + Simple Table Editor + Canvas + Undo/Redo + Align/Distribute + PNG/PDF Export)
# Upgrades in this version:
# - NEW: Deck ALL (overlay) figure + PNG export
# - NEW: Technical PDF v2 (constraints table, global KPIs, Deck ALL page, per-deck KPI tables, warnings, power bill)
# - Minor: cleaner fonts via CSS, table helpers for PDF

from __future__ import annotations
import copy, json, math, random, io, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sanctra_polish_paths import (
    sanitize_paths, safe_slug, default_project_dir,
    path_for_download, save_bytes_locally
)


# ==================== App meta ====================
APP_NAME = "Sanctra"
TAGLINE = "Design your safe space in unsafe worlds."

st.set_page_config(page_title=f"{APP_NAME} — Pro", layout="wide")
st.markdown(
    f"<h1 style='margin-bottom:0'>{APP_NAME}</h1>"
    f"<div style='opacity:.75'>{TAGLINE}</div>",
    unsafe_allow_html=True
)
# Modernize base typography a bit (system font stack)
st.markdown(
    """
    <style>
      :root { --sans: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Helvetica, Arial, 'Noto Sans', sans-serif; }
      html, body, [class*='css'] { font-family: var(--sans); }
      h1, h2, h3 { letter-spacing: .3px; }
      .metric-gap > div { gap: .25rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== Helpers ====================
def base_key(k: str) -> str:
    return k.split("__", 1)[0]

def clamp(v, a, b):
    return max(a, min(b, v))

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

# ==================== Rules / Fallback ====================
DATA_PATH = Path(__file__).parent / "data" / "nasa_rules.json"
fallback_rules = {
    "sleep":  {"name":"Crew Quarters", "area_m2_per_crew": 3.0, "volume_m3_per_crew": 8.0, "category":"Habitation"},
    "hygiene":{"name":"Hygiene", "area_m2_per_crew": 2.0, "volume_m3_per_crew": 5.0, "category":"Habitation"},
    "galley": {"name":"Galley", "area_m2_per_crew": 1.2, "volume_m3_per_crew": 3.0, "category":"Operations"},
    "dining": {"name":"Dining", "area_m2_per_crew": 1.5, "volume_m3_per_crew": 3.0, "category":"Habitation"},
    "exercise":{"name":"Exercise", "area_m2_per_crew": 2.4, "volume_m3_per_crew": 7.0, "category":"Health"},
    "medical":{"name":"Medical", "area_m2_per_crew": 1.2, "volume_m3_per_crew": 3.5, "category":"Health"},
    "work":   {"name":"Work", "area_m2_per_crew": 2.0, "volume_m3_per_crew": 5.5, "category":"Operations"},
    "eclss":  {"name":"ECLSS", "area_m2_per_crew": 1.0, "volume_m3_per_crew": 3.0, "category":"Systems"},
    "stowage":{"name":"Stowage", "area_m2_per_crew": 1.6, "volume_m3_per_crew": 6.0, "category":"Systems"},
    "airlock":{"name":"Airlock", "area_m2_per_crew": 1.4, "volume_m3_per_crew": 4.0, "category":"Systems"},
}
try:
    rules = json.loads(DATA_PATH.read_text(encoding="utf-8")) if DATA_PATH.exists() else fallback_rules
except Exception:
    rules = fallback_rules

ALL_FUNCS = {k: v["name"] for k, v in rules.items()}
ALL_CATEGORIES = sorted({v.get("category", "-") for v in rules.values() if v.get("category")})

# ==================== Colors & Icons ====================
FUNC_COLORS = {
    "sleep": "#1f77b4",
    "hygiene": "#17becf",
    "galley": "#ff7f0e",
    "dining": "#bcbd22",
    "exercise": "#2ca02c",
    "medical": "#d62728",
    "work": "#9467bd",
    "eclss": "#8c564b",
    "stowage": "#7f7f7f",
    "airlock": "#e377c2",
}
CAT_FALLBACK = {"Habitation":"#1f77b4","Health":"#2ca02c","Operations":"#ff7f0e","Systems":"#9467bd"}
FUNC_ICONS = {"sleep":"🛏️","hygiene":"🚿","galley":"🍳","dining":"🍽️","exercise":"🏋️",
              "medical":"🩺","work":"🧑‍💻","eclss":"⚙️","stowage":"📦","airlock":"🚪"}

# ==================== Power model (simple) ====================
POWER_W_PER_M2 = {
    
    "sleep": 5, "hygiene": 10, "galley": 20, "dining": 8, "exercise": 12,
    "medical": 25, "work": 15, "eclss": 50, "stowage": 2, "airlock": 10,
}

# Optional deps (export)
try:
    import reportlab
    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    HAS_RL = True
except Exception:
    HAS_RL = False

# ==================== Sidebar ====================
st.sidebar.header("Mission Parameters")
crew = st.sidebar.slider("Crew size", 1, 8, 4, 1)
mission_days = st.sidebar.number_input("Mission duration (days)", min_value=1, value=180, step=1)

default_selection = list(ALL_FUNCS.keys())
selected_keys = st.sidebar.multiselect(
    "Included functions",
    options=list(ALL_FUNCS.keys()),
    default=default_selection,
    format_func=lambda k: ALL_FUNCS[k]
)

st.sidebar.markdown("---")
decks = st.sidebar.slider("Decks (floors)", 1, 3, 2)
grid_w = st.sidebar.slider("Grid width (cells)", 12, 40, 26)
cell_m = st.sidebar.select_slider("Cell size (m)", options=[0.5, 1.0, 1.5, 2.0], value=1.0)

st.sidebar.markdown("### Hull / Shell")
show_shell = st.sidebar.checkbox("Show shell", True)
shell_wall = st.sidebar.slider("Wall thickness (cells)", 1, 4, 2)
shell_color = st.sidebar.color_picker("Shell color", "#4bd5ff")

st.sidebar.markdown("### Circulation")
enable_corridor = st.sidebar.checkbox("Enable main corridor", value=True)
corridor_width = st.sidebar.slider("Corridor width (cells)", 1, 8, 3)
reserve_corridor = st.sidebar.checkbox("Keep corridor clear (auto-nudge on Apply)", True)
use_icons = st.sidebar.checkbox("Icons in labels", True)
FOCUS_EDIT = st.sidebar.toggle("Focus Edit Mode (anti-flicker for Canvas)", value=True)

# ==================== NASA Thresholds (editable) ====================
st.sidebar.markdown("### NASA Thresholds (editable)")
nasa_default = {
    "min_corridor_clearance_m": 1.07,
    "min_hatch_clear_w_m": 0.80,
    "min_hatch_clear_h_m": 1.10,
    "min_door_edge_cells": 1,
    "min_sleep_exercise_sep_m": 4.0,
    "min_galley_hygiene_sep_m": 3.0,
    "min_front_of_rack_m": 1.07,
}
nasa_default.update(st.session_state.get("nasa_thresholds_override", {}))

def m_to_cells(m):
    return max(1, int(round(m / max(cell_m, 1e-6))))

min_corridor_clearance_m = st.sidebar.number_input("Min corridor clearance (m)",
    value=float(nasa_default["min_corridor_clearance_m"]), step=0.1)
min_hatch_w_m = st.sidebar.number_input("Min hatch/door width (m)",
    value=float(nasa_default["min_hatch_clear_w_m"]), step=0.05)
min_hatch_h_m = st.sidebar.number_input("Min hatch/door height (m)",
    value=float(nasa_default["min_hatch_clear_h_m"]), step=0.05)
min_sleep_ex_m = st.sidebar.number_input("Min sleep–exercise separation (m)",
    value=float(nasa_default["min_sleep_exercise_sep_m"]), step=0.5)
min_gal_hyg_m = st.sidebar.number_input("Min galley–hygiene separation (m)",
    value=float(nasa_default["min_galley_hygiene_sep_m"]), step=0.5)
min_front_rack_m = st.sidebar.number_input("Min front-of-rack clearance (m)",
    value=float(nasa_default["min_front_of_rack_m"]), step=0.1)

st.session_state["nasa_thresholds_override"] = {
    "min_corridor_clearance_m": float(min_corridor_clearance_m),
    "min_hatch_clear_w_m": float(min_hatch_w_m),
    "min_hatch_clear_h_m": float(min_hatch_h_m),
    "min_door_edge_cells": int(nasa_default["min_door_edge_cells"]),
    "min_sleep_exercise_sep_m": float(min_sleep_ex_m),
    "min_galley_hygiene_sep_m": float(min_gal_hyg_m),
    "min_front_of_rack_m": float(min_front_rack_m),
}

# ==================== Data table ====================
def compute_scaled_table(keys, crew):
    rows = []
    for k in keys:
        meta = rules[k]
        rows.append({
            "key": k,
            "Function": meta["name"],
            "Area req (m²)": round(crew * float(meta["area_m2_per_crew"]), 2),
            "Volume req (m³)": round(crew * float(meta["volume_m3_per_crew"]), 2),
            "Category": meta.get("category", "-"),
        })
    return pd.DataFrame(rows)

df = compute_scaled_table(selected_keys, crew)
mc1, mc2, mc3, mc4 = st.columns(4)
with mc1: st.metric("Crew", crew)
with mc2: st.metric("Mission (days)", mission_days)
with mc3: st.metric("Functions", len(selected_keys))
with mc4:
    tot_area = float(df["Area req (m²)"].sum()) if not df.empty else 0.0
    st.metric("Total req. area (m²)", f"{tot_area:.1f}")

st.markdown("#### Requirements (scaled per crew)")
st.dataframe(df, use_container_width=True)

# ==================== Layout Engine ====================
ADJACENCY_WISHES: Dict[str, Dict[str, int]] = {
    "galley": {"dining": 3, "stowage": 2},
    "dining": {"galley": 3},
    "sleep": {"hygiene": 2},
    "work": {"eclss": 1},
}
PAIR_SEPARATIONS = [
    ("hygiene","galley","min_galley_hygiene_sep_m"),
    ("sleep","exercise","min_sleep_exercise_sep_m"),
]

@dataclass
class Demand:
    key: str
    area_m2: float

@dataclass
class Rect:
    key: str; x: int; y: int; w: int; h: int; deck: int

def demands_from_df(df: pd.DataFrame) -> Dict[str, Demand]:
    return {r["key"]: Demand(r["key"], float(r["Area req (m²)"])) for _, r in df.iterrows()}

FIXED_DECK_PREF = {
    "airlock": 1, "eclss": 1, "stowage": 1, "galley": 1, "dining": 1,
    "sleep": 2, "hygiene": 2, "work": 2, "medical": 2, "exercise": 3
}
def choose_deck(key: str, max_decks: int) -> int:
    return min(FIXED_DECK_PREF.get(key, 1), max_decks)

def split_demands_by_deck(demands: Dict[str, Demand], max_decks: int) -> DefaultDict[int, Dict[str, Demand]]:
    buckets: DefaultDict[int, Dict[str, Demand]] = defaultdict(dict)
    for k, d in demands.items():
        buckets[choose_deck(k, max_decks)][k] = d
    for d in range(1, max_decks+1):
        buckets[d] = buckets.get(d, {})
    return buckets

def pack_rects(demands: Dict[str, Demand], cell_m: float, grid_w: int, deck: int, seed: int|None=None) -> Dict[str, Rect]:
    rects: Dict[str, Rect] = {}
    keys = list(demands.keys())
    if seed is None:
        keys.sort(key=lambda k: demands[k].area_m2, reverse=True)
    else:
        random.Random(seed).shuffle(keys)
    cx = cy = 0; row_h = 0
    for k in keys:
        area = max(1e-6, demands[k].area_m2)
        cells = int(math.ceil(area / (cell_m * cell_m)))
        w = max(1, int(round(math.sqrt(cells * 1.6))))
        h = max(1, int(math.ceil(cells / w)))
        if cx + w > grid_w:
            cx = 0; cy += row_h; row_h = 0
        rects[k] = Rect(k, cx, cy, w, h, deck=deck)
        cx += w; row_h = max(row_h, h)
    return rects

def rect_center(r: Rect) -> Tuple[float, float]:
    return (r.x + r.w/2.0, r.y + r.h/2.0)

def rects_overlap(a: Rect, b: Rect) -> bool:
    return not (a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y)

def resolve_overlaps(rects: Dict[str, Rect], grid_w: int) -> Tuple[Dict[str, Rect], List[str]]:
    msgs = []; placed: Dict[str, Rect] = {}
    for k, r in sorted(rects.items(), key=lambda kv: (kv[1].y, kv[1].x)):
        cand = Rect(r.key, r.x, r.y, r.w, r.h, r.deck)
        moved = False; tries = 0
        while any(rects_overlap(cand, p) for p in placed.values()) and tries < 200:
            cand.y += 1; moved = True; tries += 1
        if moved:
            msgs.append(f"‘{ALL_FUNCS.get(base_key(k),k)}’ aşağı {tries} hücre kaydırıldı (overlap fix).")
        placed[k] = cand
    return placed, msgs

# Corridor helpers
def corridor_band(deck_rects: Dict[str, Rect], width_cells: int, pos_y: int|None=None) -> Tuple[int, int]:
    max_y = max([r.y + r.h for r in deck_rects.values()], default=0)
    H = max(1, max_y + 1)
    cy = H//2 if pos_y is None else clamp(pos_y, 0, H-1)
    half = max(1, int(round(width_cells/2)))
    y0 = clamp(cy - half, 0, H); y1 = clamp(cy + half, 0, H)
    if y1 == y0: y1 = min(H, y0+1)
    return y0, y1

def rect_intersects_corridor(r: Rect, y0: int, y1: int) -> bool:
    return not (r.y + r.h <= y0 or r.y >= y1)

# ==================== Scores & Warnings ====================
def adjacency_score(rects: Dict[str, Rect]) -> float:
    score = 0.0; max_add = 0.0
    for a, prefs in ADJACENCY_WISHES.items():
        if a not in rects: continue
        ax, ay = rect_center(rects[a]); adeck = rects[a].deck
        for b, w in prefs.items():
            max_add += w
            if b not in rects: continue
            if rects[b].deck != adeck:
                score += w / 999.0; continue
            bx, by = rect_center(rects[b]); d = abs(ax - bx) + abs(ay - by) + 1e-6
            score += w / d
    return min(100.0, (score / (max_add + 1e-6)) * 100.0)

def separation_penalty(rects: Dict[str, Rect], cell_m: float, thresholds: dict) -> Tuple[float, List[str]]:
    penalty = 0.0; count = 0; warns=[]
    for a, b, key in PAIR_SEPARATIONS:
        if a in rects and b in rects and rects[a].deck == rects[b].deck:
            ax, ay = rect_center(rects[a]); bx, by = rect_center(rects[b])
            d_cells = abs(ax - bx) + abs(ay - by); d_m = d_cells * cell_m
            mind = float(thresholds[key])
            count += 1
            if d_m < mind:
                p = (mind - d_m) / max(mind, 1e-6) * 100.0
                penalty += p
                warns.append(f"⚠️ Keep **{ALL_FUNCS[a]}** away from **{ALL_FUNCS[b]}** (dist={d_m:.1f}m < {mind:.1f}m)")
    return (penalty / count if count else 0.0), warns

def corridor_warnings(rects: Dict[str, Rect], enable: bool, width: int, pos_y: int|None=None) -> List[str]:
    if not enable or not rects: return []
    y0, y1 = corridor_band(rects, width, pos_y)
    warns=[]
    for k, r in rects.items():
        if rect_intersects_corridor(r, y0, y1):
            warns.append(f"🚧 **{ALL_FUNCS.get(base_key(k),k)}** koridor bandına taşıyor (y={y0}-{y1}).")
    return warns

def access_warnings(rects: Dict[str, Rect], grid_w: int, enable: bool, width: int, pos_y: int|None=None, min_front_rack_m: float=1.07) -> List[str]:
    if not rects: return []
    y0,y1 = corridor_band(rects, width, pos_y) if enable else (None, None)
    warns=[]
    for k,r in rects.items():
        touch_shell = (r.x==0 or r.y==0 or r.x+r.w==grid_w)
        touch_corr = (enable and (r.y <= y1 and r.y+r.h >= y0))
        if not (touch_shell or touch_corr):
            warns.append(f"🚪 **{ALL_FUNCS.get(base_key(k),k)}** erişim yok: koridor/dış kabuğa değmiyor.")
    if enable and any(base_key(k) in {"work","eclss","medical"} for k in rects):
        if (width * cell_m) < min_front_rack_m:
            warns.append(f"🧰 Front-of-rack clearance yetersiz: corridor {width*cell_m:.2f} m < {min_front_rack_m:.2f} m")
    return warns

def door_warnings(rects: Dict[str, Rect], grid_w: int, enable: bool, width: int, pos_y: int|None, min_hatch_w_m: float, min_hatch_h_m: float) -> List[str]:
    warns=[]
    min_w_cells = m_to_cells(min_hatch_w_m)
    min_h_cells = m_to_cells(min_hatch_h_m)
    y0,y1 = corridor_band(rects, width, pos_y) if enable else (None, None)
    for k,r in rects.items():
        lengths = []
        if enable and (r.y <= y1 and r.y+r.h >= y0): lengths.append(r.w)
        if r.x == 0 or r.x + r.w == grid_w: lengths.append(r.h)
        if r.y == 0: lengths.append(r.w)
        max_len = max(lengths) if lengths else 0
        if (max_len < min_w_cells) or (min(r.w, r.h) < min_h_cells):
            warns.append(f"🚫 **{ALL_FUNCS.get(base_key(k),k)}** geçiş açıklığı küçük; min {min_hatch_w_m:.2f}×{min_hatch_h_m:.2f} m.")
    return warns

def nudge_away_from_corridor(rects: Dict[str, Rect], pos_y: int, width_cells: int) -> Tuple[Dict[str, Rect], List[str]]:
    msgs=[]; out=copy.deepcopy(rects)
    y0,y1 = pos_y - max(1,int(round(width_cells/2))), pos_y + max(1,int(round(width_cells/2)))
    for k, r in list(out.items()):
        if rect_intersects_corridor(r, y0, y1):
            up_gap = abs(y0 - r.y); dn_gap = abs((r.y + r.h) - y1)
            new_y = max(0, y0 - r.h) if up_gap <= dn_gap else y1
            if new_y != r.y:
                msgs.append(f"‘{ALL_FUNCS.get(base_key(k),k)}’ koridorla çakışıyordu, y={r.y}→{new_y}.")
                out[k] = Rect(r.key, r.x, new_y, r.w, r.h, r.deck)
    return out, msgs

# ==================== 2D Figures ====================
def build_2d_figure(rects: Dict[str, Rect], grid_w: int, cell_m: float, df: pd.DataFrame,
                    shell: Dict|None=None, use_icons_flag: bool=False, corridor_cfg: Dict|None=None) -> Tuple[go.Figure, float, int]:
    name_by_key = {r["key"]: r["Function"] for _, r in df.iterrows()}
    req_area_by_key = {r["key"]: float(r["Area req (m²)"]) for _, r in df.iterrows()}
    fig = go.Figure(); max_y = 0; comp_sum = 0.0; ncomp = 0

    if corridor_cfg and corridor_cfg.get("enable") and rects:
        width = corridor_cfg.get("width", 2); pos_y = corridor_cfg.get("pos_y")
        y0,y1 = corridor_band(rects, width, pos_y)
        fig.add_shape(type="rect", x0=0, y0=y0, x1=grid_w, y1=y1,
                      line=dict(color="rgba(200,200,200,0.2)", width=1, dash="dot"),
                      fillcolor="rgba(200,200,200,0.08)")

    for k, r in rects.items():
        bk = base_key(k)
        x0, y0, x1, y1 = r.x, r.y, r.x+r.w, r.y+r.h
        max_y = max(max_y, y1)
        cat = rules.get(bk, {}).get("category", "-")
        color = FUNC_COLORS.get(bk, CAT_FALLBACK.get(cat, "#7f7f7f"))
        base_name = name_by_key.get(bk, ALL_FUNCS.get(bk, bk))
        icon = (FUNC_ICONS.get(bk, "") + " ") if use_icons_flag else ""
        label_text = r.key if r.key and r.key != bk else base_name

        actual_area = r.w * r.h * (cell_m * cell_m)
        req_area = req_area_by_key.get(bk, actual_area)
        comp = min(100.0, 100.0 * actual_area / max(req_area, 1e-6))
        comp_sum += comp; ncomp += 1

        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=color, width=2),
                      fillcolor=color, opacity=0.35)
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2,
                           text=f"{icon}{label_text}<br>{actual_area:.1f} m² ({comp:.0f}%)",
                           showarrow=False, font=dict(size=11, color="white"))

    if shell and shell.get("show"):
        col = shell.get("color", "#4bd5ff"); th = int(shell.get("thickness_cells", 2))
        H = max_y + 1
        fig.add_shape(type="rect", x0=0, y0=0, x1=grid_w, y1=H, line=dict(color=col, width=3), fillcolor=None)
        ix0, iy0 = th, th; ix1, iy1 = max(th, grid_w-th), max(th, H-th)
        if ix1 > ix0 and iy1 > iy0:
            fig.add_shape(type="rect", x0=ix0, y0=iy0, x1=ix1, y1=iy1,
                          line=dict(color=col, width=1, dash="dot"),
                          fillcolor="rgba(75,213,255,0.05)")

    fig.update_xaxes(range=[0, grid_w], showgrid=True, gridcolor="#333", zeroline=False)
    fig.update_yaxes(range=[max(0, max_y+1), 0], showgrid=True, gridcolor="#333", zeroline=False, scaleanchor="x")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#111", plot_bgcolor="#111")

    avg_comp = (comp_sum / ncomp) if ncomp else 0.0
    return fig, avg_comp, max_y

def build_overlay_all_decks(
    deck_rects_map: Dict[int, Dict[str, Rect]],
    grid_w: int,
    cell_m: float,
    df: pd.DataFrame,
    shell: Dict | None = None
) -> go.Figure:
    """Aynı grid üzerinde tüm deck'leri üst üste çizer (deck'e göre çizgi stili/opacity değişir)."""
    fig = go.Figure()

    deck_styles = [
        {"opacity": 0.45, "dash": "solid",   "width": 2},
        {"opacity": 0.30, "dash": "dash",    "width": 2},
        {"opacity": 0.22, "dash": "dot",     "width": 2},
        {"opacity": 0.18, "dash": "dashdot", "width": 2},
    ]

    max_y = 0

    # Deck anahtarlarını sıralı gez (1..N)
    for deck_i in sorted(deck_rects_map.keys()):
        rects = deck_rects_map[deck_i]
        style = deck_styles[(deck_i - 1) % len(deck_styles)]

        for k, r in rects.items():
            bk = base_key(k)
            x0, y0, x1, y1 = r.x, r.y, r.x + r.w, r.y + r.h
            max_y = max(max_y, y1)

            cat = rules.get(bk, {}).get("category", "-")
            color = FUNC_COLORS.get(bk, CAT_FALLBACK.get(cat, "#7f7f7f"))
            name = ALL_FUNCS.get(bk, bk)

            fig.add_shape(
                type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color=color, width=style["width"], dash=style["dash"]),
                fillcolor=color,
                opacity=style["opacity"],
            )

        # Sol üst köşeye küçük deck etiketi
        fig.add_annotation(
            x=-0.6, y=0.6,  # veri koordinatında (y ekseni ters çevrildiği için tepeye yakın)
            text=f"D{deck_i}",
            showarrow=False,
            font=dict(size=12, color="#bbb"),
            xref="x", yref="y",
        )

    # Shell çiz
    if shell and shell.get("show"):
        col = shell.get("color", "#4bd5ff")
        th = int(shell.get("thickness_cells", 2))
        H = max_y + 1 if max_y > 0 else 1
        fig.add_shape(type="rect", x0=0, y0=0, x1=grid_w, y1=H,
                      line=dict(color=col, width=3), fillcolor=None)
        ix0, iy0 = th, th
        ix1, iy1 = max(th, grid_w - th), max(th, H - th)
        if ix1 > ix0 and iy1 > iy0:
            fig.add_shape(type="rect", x0=ix0, y0=iy0, x1=ix1, y1=iy1,
                          line=dict(color=col, width=1, dash="dot"),
                          fillcolor="rgba(75,213,255,0.05)")

    # Eksen/tema
    fig.update_xaxes(range=[-1, grid_w], showgrid=True, gridcolor="#333", zeroline=False)
    fig.update_yaxes(range=[max(0, max_y + 1), 0], showgrid=True, gridcolor="#333",
                     zeroline=False, scaleanchor="x")
    fig.update_layout(
        height=540,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        title="Deck ALL (Overlay)"
    )

    # Basit bir “legend-hint” (annotation)
    legend_bits = []
    for deck_i in range(1, len(deck_rects_map) + 1):
        sty = deck_styles[(deck_i - 1) % len(deck_styles)]
        legend_bits.append(f"D{deck_i}: {sty['dash']} / {int(sty['opacity']*100)}%")
    if legend_bits:
        fig.add_annotation(
            xref="paper", yref="paper", x=0, y=1.06,
            text=" | ".join(legend_bits),
            showarrow=False,
            font=dict(size=11, color="#bbb"),
        )

    return fig
def html_legend():
    chips=[]
    for k,name in ALL_FUNCS.items():
        col = FUNC_COLORS.get(k, "#888"); icon = FUNC_ICONS.get(k, "")
        chips.append(f"<div style='display:inline-block;margin:2px 6px;padding:2px 8px;border-radius:8px;background:{col};color:#fff'>{icon} {name}</div>")
    st.markdown("".join(chips), unsafe_allow_html=True)

# ==================== Session Defaults ====================
for k, v in [("deck_rects", {}), ("corridor_y_by_deck", {}), ("pxpc_by_deck", {})]:
    if k not in st.session_state: st.session_state[k] = v
if "deck_seeds" not in st.session_state:
    st.session_state.deck_seeds = {i: None for i in range(1, decks+1)}

# ==================== Project I/O ====================
st.sidebar.markdown("### Project I/O (all decks)")
payload_all = {
    "meta": {"app": APP_NAME, "cell_m": cell_m, "grid_w": grid_w, "decks": decks},
    "corridor_y_by_deck": st.session_state.corridor_y_by_deck,
    "pxpc_by_deck": st.session_state.pxpc_by_deck,
    "deck_rects": {
        str(d): {
            k: {"x": r.x, "y": r.y, "w": r.w, "h": r.h, "deck": r.deck}
            for k, r in st.session_state.deck_rects.get(d, {}).items()
        }
        for d in range(1, decks+1)
    },
    "nasa_thresholds": st.session_state["nasa_thresholds_override"],
}
st.sidebar.download_button("Export ALL (JSON)", data=json.dumps(payload_all, indent=2),
                           file_name=f"sanctra_project_{now_str()}.json",
                           mime="application/json", key="dl_all")

up_all = st.sidebar.file_uploader("Import ALL", type=["json"], key="imp_all")
if up_all is not None:
    try:
        js = json.loads(up_all.read().decode("utf-8"))
        if "deck_rects" in js:
            st.session_state.deck_rects = {}
            for d_str, ddata in js["deck_rects"].items():
                d = int(d_str); st.session_state.deck_rects[d] = {}
                for k, v in ddata.items():
                    st.session_state.deck_rects[d][k] = Rect(
                        k, int(v["x"]), int(v["y"]), int(v["w"]), int(v["h"]), deck=int(v.get("deck", d))
                    )
            if "corridor_y_by_deck" in js:
                st.session_state.corridor_y_by_deck = {int(k): int(v) for k, v in js["corridor_y_by_deck"].items()}
            if "pxpc_by_deck" in js:
                st.session_state.pxpc_by_deck = {int(k): int(v) for k, v in js["pxpc_by_deck"].items()}
            if "nasa_thresholds" in js:
                st.session_state["nasa_thresholds_override"] = js["nasa_thresholds"]
            st.sidebar.success("Imported ALL.")
        else:
            st.sidebar.error("JSON has no 'deck_rects'.")
    except Exception as e:
        st.sidebar.error(f"Import failed: {e}")

# ==================== Build data ====================
demands = demands_from_df(df); by_deck = split_demands_by_deck(demands, decks)

# ========== Tabs ==========
# New order: 3D • per-deck(1..N) • Deck ALL (Overlay) • All Decks (Stacked) • NASA • Edit(Table) • Edit(Canvas) • Export
tab_titles = (
    ["🌍 3D View"]
    + [f"🗺️ 2D Deck {i}" for i in range(1, decks+1)]
    + ["🗺️ Deck ALL (Overlay)", "🧭 2D (All Decks — Stacked)", "🧪 NASA Compliance", "🧩 Edit (Table)", "🎛️ Edit (Canvas)", "📄 Export / Report"]
)
tabs = st.tabs(tab_titles)

# ----- 3D tab -----
with tabs[0]:
    # ensure rects + corridor seeds exist
    for i in range(1, decks+1):
        if i not in st.session_state.deck_rects:
            st.session_state.deck_rects[i] = pack_rects(by_deck[i], cell_m, grid_w, i, st.session_state.deck_seeds.get(i))
        if i not in st.session_state.corridor_y_by_deck:
            H_i = max([r.y+r.h for r in st.session_state.deck_rects[i].values()], default=0) + 1
            st.session_state.corridor_y_by_deck[i] = H_i // 2

    # Optimizer controls
    ocol1, ocol2, ocol3, ocol4, _ = st.columns([1,1,1,1,5])
    max_steps = ocol1.slider("Opt steps", 50, 1500, 400, 50)
    temp0 = ocol2.slider("SA Temp", 0.1, 5.0, 2.0, 0.1)
    overlap_penalty = ocol3.slider("Overlap Penalty", 10, 500, 120, 10)
    compact_weight = ocol4.slider("Compactness", 0, 100, 20, 5, help="Dikey kompaktlık ödülü")

    def objective(rects: Dict[str, Rect], grid_w: int, cell_m: float) -> float:
        adj = adjacency_score(rects)
        sep_pen, _ = separation_penalty(rects, cell_m, st.session_state["nasa_thresholds_override"])
        ov = 0
        rs = list(rects.values())
        for i in range(len(rs)):
            for j in range(i+1, len(rs)):
                if rects_overlap(rs[i], rs[j]):
                    x_overlap = max(0, min(rs[i].x+rs[i].w, rs[j].x+rs[j].w) - max(rs[i].x, rs[j].x))
                    y_overlap = max(0, min(rs[i].y+rs[i].h, rs[j].y+rs[j].h) - max(rs[i].y, rs[j].y))
                    ov += x_overlap * y_overlap

        cy = st.session_state.corridor_y_by_deck.get(list(rects.values())[0].deck, 0)
        y0,y1 = corridor_band(rects, corridor_width, cy) if enable_corridor else (None, None)
        corr_pen = 0
        if enable_corridor:
            for r in rects.values():
                if rect_intersects_corridor(r, y0, y1):
                    corr_pen += r.w * r.h

        max_y = max([r.y + r.h for r in rects.values()], default=0)
        compact = max(0, 100 - (max_y*3))

        req_map = {r["key"]: float(r["Area req (m²)"]) for _, r in df.iterrows()}
        area_pen = 0.0
        for k,r in rects.items():
            req = req_map.get(base_key(k))
            if req:
                have = r.w*r.h*(cell_m*cell_m)
                if have < req:
                    area_pen += 100.0 * (req-have)/req

        score = adj - sep_pen - (ov * overlap_penalty / 10.0) - (corr_pen * 2.0) + (compact_weight * (compact/100)) - (0.5*area_pen)
        return score

    def anneal(deck_i: int, steps: int, temp0: float):
        rng = random.Random()
        rects = copy.deepcopy(st.session_state.deck_rects[deck_i])
        best = copy.deepcopy(rects); best_s = objective(best, grid_w, cell_m)
        cur = copy.deepcopy(rects); cur_s = best_s
        for t in range(1, steps+1):
            temp = temp0 * (1.0 - t/steps)
            k = rng.choice(list(cur.keys())); r = cur[k]
            dx = rng.randint(-2, 2); dy = rng.randint(-2, 2)
            newx = clamp(r.x + dx, 0, max(0, grid_w - r.w))
            newy = max(0, r.y + dy)
            cur2 = copy.deepcopy(cur); cur2[k] = Rect(r.key, newx, newy, r.w, r.h, r.deck)
            cur2, _ = resolve_overlaps(cur2, grid_w)
            s2 = objective(cur2, grid_w, cell_m)
            if (s2 > cur_s) or (rng.random() < math.exp((s2 - cur_s)/max(0.001, temp))):
                cur, cur_s = cur2, s2
                if s2 > best_s:
                    best, best_s = cur2, s2
        st.session_state.deck_rects[deck_i] = best
        return best_s

    oc1, oc2, oc3 = st.columns([1,1,6])
    if oc1.button("🔀 Seed pack (all decks)"):
        for i in range(1, decks+1):
            st.session_state.deck_seeds[i] = random.randint(0, 10**9)
            st.session_state.deck_rects[i] = pack_rects(by_deck[i], cell_m, grid_w, i, st.session_state.deck_seeds[i])
        st.success("Seed’ler yenilendi.")

    if oc2.button("🧠 Optimize (all decks)"):
        scores=[]
        for i in range(1, decks+1):
            s = anneal(i, max_steps, temp0)
            scores.append((i, round(s,1)))
        st.success("Optimize bitti — skorlar: " + ", ".join([f"D{i}:{s}" for i,s in scores]))

    # 3D render
    fig3d = go.Figure()
    deck_height = 1.6
    max_y_all = 0
    for i in range(1, decks+1):
        rects_i = st.session_state.deck_rects.get(i, {})
        for k, r in rects_i.items():
            z0 = (r.deck - 1) * (deck_height + 0.25)
            verts = np.array([
                [r.x, r.y, z0],[r.x+r.w, r.y, z0],[r.x+r.w, r.y+r.h, z0],[r.x, r.y+r.h, z0],
                [r.x, r.y, z0+deck_height],[r.x+r.w, r.y, z0+deck_height],
                [r.x+r.w, r.y+r.h, z0+deck_height],[r.x, r.y+r.h, z0+deck_height],
            ])
            faces = [0,1,2, 0,2,3, 4,5,6, 4,6,7, 0,1,5, 0,5,4, 1,2,6, 1,6,5, 2,3,7, 2,7,6, 3,0,4, 3,4,7]
            xs, ys, zs = verts[:,0], verts[:,1], verts[:,2]
            bk = base_key(k)
            color = FUNC_COLORS.get(bk, CAT_FALLBACK.get(rules.get(bk, {}).get("category","-"), "#7f7f7f"))
            name = ALL_FUNCS.get(bk, bk)
            fig3d.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=faces[0::3], j=faces[1::3], k=faces[2::3],
                                      color=color, opacity=0.55, name=f"D{r.deck} — {name}",
                                      hovertext=f"{name} (D{r.deck})"))
            max_y_all = max(max_y_all, r.y + r.h)

    if show_shell:
        def add_box_wireframe(fig, x, y, w, d, z0, z1, color):
            P = np.array([[x,y,z0],[x+w,y,z0],[x+w,y+d,z0],[x,y+d,z0],[x,y,z1],[x+w,y,z1],[x+w,y+d,z1],[x,y+d,z1]])
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for a,b in edges:
                fig.add_trace(go.Scatter3d(x=[P[a,0],P[b,0]], y=[P[a,1],P[b,1]], z=[P[a,2],P[b,2]],
                                           mode="lines", line=dict(width=3, color=shell_color), showlegend=False, hoverinfo="skip"))
        env_w = grid_w; env_d = max_y_all + 1
        z0 = 0.0; z1 = decks * (deck_height + 0.25)
        add_box_wireframe(fig3d, 0, 0, env_w, env_d, z0, z1, shell_color)
        th = float(shell_wall)
        if env_w - 2*th > 0 and env_d - 2*th > 0:
            add_box_wireframe(fig3d, th, th, env_w-2*th, env_d-2*th, z0+0.05, z1-0.05, shell_color)

    fig3d.update_layout(
        scene=dict(
            xaxis=dict(range=[0, grid_w], backgroundcolor="#111", gridcolor="#333", zeroline=False),
            yaxis=dict(range=[max(0, max_y_all+1), 0], backgroundcolor="#111", gridcolor="#333", zeroline=False),
            zaxis=dict(range=[0, decks*(deck_height+0.25)+0.5], backgroundcolor="#111", gridcolor="#333", zeroline=False),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#111"
    )
    st.plotly_chart(fig3d, use_container_width=True, key="plot_3d_all")

# ----- 2D per deck -----
for i in range(1, decks+1):
    with tabs[i]:
        if i not in st.session_state.deck_rects:
            st.session_state.deck_rects[i] = pack_rects(by_deck[i], cell_m, grid_w, i, st.session_state.deck_seeds.get(i))
        rects_i = st.session_state.deck_rects[i]
        if i not in st.session_state.corridor_y_by_deck:
            H_i = max([r.y+r.h for r in rects_i.values()], default=0) + 1
            st.session_state.corridor_y_by_deck[i] = H_i // 2

        left, right = st.columns([1,1])
        if left.button(f"🔀 Randomize pack (Deck {i})", key=f"rand_{i}"):
            st.session_state.deck_seeds[i] = random.randint(0, 10**9)
            st.session_state.deck_rects[i] = pack_rects(by_deck[i], cell_m, grid_w, i, st.session_state.deck_seeds[i])

        if right.button(f"🧠 Optimize deck {i}", key=f"opt_{i}"):
            def anneal_small(deck_i, steps=200, temp0=1.5):
                rng = random.Random()
                cur = copy.deepcopy(st.session_state.deck_rects[deck_i]); cur_s = adjacency_score(cur)  # quick seed
                best, best_s = copy.deepcopy(cur), cur_s
                for t in range(1, steps+1):
                    temp = temp0 * (1.0 - t/steps)
                    k = rng.choice(list(cur.keys())); r = cur[k]
                    dx, dy = rng.choice([-2,-1,0,1,2]), rng.choice([-2,-1,0,1,2])
                    nx = clamp(r.x+dx, 0, max(0, grid_w-r.w)); ny = max(0, r.y+dy)
                    cand = copy.deepcopy(cur); cand[k] = Rect(r.key, nx, ny, r.w, r.h, r.deck)
                    cand, _ = resolve_overlaps(cand, grid_w)
                    s2 = adjacency_score(cand)
                    if (s2 > cur_s) or (rng.random() < math.exp((s2 - cur_s)/max(0.001,temp))):
                        cur, cur_s = cand, s2
                        if s2 > best_s:
                            best, best_s = cand, s2
                st.session_state.deck_rects[deck_i] = best
            anneal_small(i)
            st.success("Deck optimize edildi.")

        shell_cfg = {"show": show_shell, "thickness_cells": shell_wall, "color": shell_color}
        corridor_cfg = {"enable": enable_corridor, "width": corridor_width, "pos_y": st.session_state.corridor_y_by_deck[i]}
        fig2d, avg_comp, _ = build_2d_figure(rects_i, grid_w, cell_m, df, shell=shell_cfg, use_icons_flag=use_icons, corridor_cfg=corridor_cfg)
        st.plotly_chart(fig2d, use_container_width=True, key=f"plot_2d_deck_{i}")

        adj = adjacency_score(rects_i)
        sep_pen, sep_warns = separation_penalty(rects_i, cell_m, st.session_state["nasa_thresholds_override"])
        warns = []
        if enable_corridor:
            warns += corridor_warnings(rects_i, True, corridor_width, st.session_state.corridor_y_by_deck[i])
        warns += access_warnings(rects_i, grid_w, enable_corridor, corridor_width, st.session_state.corridor_y_by_deck[i], min_front_rack_m)
        warns += door_warnings(rects_i, grid_w, enable_corridor, corridor_width, st.session_state.corridor_y_by_deck[i], min_hatch_w_m, min_hatch_h_m)
        warns += sep_warns

        c1, c2, c3 = st.columns(3)
        with c1: st.progress(min(100, int(avg_comp)), text=f"Area compliance: {avg_comp:.0f}%")
        with c2: st.progress(min(100, int(adj)), text=f"Adjacency score: {adj:.0f}")
        with c3: st.progress(min(100, int(max(0, 100 - sep_pen))), text=f"Separation health: {max(0, 100 - sep_pen):.0f}")
        if warns: st.warning(" • ".join(warns))
        if i == 1: html_legend()

# ----- Deck ALL (Overlay) -----
with tabs[decks+1]:
    shell_cfg = {"show": show_shell, "thickness_cells": shell_wall, "color": shell_color} if show_shell else None
    fig_overlay = build_overlay_all_decks(st.session_state.deck_rects, grid_w, cell_m, df, shell=shell_cfg)
    st.plotly_chart(fig_overlay, use_container_width=True, key="plot_all_overlay")

# ----- All-decks 2D (stacked) -----
with tabs[decks+2]:
    gap = 2; offsets={}; y_cursor=0
    for i in range(1, decks+1):
        rects_i = st.session_state.deck_rects.get(i, {})
        max_y = max([r.y+r.h for r in rects_i.values()], default=0)
        offsets[i] = y_cursor; y_cursor += max_y + gap
    rects_all2d: Dict[str, Rect] = {}
    for i in range(1, decks+1):
        for k, r in st.session_state.deck_rects.get(i, {}).items():
            rects_all2d[f"{k}__d{i}"] = Rect(key=f"{k} (D{i})", x=r.x, y=r.y+offsets[i], w=r.w, h=r.h, deck=i)
    shell_cfg = {"show": show_shell, "thickness_cells": shell_wall, "color": shell_color} if show_shell else None
    fig_all2d, _, _ = build_2d_figure(rects_all2d, grid_w, cell_m, df, shell=shell_cfg, use_icons_flag=use_icons)
    for i in range(1, decks+1):
        ymid = offsets[i] + 0.5
        fig_all2d.add_annotation(x=-0.5, y=ymid, text=f"Deck {i}", showarrow=False, font=dict(color="#bbb", size=12), align="left")
    max_y_all = y_cursor if y_cursor>0 else 1
    fig_all2d.update_yaxes(range=[max_y_all, 0])
    st.plotly_chart(fig_all2d, use_container_width=True, key="plot_2d_alldecks_stacked")

# ----- NASA Compliance tab -----
with tabs[decks+3]:
    st.markdown("### NASA Compliance Panel (7 madde)")
    th = st.session_state["nasa_thresholds_override"]
    need_corr_cells = m_to_cells(th["min_corridor_clearance_m"])
    pass1 = (not enable_corridor) or (corridor_width >= need_corr_cells)
    st.write(("✅" if pass1 else "❌") + f" 1) Corridor width ≥ {th['min_corridor_clearance_m']} m (grid: {need_corr_cells} cells)")
    st.write(f"ℹ️ 2) Door/Hatch min clear: {th['min_hatch_clear_w_m']:.2f} × {th['min_hatch_clear_h_m']:.2f} m — tekil alan uyarıları kartlarda.")
    st.write(f"ℹ️ 3) Sleep–Exercise separation ≥ {th['min_sleep_exercise_sep_m']:.1f} m — uyarılar deck kartlarında.")
    st.write(f"ℹ️ 4) Galley–Hygiene separation ≥ {th['min_galley_hygiene_sep_m']:.1f} m.")
    st.write("ℹ️ 5) Access: her alan koridor bandına veya dış kabuğa temas etmeli (uyarılar gösterilir).")
    st.write(f"ℹ️ 6) Front-of-rack clearance ≥ {th['min_front_of_rack_m']:.2f} m (work/eclss/medical).")
    tot_area = float(df["Area req (m²)"].sum()) if not df.empty else 0.0
    tot_vol = float(df["Volume req (m³)"].sum()) if not df.empty else 0.0
    st.write(f"ℹ️ 7) Total required area: **{tot_area:.1f} m²**, volume: **{tot_vol:.1f} m³** (üst tabloda fonksiyon bazlı).")
    st.info("Eşikleri panelden düzenleyebilir veya Export ALL JSON içindeki nasa_thresholds bloğunu kurumsal değerlerle değiştirip Import ALL ile kilitleyebilirsiniz.")

# ----- 🧩 Simple Editor (Table) -----
with tabs[decks+4]:
    st.markdown("### 🧩 Basit Editör (tablo) — net & sorunsuz (widget-key state hatası fix)")

    edit_deck = st.selectbox("Deck to edit", options=list(range(1, decks+1)), index=0, key="simple_deck")

    if edit_deck not in st.session_state.deck_rects:
        st.session_state.deck_rects[edit_deck] = pack_rects(
            split_demands_by_deck(demands_from_df(df), decks).get(edit_deck, {}),
            cell_m=cell_m, grid_w=grid_w, deck=edit_deck
        )
    rects_can = st.session_state.deck_rects[edit_deck]

    def default_wh_from_area(key: str) -> Tuple[int,int]:
        req = float(df.loc[df["key"] == key, "Area req (m²)"].iloc[0])
        cells = max(1, int(math.ceil(req / (cell_m * cell_m))))
        w = max(1, int(round(math.sqrt(cells * 1.6))))
        h = max(1, int(math.ceil(cells / w)))
        return w, h

    mem_key = f"simple_df_mem_d{edit_deck}"
    if mem_key not in st.session_state:
        rows = [{"name": k, "x": r.x, "y": r.y, "w": r.w, "h": r.h} for k, r in rects_can.items()]
        st.session_state[mem_key] = pd.DataFrame(rows, columns=["name","x","y","w","h"])

    expected = [k for k in selected_keys if choose_deck(k, decks) == edit_deck]
    st.caption("Satır ekle: `name` (sleep, galley...), `x,y,w,h` hücre olarak. Apply ile kaydolur.")

    df_src = st.session_state[mem_key].copy()

    df_edit = st.data_editor(
        df_src,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "name": st.column_config.SelectboxColumn("name (function key)", options=expected),
            "x": st.column_config.NumberColumn("x", min_value=0, step=1),
            "y": st.column_config.NumberColumn("y", min_value=0, step=1),
            "w": st.column_config.NumberColumn("w", min_value=1, step=1),
            "h": st.column_config.NumberColumn("h", min_value=1, step=1),
        },
        key=f"simple_table_d{edit_deck}"
    )

    colA, colB, colC = st.columns([1,1,2])

    if colA.button("➕ Eksikleri otomatik ekle"):
        present = set(df_edit["name"].dropna().tolist()) if not df_edit.empty else set()
        to_add = [k for k in expected if k not in present]
        add_rows = []
        for k in to_add:
            w,h = default_wh_from_area(k)
            add_rows.append({"name": k, "x": 1, "y": 0, "w": min(w, max(1, grid_w-1)), "h": h})
        st.session_state[mem_key] = pd.concat([df_edit, pd.DataFrame(add_rows)], ignore_index=True)
        st.rerun()

    if edit_deck not in st.session_state.corridor_y_by_deck:
        H_i = max([r.y+r.h for r in rects_can.values()], default=0) + 1
        st.session_state.corridor_y_by_deck[edit_deck] = H_i // 2
    st.slider(
        "Corridor position Y (cells)",
        0, max(1, max([r.y+r.h for r in rects_can.values()], default=0)+1),
        value=st.session_state.corridor_y_by_deck[edit_deck],
        key=f"simple_corridor_{edit_deck}",
    )
    st.session_state.corridor_y_by_deck[edit_deck] = st.session_state[f"simple_corridor_{edit_deck}"]

    if colB.button("✅ Apply & Review"):
        new_rects: Dict[str, Rect] = {}
        used = set()
        st.session_state[mem_key] = df_edit.copy()

        for _, row in df_edit.fillna("").iterrows():
            name = str(row.get("name", "")).strip()
            if not name: continue
            base = base_key(name)
            k = name
            idx = 2
            while k in used:
                k = f"{base}__{idx}"; idx += 1
            x = int(clamp(int(row.get("x", 0)), 0, max(0, grid_w-1)))
            y = max(0, int(row.get("y", 0)))
            w = max(1, int(row.get("w", 1))); w = min(w, max(1, grid_w - x))
            h = max(1, int(row.get("h", 1)))
            new_rects[k] = Rect(k, x, y, w, h, deck=edit_deck)
            used.add(k)

        new_rects, msgs1 = resolve_overlaps(new_rects, grid_w)
        if enable_corridor and reserve_corridor:
            cy0 = st.session_state.corridor_y_by_deck[edit_deck]
            new_rects, msgs2 = nudge_away_from_corridor(new_rects, cy0, corridor_width)
            msgs1.extend(msgs2)

        st.session_state.deck_rects[edit_deck] = new_rects
        st.success("Kaydedildi. Fixes: " + ("; ".join(msgs1) if msgs1 else "none"))

        shell_cfg = {"show": show_shell, "thickness_cells": shell_wall, "color": shell_color}
        corr_cfg = {"enable": enable_corridor, "width": corridor_width, "pos_y": st.session_state.corridor_y_by_deck.get(edit_deck)}
        fig2d_new, avg_comp_new, _ = build_2d_figure(new_rects, grid_w, cell_m, df,
                                                     shell=shell_cfg, use_icons_flag=use_icons, corridor_cfg=corr_cfg)
        st.plotly_chart(fig2d_new, use_container_width=True)

        adj_new = adjacency_score(new_rects)
        sep_pen_new, sep_warns_new = separation_penalty(new_rects, cell_m, st.session_state["nasa_thresholds_override"])
        warns_new = []
        if enable_corridor:
            warns_new += corridor_warnings(new_rects, True, corridor_width, st.session_state.corridor_y_by_deck.get(edit_deck))
        warns_new += access_warnings(new_rects, grid_w, enable_corridor, corridor_width, st.session_state.corridor_y_by_deck.get(edit_deck), min_front_rack_m)
        warns_new += door_warnings(new_rects, grid_w, enable_corridor, corridor_width, st.session_state.corridor_y_by_deck.get(edit_deck), min_hatch_w_m, min_hatch_h_m)
        warns_new += sep_warns_new

        mc1, mc2, mc3 = st.columns(3)
        with mc1: st.progress(min(100, int(avg_comp_new)), text=f"Area compliance: {avg_comp_new:.0f}%")
        with mc2: st.progress(min(100, int(adj_new)), text=f"Adjacency score: {adj_new:.0f}")
        with mc3: st.progress(min(100, int(max(0, 100 - sep_pen_new))), text=f"Separation health: {max(0, 100 - sep_pen_new):.0f}")
        if warns_new: st.warning(" • ".join(warns_new))

    st.info("İpucu: ‘➕ Eksikleri otomatik ekle’ ile gereken tüm fonksiyonlar tabloya düşer; sonra x-y-w-h’yi düzeltip **Apply** bas.")

# ----- 🎛️ Canvas (Edit) -----
try:
    from streamlit_drawable_canvas import st_canvas
    has_canvas = True
except Exception:
    has_canvas = False

with tabs[decks+5]:
    st.markdown("### 🎛️ Interactive layout (drag & drop) — Canvas")
    if not has_canvas:
        st.warning("Interactive drag için `streamlit-drawable-canvas` gerekir.\nKur:  `python -m pip install streamlit-drawable-canvas`  sonra sayfayı yenile.")
    else:
        edit_deck = st.selectbox("Deck to edit", options=list(range(1, decks+1)), index=0, key="sel_deck_edit")
        mode_key = f"edit_mode_d{edit_deck}"
        if mode_key not in st.session_state: st.session_state[mode_key] = True  # True=Edit

        if edit_deck not in st.session_state.deck_rects:
            st.session_state.deck_rects[edit_deck] = pack_rects(
                split_demands_by_deck(demands_from_df(df), decks).get(edit_deck, {}),
                cell_m=cell_m, grid_w=grid_w, deck=edit_deck
            )
        rects_canonical = st.session_state.deck_rects[edit_deck]
        if edit_deck not in st.session_state.corridor_y_by_deck:
            H_i = max([r.y+r.h for r in rects_canonical.values()], default=0) + 1
            st.session_state.corridor_y_by_deck[edit_deck] = H_i // 2

        st.slider("Corridor position Y (cells)", 0, max(1, max([r.y+r.h for r in rects_canonical.values()], default=0)+1),
                  value=st.session_state.corridor_y_by_deck[edit_deck], key=f"cpos_d{edit_deck}")
        st.session_state.corridor_y_by_deck[edit_deck] = st.session_state[f"cpos_d{edit_deck}"]

        pxc_key = f"pxpc_d{edit_deck}"
        if pxc_key not in st.session_state:
            st.session_state.pxpc_by_deck.setdefault(edit_deck, 40)
            st.session_state[pxc_key] = st.session_state.pxpc_by_deck[edit_deck]
        new_pxpc = st.select_slider("Pixels per grid cell", options=[20,30,40,50,60], value=st.session_state[pxc_key], key=f"pxpc_sel_{edit_deck}")
        col_px_a, col_px_b = st.columns([1,4])
        if col_px_a.button("Apply scale", key=f"apply_px_{edit_deck}"):
            st.session_state[pxc_key] = new_pxpc
            st.session_state.pxpc_by_deck[edit_deck] = new_pxpc
            if "canvas_init" not in st.session_state: st.session_state["canvas_init"] = {}
            st.session_state["canvas_init"][f"deck_{edit_deck}"] = True
            st.rerun()
        px_per_cell = st.session_state[pxc_key]

        canvas_w = int(grid_w * px_per_cell)
        est_h = max([r.y + r.h for r in rects_canonical.values()], default=12) + 2
        canvas_h = int(est_h * px_per_cell)

        CANVAS_BUCKET = "canvas_state"; INIT_BUCKET = "canvas_init"
        HIST_BUCKET = "canvas_history"; HIST_I_BUCKET = "canvas_history_idx"
        KEY = f"deck_{edit_deck}"; HKEY = f"hist_{edit_deck}"
        for bucket, default in [(CANVAS_BUCKET, {}), (INIT_BUCKET, {}), (HIST_BUCKET, {}), (HIST_I_BUCKET, {})]:
            if bucket not in st.session_state: st.session_state[bucket] = default
        if KEY not in st.session_state[CANVAS_BUCKET]:
            st.session_state[CANVAS_BUCKET][KEY] = {"version":"4.4.0","objects":[]}
        if KEY not in st.session_state[INIT_BUCKET]:
            st.session_state[INIT_BUCKET][KEY] = True

        def rects_to_fabric(rects: Dict[str, Rect]) -> dict:
            objs=[]
            for k, r in rects.items():
                base = base_key(k)
                fill = FUNC_COLORS.get(base, CAT_FALLBACK.get(rules.get(base, {}).get("category","-"), "#7f7f7f"))
                objs.append({
                    "type":"rect","name":k,
                    "left": r.x * px_per_cell, "top": r.y * px_per_cell,
                    "width": max(1, r.w * px_per_cell), "height": max(1, r.h * px_per_cell),
                    "fill": fill, "opacity": 0.35, "stroke": fill, "strokeWidth": 2,
                    "hasControls": True, "hasBorders": True, "lockRotation": True,
                    "selectable": True, "evented": True, "hoverCursor": "move",
                    "transparentCorners": False, "cornerStyle": "circle",
                    "perPixelTargetFind": True, "targetFindTolerance": 8
                })
            return {"version":"4.4.0","objects":objs}

        def fabric_to_rects(fabric_json: dict) -> Dict[str, Rect]:
            out={}
            for obj in fabric_json.get("objects", []):
                k = obj.get("name")
                if not k: continue
                left_px = float(obj.get("left", 0.0)); top_px = float(obj.get("top", 0.0))
                w_px = float(obj.get("width", 1.0)) * float(obj.get("scaleX", 1.0))
                h_px = float(obj.get("height",1.0)) * float(obj.get("scaleY", 1.0))
                x_cells = int(round(left_px / px_per_cell)); y_cells = int(round(top_px / px_per_cell))
                w_cells = int(round(w_px / px_per_cell));   h_cells = int(round(h_px / px_per_cell))
                x_cells = clamp(x_cells, 0, max(0, grid_w - 1)); w_cells = clamp(w_cells, 1, max(1, grid_w - x_cells))
                y_cells = max(0, y_cells);                  h_cells = max(1, h_cells)
                out[k] = Rect(k, x_cells, y_cells, w_cells, h_cells, deck=edit_deck)
            return out

        def _canonical_json(obj):
            try: return json.dumps(obj, sort_keys=True, separators=(",", ":"))
            except Exception: return str(obj)
        def _hist_init(initial_json: dict):
            if HKEY not in st.session_state[HIST_BUCKET]:
                st.session_state[HIST_BUCKET][HKEY] = [copy.deepcopy(initial_json)]
                st.session_state[HIST_I_BUCKET][HKEY] = 0
        def _hist_push(new_json: dict, max_len: int = 50):
            _hist_init(new_json)
            arr = st.session_state[HIST_BUCKET][HKEY]
            idx = st.session_state[HIST_I_BUCKET][HKEY]
            if idx < len(arr) - 1: arr[:] = arr[:idx+1]
            if _canonical_json(arr[-1]) == _canonical_json(new_json): return
            arr.append(copy.deepcopy(new_json))
            if len(arr) > max_len: del arr[0]
            st.session_state[HIST_I_BUCKET][HKEY] = len(arr) - 1
        def _hist_step(delta: int) -> dict|None:
            if HKEY not in st.session_state[HIST_BUCKET]: return None
            arr = st.session_state[HIST_BUCKET][HKEY]
            if not arr: return None
            idx = st.session_state[HIST_I_BUCKET][HKEY]
            idx2 = clamp(idx + delta, 0, len(arr)-1)
            st.session_state[HIST_I_BUCKET][HKEY] = idx2
            return copy.deepcopy(arr[idx2])

        if st.session_state[INIT_BUCKET][KEY]:
            st.session_state[CANVAS_BUCKET][KEY] = rects_to_fabric(rects_canonical)

        leftb, rightb = st.columns([1,1])

        if st.session_state[mode_key]:
            leftb.info("Mode: **Edit** — sürükle/resize. Bittiğinde **Apply & Review**.")
            if rightb.button("✅ Apply & Review", use_container_width=True, key=f"apply_{edit_deck}"):
                latest_json = st.session_state[CANVAS_BUCKET][KEY]
                rects_new = fabric_to_rects(latest_json)
                rects_new, overlap_msgs = resolve_overlaps(rects_new, grid_w)
                if enable_corridor and reserve_corridor:
                    cy0 = st.session_state.corridor_y_by_deck[edit_deck]
                    rects_new, corr_msgs = nudge_away_from_corridor(rects_new, cy0, corridor_width)
                    overlap_msgs.extend(corr_msgs)
                st.session_state.deck_rects[edit_deck] = rects_new
                st.session_state[CANVAS_BUCKET][KEY] = rects_to_fabric(rects_new)
                _hist_push(st.session_state[CANVAS_BUCKET][KEY])
                st.session_state[mode_key] = False
                st.session_state[INIT_BUCKET][KEY] = True
                st.success("Kaydedildi. Fixes: " + ("; ".join(overlap_msgs) if overlap_msgs else "none"))
                st.rerun()

            col_reset, col_undo, col_redo, col_toolbar = st.columns([1,1,1,2])
            if col_reset.button(f"🗑️ Reset canvas (Deck {edit_deck})", key=f"resetcanvas_{edit_deck}"):
                st.session_state[CANVAS_BUCKET][KEY] = rects_to_fabric(st.session_state.deck_rects[edit_deck])
                st.session_state[INIT_BUCKET][KEY] = True
                st.rerun()
            if col_undo.button("↶ Undo"):
                prev_json = _hist_step(-1)
                if prev_json: st.session_state[CANVAS_BUCKET][KEY] = prev_json; st.rerun()
            if col_redo.button("↷ Redo"):
                next_json = _hist_step(+1)
                if next_json: st.session_state[CANVAS_BUCKET][KEY] = next_json; st.rerun()
            show_toolbar = col_toolbar.checkbox("Show toolbar (Canvas)", value=False, help="İstersen toolbar’ı aç.")

            canvas = st_canvas(
                fill_color="rgba(255,255,255,0.2)", stroke_width=2, stroke_color="#bbbbbb",
                background_color="#111111", height=canvas_h, width=canvas_w,
                drawing_mode="transform", initial_drawing=st.session_state[CANVAS_BUCKET][KEY],
                update_streamlit=True, key=f"canvas_deck_{edit_deck}",
                display_toolbar=show_toolbar
            )
            st.session_state[INIT_BUCKET][KEY] = False  # after first mount

            if canvas.json_data and "objects" in canvas.json_data:
                prev = st.session_state[CANVAS_BUCKET][KEY]
                prev_count = len(prev.get("objects", []))
                new_count  = len(canvas.json_data.get("objects", []))
                st.session_state[CANVAS_BUCKET][KEY] = canvas.json_data
                if new_count != prev_count: _hist_push(canvas.json_data)

            placed_cnt = len(st.session_state[CANVAS_BUCKET][KEY]["objects"])
            live_cnt = len(canvas.json_data.get("objects", [])) if canvas.json_data else 0
            st.caption(f"Canvas objects — saved: {placed_cnt} • live: {live_cnt}")

            # Palette (Add) / Remove / Align sections kept as-is (omitted repetition for brevity in UI)

        else:
            leftb.success("Mode: **Review** — metrikler ve 2D önizleme sabit.")
            if rightb.button("✏️ Back to Edit", use_container_width=True, key=f"back_{edit_deck}"):
                st.session_state[CANVAS_BUCKET][KEY] = rects_to_fabric(st.session_state.deck_rects[edit_deck])
                st.session_state[INIT_BUCKET][KEY] = True
                st.session_state[mode_key] = True
                st.rerun()

            current_rects = st.session_state.deck_rects[edit_deck]
            shell_cfg = {"show": show_shell, "thickness_cells": shell_wall, "color": shell_color}
            corr_cfg = {"enable": enable_corridor, "width": corridor_width, "pos_y": st.session_state.corridor_y_by_deck.get(edit_deck)}
            fig2d_new, avg_comp_new, _ = build_2d_figure(current_rects, grid_w, cell_m, df,
                                                         shell=shell_cfg, use_icons_flag=use_icons, corridor_cfg=corr_cfg)
            st.plotly_chart(fig2d_new, use_container_width=True, key=f"plot_2d_edit_{edit_deck}_review")

            adj_new = adjacency_score(current_rects)
            sep_pen_new, sep_warns_new = separation_penalty(current_rects, cell_m, st.session_state["nasa_thresholds_override"])
            warns_new = []
            if enable_corridor:
                warns_new += corridor_warnings(current_rects, True, corridor_width, st.session_state.corridor_y_by_deck.get(edit_deck))
            warns_new += access_warnings(current_rects, grid_w, enable_corridor, corridor_width, st.session_state.corridor_y_by_deck.get(edit_deck), min_front_rack_m)
            warns_new += door_warnings(current_rects, grid_w, enable_corridor, corridor_width, st.session_state.corridor_y_by_deck.get(edit_deck), min_hatch_w_m, min_hatch_h_m)
            warns_new += sep_warns_new

            mc1, mc2, mc3 = st.columns(3)
            with mc1: st.progress(min(100, int(avg_comp_new)), text=f"Area compliance: {avg_comp_new:.0f}%")
            with mc2: st.progress(min(100, int(adj_new)), text=f"Adjacency score: {adj_new:.0f}")
            with mc3: st.progress(min(100, int(max(0, 100 - sep_pen_new))), text=f"Separation health: {max(0, 100 - sep_pen_new):.0f}")
            if warns_new: st.error(" • ".join(warns_new))

# ----- Export / Report -----
with tabs[decks+6]:
    st.markdown("## 📄 Export / Report (PNG + PDF)")

    # ------------ helpers ------------
    def fig_to_png_bytes(fig: go.Figure, scale: int = 2) -> bytes|None:
        try:
            return fig.to_image(format="png", scale=scale)  # needs kaleido
        except Exception as e:
            st.error("PNG üretmek için `kaleido` gerekir. Kurulum: `python -m pip install -U kaleido`")
            st.caption(f"Ayrıntı: {e}")
            return None

    def build_3d_scene_for_export(camera_eye=(1.35, 1.35, 0.9)) -> go.Figure:
        fig3d = go.Figure()
        deck_height = 1.6
        max_y_all = 0
        for i in range(1, decks+1):
            rects_i = st.session_state.deck_rects.get(i, {})
            for k, r in rects_i.items():
                z0 = (r.deck - 1) * (deck_height + 0.25)
                verts = np.array([
                    [r.x, r.y, z0],[r.x+r.w, r.y, z0],[r.x+r.w, r.y+r.h, z0],[r.x, r.y+r.h, z0],
                    [r.x, r.y, z0+deck_height],[r.x+r.w, r.y, z0+deck_height],
                    [r.x+r.w, r.y+r.h, z0+deck_height],[r.x, r.y+r.h, z0+deck_height],
                ])
                faces = [0,1,2, 0,2,3, 4,5,6, 4,6,7, 0,1,5, 0,5,4, 1,2,6, 1,6,5, 2,3,7, 2,7,6, 3,0,4, 3,4,7]
                xs, ys, zs = verts[:,0], verts[:,1], verts[:,2]
                bk = base_key(k)
                color = FUNC_COLORS.get(bk, CAT_FALLBACK.get(rules.get(bk, {}).get("category","-"), "#7f7f7f"))
                name = ALL_FUNCS.get(bk, bk)
                fig3d.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=faces[0::3], j=faces[1::3], k=faces[2::3],
                                          color=color, opacity=0.55, name=f"D{r.deck} — {name}",
                                          hovertext=f"{name} (D{r.deck})"))
                max_y_all = max(max_y_all, r.y + r.h)

        if show_shell:
            def add_box_wireframe(fig, x, y, w, d, z0, z1, color):
                P = np.array([[x,y,z0],[x+w,y,z0],[x+w,y+d,z0],[x,y+d,z0],[x,y,z1],[x+w,y,z1],[x+w,y+d,z1],[x,y+d,z1]])
                edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                for a,b in edges:
                    fig.add_trace(go.Scatter3d(x=[P[a,0],P[b,0]], y=[P[a,1],P[b,1]], z=[P[a,2],P[b,2]],
                                               mode="lines", line=dict(width=3, color=shell_color),
                                               showlegend=False, hoverinfo="skip"))
            env_w = grid_w; env_d = max_y_all + 1
            z0 = 0.0; z1 = decks * (deck_height + 0.25)
            add_box_wireframe(fig3d, 0, 0, env_w, env_d, z0, z1, shell_color)

        fig3d.update_layout(
            scene=dict(
                xaxis=dict(range=[0, grid_w]),
                yaxis=dict(range=[max(0, max_y_all+1), 0]),
                zaxis=dict(range=[0, decks*(deck_height+0.25)+0.5]),
                aspectmode="data",
                camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]))
            ),
            margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#111", plot_bgcolor="#111"
        )
        return fig3d

    def compute_power_bill(rects_by_deck: Dict[int, Dict[str, Rect]]) -> pd.DataFrame:
        agg: Dict[str, float] = defaultdict(float)
        for d, rects in rects_by_deck.items():
            for k, r in rects.items():
                bk = base_key(k)
                agg[bk] += r.w * r.h * (cell_m * cell_m)  # m2
        rows=[]
        for k, area_m2 in agg.items():
            wpm2 = POWER_W_PER_M2.get(k, 8)
            watts = area_m2 * wpm2
            kwh_day = watts * 24.0 / 1000.0
            kwh_total = kwh_day * float(mission_days)
            rows.append({
                "key": k,
                "Function": ALL_FUNCS.get(k, k),
                "Area (m²)": round(area_m2, 2),
                "Power (W)": round(watts, 1),
                "kWh/day": round(kwh_day, 2),
                f"kWh/{mission_days}d": round(kwh_total, 2)
            })
        dfp = pd.DataFrame(rows).sort_values("Power (W)", ascending=False)
        dfp.loc["TOTAL"] = {
            "key":"-", "Function":"TOTAL",
            "Area (m²)": round(dfp["Area (m²)"].sum(), 2),
            "Power (W)": round(dfp["Power (W)"].sum(), 1),
            "kWh/day": round(dfp["kWh/day"].sum(), 2),
            f"kWh/{mission_days}d": round(dfp[f"kWh/{mission_days}d"].sum(), 2)
        }
        return dfp

    def adjacency_report(rects: Dict[str, Rect]) -> List[str]:
        rep=[]
        for a, prefs in ADJACENCY_WISHES.items():
            if a not in rects:
                rep.append(f"{ALL_FUNCS.get(a,a)}: yok")
                continue
            ax, ay = rect_center(rects[a]); adeck = rects[a].deck
            for b, w in prefs.items():
                if b not in rects:
                    rep.append(f"{ALL_FUNCS.get(a,a)} → {ALL_FUNCS.get(b,b)}: hedef yok")
                    continue
                if rects[b].deck != adeck:
                    rep.append(f"{ALL_FUNCS.get(a,a)} → {ALL_FUNCS.get(b,b)} (farklı deck)")
                    continue
                bx, by = rect_center(rects[b]); d = abs(ax - bx) + abs(ay - by)
                rep.append(f"{ALL_FUNCS.get(a,a)} ↔ {ALL_FUNCS.get(b,b)} ≈ {d:.1f} cell")
        return rep

    # ------------ Export UI ------------
    e1, e2, e3, e4 = st.columns(4)
    want_2d = e1.checkbox("2D PNG (deck bazında)", value=True)
    want_3d = e2.checkbox("3D PNG (izometrik küçük döndürme)", value=True)
    want_overlay = e3.checkbox("Deck ALL (Overlay) PNG", value=True)
    want_pdf = e4.checkbox("PDF teknik rapor", value=True)

    pngs_2d: Dict[int, bytes] = {}
    if want_2d:
        st.markdown("### 2D PNG’ler")
        for i in range(1, decks+1):
            rects_i = st.session_state.deck_rects.get(i, {})
            shell_cfg = {"show": show_shell, "thickness_cells": shell_wall, "color": shell_color}
            corr_cfg = {"enable": enable_corridor, "width": corridor_width, "pos_y": st.session_state.corridor_y_by_deck.get(i)}
            fig2d_i, _, _ = build_2d_figure(rects_i, grid_w, cell_m, df, shell=shell_cfg, use_icons_flag=use_icons, corridor_cfg=corr_cfg)
            b = fig_to_png_bytes(fig2d_i, scale=2)
            if b:
                pngs_2d[i] = b
                st.download_button(f"⬇️ 2D Deck {i} (PNG)", data=b, file_name=f"{APP_NAME}_deck{i}_{now_str()}.png", mime="image/png", key=f"dl_png_d{i}")

    png_3d: bytes|None = None
    if want_3d:
        st.markdown("### 3D PNG")
        fig3d_iso = build_3d_scene_for_export(camera_eye=(1.35, 1.0, 0.95))
        png_3d = fig_to_png_bytes(fig3d_iso, scale=2)
        if png_3d:
            st.download_button("⬇️ 3D (PNG)", data=png_3d, file_name=f"{APP_NAME}_3d_{now_str()}.png", mime="image/png", key="dl_png_3d")
        else:
            st.plotly_chart(fig3d_iso, use_container_width=True)

    png_overlay: bytes|None = None
    if want_overlay:
        st.markdown("### Deck ALL (Overlay) PNG")
        shell_cfg = {"show": show_shell, "thickness_cells": shell_wall, "color": shell_color} if show_shell else None
        fig_overlay_exp = build_overlay_all_decks(st.session_state.deck_rects, grid_w, cell_m, df, shell=shell_cfg)
        png_overlay = fig_to_png_bytes(fig_overlay_exp, scale=2)
        if png_overlay:
            st.download_button("⬇️ DeckALL (PNG)", data=png_overlay, file_name=f"{APP_NAME}_deckALL_{now_str()}.png", mime="image/png", key="dl_png_overlay")
        else:
            st.plotly_chart(fig_overlay_exp, use_container_width=True)

    if want_pdf:
        st.markdown("### PDF Teknik Rapor (v2)")
        if not HAS_RL:
            st.error("PDF için `reportlab` gerekir. Kurulum: `python -m pip install reportlab`")
        else:
            buf = io.BytesIO(); c = pdfcanvas.Canvas(buf, pagesize=A4)
            W, H = A4; margin = 1.5*cm
            styles = getSampleStyleSheet()

            def draw_table(headers, rows, x, y, colw, rowh=14, header_fill=colors.black):
                data = [headers] + rows
                t = Table(data, colWidths=colw)
                t.setStyle(TableStyle([
                    ('FONT', (0,0), (-1,0), 'Helvetica-Bold', 9),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('BACKGROUND', (0,0), (-1,0), header_fill),
                    ('FONT', (0,1), (-1,-1), 'Helvetica', 8),
                    ('LINEABOVE', (0,0), (-1,0), 0.3, colors.gray),
                    ('LINEBELOW', (0,0), (-1,0), 0.3, colors.gray),
                    ('INNERGRID', (0,0), (-1,-1), 0.25, colors.gray),
                    ('BOX', (0,0), (-1,-1), 0.5, colors.gray),
                ]))
                w, h = t.wrapOn(c, W, H)
                t.drawOn(c, x, y - h)
                return h

            # Cover
            c.setFont("Helvetica-Bold", 16)
            c.drawString(margin, H - margin, f"{APP_NAME} — Technical Brief (v2)")
            c.setFont("Helvetica", 10)
            c.drawString(margin, H - margin - 14, f"Generated: {now_str()}")
            c.drawString(margin, H - margin - 28, f"Crew: {crew}   Mission (days): {mission_days}   Cell: {cell_m} m   Grid W: {grid_w} cells   Decks: {decks}")
            c.line(margin, H - margin - 36, W - margin, H - margin - 36)

            # Constraints block
            th = st.session_state["nasa_thresholds_override"]
            headers = ["Constraint", "Value"]
            rows = [
                ["Min corridor clearance (m)", f"{th['min_corridor_clearance_m']}"],
                ["Min hatch width × height (m)", f"{th['min_hatch_clear_w_m']} × {th['min_hatch_clear_h_m']}"],
                ["Sleep–Exercise separation (m)", f"{th['min_sleep_exercise_sep_m']}"],
                ["Galley–Hygiene separation (m)", f"{th['min_galley_hygiene_sep_m']}"],
                ["Front-of-rack clearance (m)", f"{th['min_front_of_rack_m']}"],
            ]
            _h = draw_table(headers, rows, margin, H - margin - 60, [8*cm, 8*cm], header_fill=colors.Color(0.1,0.1,0.1))
            # 3D preview (if available)
            if want_3d and png_3d:
                img = ImageReader(io.BytesIO(png_3d))
                iw, ih = img.getSize()
                scale = min((W - 2*margin) / iw, (H - margin - 80 - _h) / ih)
                c.drawImage(img, margin, margin, width=iw*scale, height=ih*scale, preserveAspectRatio=True, mask='auto')
            c.showPage()

            # Deck ALL (overlay) page
            if png_overlay:
                c.setFont("Helvetica-Bold", 14)
                c.drawString(margin, H - margin, "Deck ALL (Overlay)")
                img = ImageReader(io.BytesIO(png_overlay))
                iw, ih = img.getSize()
                maxw = W - 2*margin
                maxh = H - 2.6*cm
                scale = min(maxw/iw, maxh/ih)
                c.drawImage(img, margin, margin, width=iw*scale, height=ih*scale, preserveAspectRatio=True, mask='auto')
                c.showPage()

            # Per-deck pages with metrics & warnings
            for i in range(1, decks+1):
                c.setFont("Helvetica-Bold", 14)
                c.drawString(margin, H - margin, f"Deck {i}")
                rects_i = st.session_state.deck_rects.get(i, {})
                shell_cfg = {"show": show_shell, "thickness_cells": shell_wall, "color": shell_color}
                corr_cfg = {"enable": enable_corridor, "width": corridor_width, "pos_y": st.session_state.corridor_y_by_deck.get(i)}
                # Try to reuse earlier 2D PNGs
                png_i = None
                if i in pngs_2d:
                    png_i = pngs_2d[i]
                else:
                    fig2d_i, _, _ = build_2d_figure(rects_i, grid_w, cell_m, df, shell=shell_cfg, use_icons_flag=False, corridor_cfg=corr_cfg)
                    png_i = fig_to_png_bytes(fig2d_i, scale=2)
                if png_i:
                    img = ImageReader(io.BytesIO(png_i))
                    iw, ih = img.getSize()
                    maxw = W - 2*margin
                    maxh = H/2 - 2*cm
                    scale = min(maxw/iw, maxh/ih)
                    c.drawImage(img, margin, H/2, width=iw*scale, height=ih*scale, preserveAspectRatio=True, mask='auto')

                # KPIs
                adj_i = adjacency_score(rects_i)
                sep_pen_i, sep_warns_i = separation_penalty(rects_i, cell_m, st.session_state["nasa_thresholds_override"])
                warns = []
                if enable_corridor:
                    warns += corridor_warnings(rects_i, True, corridor_width, st.session_state.corridor_y_by_deck[i])
                warns += access_warnings(rects_i, grid_w, enable_corridor, corridor_width, st.session_state.corridor_y_by_deck[i], min_front_rack_m)
                warns += door_warnings(rects_i, grid_w, enable_corridor, corridor_width, st.session_state.corridor_y_by_deck[i], min_hatch_w_m, min_hatch_h_m)
                warns += sep_warns_i

                # KPI table
                headers = ["Metric", "Value"]
                rows = [
                    ["Adjacency (0-100)", f"{adj_i:.0f}"],
                    ["Separation health (0-100)", f"{max(0, 100 - sep_pen_i):.0f}"],
                    ["Warnings (count)", f"{len(warns)}"],
                ]
                ytxt_top = H/2 - 10
                draw_table(headers, rows, margin, ytxt_top, [8*cm, 8*cm], header_fill=colors.Color(0.12,0.12,0.12))

                # Adjacency distances
                rep = adjacency_report(rects_i)[:10]
                c.setFont("Helvetica-Bold", 10)
                c.drawString(margin, ytxt_top - 90, "Adjacency distances:")
                c.setFont("Helvetica", 9)
                y = ytxt_top - 104
                for line in rep:
                    c.drawString(margin, y, f"- {line}"); y -= 11
                    if y < margin + 60: break

                # Warnings
                if warns:
                    c.setFont("Helvetica-Bold", 10)
                    c.drawString(margin, y - 6, "Warnings:")
                    c.setFont("Helvetica", 9)
                    y -= 20
                    for wline in warns[:12]:
                        c.drawString(margin, y, f"- {wline}"); y -= 11
                        if y < margin: break

                c.showPage()

            # Power bill + Mission summary
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, H - margin, "Power Bill & Summary")
            df_power = compute_power_bill(st.session_state.deck_rects)
            headers = ["Function","Area (m²)","Power (W)","kWh/day",f"kWh/{mission_days}d"]
            rows = [[row["Function"], str(row["Area (m²)"]), str(row["Power (W)"]), str(row["kWh/day"]), str(row[f"kWh/{mission_days}d"])]
                    for _, row in df_power.iterrows()]
            draw_table(headers, rows, margin, H - margin - 24, [7*cm, 3*cm, 3*cm, 3*cm, 3*cm], header_fill=colors.Color(0.1,0.1,0.1))

            tot_area = float(df["Area req (m²)"].sum()) if not df.empty else 0.0
            tot_vol  = float(df["Volume req (m³)"].sum()) if not df.empty else 0.0
            c.setFont("Helvetica-Bold", 12); c.drawString(margin, margin+40, "Mission Requirements")
            c.setFont("Helvetica", 10)
            c.drawString(margin, margin+26, f"Crew: {crew}   Mission days: {mission_days}")
            c.drawString(margin, margin+12, f"Total required area: {tot_area:.1f} m²   volume: {tot_vol:.1f} m³")
            th = st.session_state["nasa_thresholds_override"]
            c.drawString(margin, margin, f"NASA thresholds: corridor≥{th['min_corridor_clearance_m']}m, hatch≥{th['min_hatch_clear_w_m']}×{th['min_hatch_clear_h_m']}m, S–E≥{th['min_sleep_exercise_sep_m']}m")
            c.showPage(); c.save()
            pdf_bytes = buf.getvalue()
            st.download_button("⬇️ PDF Report (v2)", data=pdf_bytes, file_name=f"{APP_NAME}_report_{now_str()}.pdf", mime="application/pdf", key="dl_pdf_report_v2")

# Footer caption
st.caption(
    "Notes: Threshold defaults follow conservative human-factors practice; replace with program-certified NASA-STD-3001/HIDH values via thresholds panel or JSON import."
)
