"""
Emergency Vehicle Detection and Adaptive Signal Preemption
for Heterogeneous Traffic Conditions
=========================================================
[MOCK / DEMO MODE]
All detections are simulated.  No real model is loaded.
Replace mock_detect() with real YOLOv8 inference when the backend is ready.
"""

import streamlit as st
import random
import time
from datetime import datetime
from PIL import Image
import pandas as pd

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Emergency Vehicle Detection System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
#  GOOGLE FONTS
# ══════════════════════════════════════════════════════════
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════
#  GLOBAL CSS  — organised by section
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ═══════════════════════════════════════════════
   §0  CSS VARIABLES / DESIGN TOKENS
═══════════════════════════════════════════════ */
:root {
    /* palette */
    --bg:            #F4F6F8;
    --card:          #FFFFFF;
    --navy-dark:     #263445;
    --navy:          #334E68;
    --navy-light:    #4A6C8F;
    --ink:           #263445;
    --text-sec:      #5A6778;
    --text-dim:      #5E6D7E;
    --border:        #E4E9EF;
    --border-strong: #CBD5E0;

    /* semantic */
    --red:           #D64545;
    --red-dark:      #9B2C2C;
    --red-soft:      #FDF2F2;
    --red-border:    #F5C6C6;
    --green:         #2F9E6B;
    --green-dark:    #1E6B47;
    --green-soft:    #F0FAF5;
    --green-border:  #A8D5B8;
    --amber:         #E0A21B;
    --amber-dark:    #8A5E06;
    --amber-soft:    #FFFCEB;
    --amber-border:  #F0D090;

    /* signal lamps */
    --lamp-off:      #2C3A4A;
    --lamp-red:      #D64545;
    --lamp-amber:    #E0A21B;
    --lamp-green:    #2F9E6B;

    /* layered soft shadows */
    --shadow-xs:  0 1px 2px rgba(38,52,69,0.04), 0 2px 4px rgba(38,52,69,0.04);
    --shadow-sm:  0 1px 3px rgba(38,52,69,0.05), 0 4px 10px rgba(38,52,69,0.06);
    --shadow-md:  0 2px 6px rgba(38,52,69,0.06), 0 10px 22px rgba(38,52,69,0.09);
    --shadow-lg:  0 4px 10px rgba(38,52,69,0.07), 0 16px 32px rgba(38,52,69,0.12);

    /* radius (14-18px for cards & containers) */
    --r-btn:   10px;
    --r-panel: 14px;
    --r-card:  16px;
    --r-hero:  18px;

    /* spacing scale (8px base) */
    --sp-1: 4px;  --sp-2: 8px;  --sp-3: 12px;
    --sp-4: 16px; --sp-5: 20px; --sp-6: 24px;
    --sp-8: 32px;
}

/* ═══════════════════════════════════════════════
   §1  RESET & BASE
═══════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 14px;
    -webkit-font-smoothing: antialiased;
}
.stApp {
    background-color: var(--bg);
    color: var(--ink);
}
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] {
    visibility: hidden;
    display: none;
}
.block-container {
    padding: var(--sp-6) var(--sp-8) var(--sp-8) !important;
    max-width: 1440px;
}

/* thin neutral scrollbars */
* { scrollbar-width: thin; scrollbar-color: var(--border-strong) transparent; }
*::-webkit-scrollbar       { width: 6px; height: 6px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 99px; }
*::-webkit-scrollbar-thumb:hover { background: var(--text-sec); }

/* ═══════════════════════════════════════════════
   §2  ENTRANCE ANIMATIONS (once on load, no loops)
═══════════════════════════════════════════════ */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-14px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* ═══════════════════════════════════════════════
   §3  SIDEBAR
═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--card);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child {
    padding: var(--sp-6) var(--sp-5);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.15px !important;
    text-transform: uppercase !important;
    color: var(--navy) !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: var(--sp-2) !important;
    margin-top: var(--sp-5) !important;
    margin-bottom: var(--sp-3) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: var(--text-sec);
    font-size: 0.81rem;
    line-height: 1.65;
}
[data-testid="stSidebar"] .stMarkdown strong { color: var(--ink); }
[data-testid="stSidebar"] .stCaption p {
    font-size: 0.74rem;
    color: var(--text-dim);
    line-height: 1.55;
}
[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: var(--sp-4) 0 !important;
}
.sb-demo-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.75px;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 99px;
    background: var(--amber-soft);
    border: 1px solid var(--amber-border);
    color: var(--amber-dark);
}

/* ═══════════════════════════════════════════════
   §4  HERO HEADER BANNER
═══════════════════════════════════════════════ */
.hero-banner {
    background: linear-gradient(125deg, #263445 0%, #334E68 55%, #4A6C8F 100%);
    border-radius: var(--r-hero);
    padding: var(--sp-6) var(--sp-8);
    margin-bottom: var(--sp-6);
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-md);
    animation: fadeUp 0.45s ease both;
}
.hero-banner::before {
    content: '';
    position: absolute;
    right: -48px; top: -56px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    right: 72px; bottom: -78px;
    width: 150px; height: 150px;
    border-radius: 50%;
    background: rgba(255,255,255,0.035);
    pointer-events: none;
}
.hero-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--sp-4);
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}
.hero-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.25px;
    line-height: 1.25;
    margin: 0 0 var(--sp-1);
}
.hero-sub {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.8);
    margin: 0;
    letter-spacing: 0.1px;
}
.hero-pills {
    display: flex;
    gap: var(--sp-2);
    align-items: center;
    justify-content: flex-end;
    flex-wrap: wrap;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    padding: 5px 12px;
    border-radius: 99px;
    white-space: nowrap;
}
.pill-demo    { background: rgba(224,162,27,0.22); border: 1px solid rgba(255,217,122,0.6); color: #FFE39A; }
.pill-active  { background: rgba(47,158,107,0.24); border: 1px solid rgba(114,230,176,0.6); color: #A7F3D0; }
.pill-standby { background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.3); color: #E4E9EF; }
.pill-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: currentColor;
    flex-shrink: 0;
}

/* ═══════════════════════════════════════════════
   §5  SECTION OVERLINE LABELS
═══════════════════════════════════════════════ */
.sec-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.15px;
    text-transform: uppercase;
    color: var(--text-sec);
    border-bottom: 1px solid var(--border);
    padding-bottom: var(--sp-2);
    margin-bottom: var(--sp-3);
    margin-top: var(--sp-1);
}

/* ═══════════════════════════════════════════════
   §6  KPI CARDS
═══════════════════════════════════════════════ */
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--navy);
    border-radius: var(--r-card);
    padding: var(--sp-4) var(--sp-5);
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: fadeUp 0.45s ease both;
    position: relative;
    overflow: hidden;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-lg);
}
.kpi-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: var(--sp-3);
}
.kpi-chip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px; height: 34px;
    border-radius: var(--r-btn);
    background: rgba(51,78,104,0.09);
    color: var(--navy);
    font-size: 1rem;
}
.kpi-label {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.9px;
    text-transform: uppercase;
    color: var(--text-sec);
    margin-bottom: var(--sp-2);
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.05;
    letter-spacing: -0.8px;
}
.kpi-sub {
    font-size: 0.74rem;
    color: var(--text-dim);
    margin-top: var(--sp-1);
}
/* variants: normal / alert / ok */
.kpi-card.kpi-emerg { border-left-color: var(--red); }
.kpi-card.kpi-emerg .kpi-chip { background: rgba(214,69,69,0.12); color: var(--red-dark); }
.kpi-card.kpi-emerg .kpi-value { color: var(--red); }
.kpi-card.kpi-clear { border-left-color: var(--green); }
.kpi-card.kpi-clear .kpi-chip { background: rgba(47,158,107,0.12); color: var(--green-dark); }
.kpi-card.kpi-clear .kpi-value { color: var(--green); }

/* ═══════════════════════════════════════════════
   §7  UPLOAD ZONE & IMAGE PREVIEW
═══════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: var(--card);
    border: 1.5px dashed var(--border-strong);
    border-radius: var(--r-panel);
    padding: var(--sp-3);
    box-shadow: var(--shadow-xs);
    transition: border-color 0.2s ease, background-color 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--navy);
    background-color: #FAFCFE;
}
[data-testid="stFileUploader"] label {
    color: var(--text-sec) !important;
    font-size: 0.8rem;
}
[data-testid="stImage"] img {
    border-radius: var(--r-panel);
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
}
.upload-empty {
    background: var(--card);
    border: 1.5px dashed var(--border-strong);
    border-radius: var(--r-panel);
    padding: var(--sp-8) var(--sp-4);
    text-align: center;
    color: var(--text-sec);
    font-size: 0.82rem;
    box-shadow: var(--shadow-xs);
    transition: border-color 0.2s ease;
}
.upload-empty:hover { border-color: var(--navy); }

/* ═══════════════════════════════════════════════
   §8  SELECTBOX
═══════════════════════════════════════════════ */
[data-testid="stSelectbox"] > label {
    color: var(--text-sec) !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    border-radius: var(--r-btn) !important;
    border-color: var(--border-strong) !important;
    background-color: var(--card) !important;
}

/* ═══════════════════════════════════════════════
   §9  BUTTONS
═══════════════════════════════════════════════ */
div.stButton > button {
    border-radius: var(--r-btn);
    font-weight: 600;
    font-size: 0.84rem;
    letter-spacing: 0.15px;
    height: 40px;
    transition: background-color 0.15s ease, box-shadow 0.15s ease, transform 0.12s ease, border-color 0.15s ease;
    outline: none;
}
div.stButton > button:focus-visible {
    box-shadow: 0 0 0 3px rgba(51,78,104,0.32) !important;
}
div.stButton > button[kind="primary"] {
    background-color: var(--navy);
    border: 1px solid var(--navy);
    color: #FFFFFF;
}
div.stButton > button[kind="primary"]:hover:not(:disabled) {
    background-color: var(--navy-dark);
    border-color: var(--navy-dark);
    box-shadow: var(--shadow-sm);
    transform: translateY(-1px);
}
div.stButton > button[kind="primary"]:active:not(:disabled) {
    transform: translateY(0);
}
div.stButton > button[kind="secondary"] {
    background-color: var(--card);
    border: 1px solid var(--border-strong);
    color: var(--ink);
}
div.stButton > button[kind="secondary"]:hover:not(:disabled) {
    background-color: var(--bg);
    border-color: var(--navy);
    color: var(--navy-dark);
    transform: translateY(-1px);
}
div.stButton > button:disabled {
    background-color: #EDF1F5 !important;
    color: var(--text-dim) !important;
    border: 1px solid var(--border) !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* ═══════════════════════════════════════════════
   §10  EMERGENCY ALERT BANNER
═══════════════════════════════════════════════ */
.alert-panel {
    border-radius: var(--r-panel);
    padding: var(--sp-5) var(--sp-6);
    box-shadow: var(--shadow-xs);
}
.alert-panel.emergency {
    background: var(--red-soft);
    border: 1px solid var(--red-border);
    border-left: 5px solid var(--red);
    animation: slideInLeft 0.35s ease both;
}
.alert-panel.clear {
    background: var(--green-soft);
    border: 1px solid var(--green-border);
    border-left: 5px solid var(--green);
    animation: fadeUp 0.35s ease both;
}
.alert-panel.standby {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 5px solid var(--border-strong);
}
.alert-heading {
    font-size: 0.94rem;
    font-weight: 700;
    margin-bottom: var(--sp-3);
    display: flex;
    align-items: center;
    gap: var(--sp-2);
}
.alert-siren {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    border-radius: 8px;
    background: rgba(214,69,69,0.14);
    color: var(--red);
    flex-shrink: 0;
}
.alert-row   { display: flex; gap: var(--sp-6); flex-wrap: wrap; }
.alert-flabel {
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--text-sec);
    margin-bottom: 3px;
}
.alert-fvalue       { font-size: 0.9rem; font-weight: 700; color: var(--ink); }
.alert-fvalue.emerg { color: var(--red-dark); }

/* ═══════════════════════════════════════════════
   §11  TRAFFIC SIGNAL PANELS
═══════════════════════════════════════════════ */
.sig-panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r-card);
    padding: var(--sp-4) var(--sp-3);
    text-align: center;
    box-shadow: var(--shadow-sm);
    animation: fadeUp 0.4s ease both;
}
.sig-housing {
    background: #1E293B;
    border: 2px solid #2B3A4E;
    border-radius: var(--r-panel);
    width: 56px;
    margin: 0 auto var(--sp-3);
    padding: var(--sp-3) var(--sp-2);
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: center;
    box-shadow: inset 0 2px 5px rgba(0,0,0,0.35), 0 2px 6px rgba(38,52,69,0.15);
}
.sig-lamp {
    width: 30px; height: 30px;
    border-radius: 50%;
    background: var(--lamp-off);
    opacity: 0.28;
    transition: background-color 0.4s ease, box-shadow 0.4s ease, opacity 0.4s ease;
}
/* active lamps — soft restrained glow only */
.lamp-red-on   { background: var(--lamp-red);   opacity: 1; box-shadow: 0 0 10px rgba(214,69,69,0.55); }
.lamp-amber-on { background: var(--lamp-amber); opacity: 1; box-shadow: 0 0 10px rgba(224,162,27,0.55); }
.lamp-green-on { background: var(--lamp-green); opacity: 1; box-shadow: 0 0 10px rgba(47,158,107,0.55); }
.sig-state {
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.s-green { color: var(--green-dark); }
.s-red   { color: var(--red-dark); }
.s-amber { color: var(--amber-dark); }
.sig-dir-label {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.9px;
    text-transform: uppercase;
    color: var(--ink);
    margin-top: var(--sp-1);
}
.sig-lane {
    font-size: 0.68rem;
    color: var(--text-sec);
    margin-top: 2px;
}
.sig-note {
    display: inline-block;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: var(--text-sec);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 1px 7px;
    margin-top: 6px;
}

/* ═══════════════════════════════════════════════
   §12  DETECTION RESULTS TABLE CONTAINER
═══════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r-panel);
    box-shadow: var(--shadow-sm);
    padding: var(--sp-1);
    overflow: hidden;
}
[data-testid="stDataFrame"] iframe {
    border-radius: var(--r-btn) !important;
}

/* ═══════════════════════════════════════════════
   §13  EVENT LOG  — vertical timeline style
═══════════════════════════════════════════════ */
.elog {
    background: #FAFBFC;
    border: 1px solid var(--border);
    border-radius: var(--r-panel);
    padding: var(--sp-4) var(--sp-5);
    font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace;
    font-size: 0.74rem;
    max-height: 248px;
    overflow-y: auto;
    line-height: 1.75;
    color: var(--text-sec);
    box-shadow: var(--shadow-xs);
}
.elog-timeline {
    position: relative;
    padding-left: var(--sp-4);
    border-left: 2px solid var(--border);
}
.elog-row {
    position: relative;
    display: flex;
    align-items: flex-start;
    gap: var(--sp-2);
    padding: 4px 8px;
    margin-bottom: 4px;
    border-radius: 6px;
    transition: background-color 0.15s ease;
}
.elog-row:last-child { margin-bottom: 0; }
.elog-row:hover { background: rgba(51,78,104,0.06); }
.elog-dot {
    position: absolute;
    left: -21px;
    top: 9px;
    width: 9px; height: 9px;
    border-radius: 50%;
    border: 2px solid #FAFBFC;
    flex-shrink: 0;
}
.dot-info  { background: var(--navy); }
.dot-ok    { background: var(--green); }
.dot-warn  { background: var(--amber); }
.dot-err   { background: var(--red); }
.elog .ts   { color: var(--navy-dark); font-weight: 700; }
.elog .info { color: var(--text-sec); }
.elog .ok   { color: var(--green-dark); }
.elog .warn { color: var(--amber-dark); }
.elog .err  { color: var(--red-dark); font-weight: 600; }
.elog .empty { color: var(--text-sec); font-style: italic; }

/* ═══════════════════════════════════════════════
   §14  MISC STREAMLIT OVERRIDES & RESPONSIVE
═══════════════════════════════════════════════ */
[data-testid="stAlert"] {
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text-sec);
    border-radius: var(--r-panel);
    font-size: 0.82rem;
    box-shadow: var(--shadow-xs);
}
[data-testid="stMetricValue"] { color: var(--ink) !important; font-size: 1.1rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: var(--text-sec) !important; font-size: 0.72rem !important; }
[data-testid="stSpinner"] p { color: var(--navy) !important; font-size: 0.82rem !important; }
hr { border-color: var(--border) !important; margin: var(--sp-4) 0 !important; }
[data-testid="column"] { min-width: 0; }

@media (max-width: 768px) {
    .block-container { padding: var(--sp-4) var(--sp-4) var(--sp-6) !important; }
    .hero-banner { padding: var(--sp-4) var(--sp-5); }
    .hero-pills { justify-content: flex-start; }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════
VEHICLE_CATEGORIES = [
    "car", "motorcycle", "autorickshaw", "truck",
    "bus", "bicycle", "ambulance", "police_car",
]
EMERGENCY_TYPES = {"ambulance", "police_car"}
DIRECTIONS      = ["NORTH", "SOUTH", "EAST", "WEST"]

VEHICLE_LABELS = {
    "car":          "Car",
    "motorcycle":   "Motorcycle",
    "autorickshaw": "Autorickshaw",
    "truck":        "Truck",
    "bus":          "Bus",
    "bicycle":      "Bicycle",
    "ambulance":    "Ambulance",
    "police_car":   "Police Car",
}

# ══════════════════════════════════════════════════════════
#  SESSION STATE  — 8 keys, unchanged
# ══════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "detections":       None,
        "emergency_active": False,
        "emergency_type":   None,
        "emergency_conf":   0.0,
        "emergency_dir":    None,
        "signal_states":    {d: "RED" for d in DIRECTIONS},
        "event_log":        [],
        "run_count":        0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ══════════════════════════════════════════════════════════
#  MOCK DETECTION ENGINE  ← body unchanged
# ══════════════════════════════════════════════════════════
def mock_detect(camera_direction: str) -> dict:
    """
    [MOCK] Simulates object detection output.
    All values are randomly generated — not real model results.
    """
    detected = {}
    n = random.randint(2, 5)
    chosen = random.sample(VEHICLE_CATEGORIES, n)
    for cat in chosen:
        count = random.randint(1, 4) if cat not in EMERGENCY_TYPES else 1
        conf  = round(random.uniform(0.60, 0.99), 2)
        detected[cat] = {"count": count, "confidence": conf}

    # 30 % chance of emergency vehicle
    emergency_active = False
    emergency_type   = None
    emergency_conf   = 0.0
    if random.random() < 0.30:
        e_type = random.choice(list(EMERGENCY_TYPES))
        e_conf = round(random.uniform(0.80, 0.99), 2)
        detected[e_type]  = {"count": 1, "confidence": e_conf}
        emergency_active  = True
        emergency_type    = e_type
        emergency_conf    = e_conf

    return {
        "detected":         detected,
        "emergency_active": emergency_active,
        "emergency_type":   emergency_type,
        "emergency_conf":   emergency_conf,
        "camera_direction": camera_direction,
        "total_vehicles":   sum(v["count"] for v in detected.values()),
        "inference_ms":     random.randint(28, 95),
    }

# ══════════════════════════════════════════════════════════
#  SIGNAL ADAPTER  ← body unchanged
# ══════════════════════════════════════════════════════════
def compute_signals(emergency_active: bool, emergency_direction: str) -> dict:
    """
    [MOCK] Derives signal states from detection output.
    Emergency direction -> GREEN, all others -> RED.
    No emergency -> all GREEN (normal cycle placeholder).
    """
    if not emergency_active:
        return {d: "GREEN" for d in DIRECTIONS}
    return {d: ("GREEN" if d == emergency_direction else "RED") for d in DIRECTIONS}

# ══════════════════════════════════════════════════════════
#  HELPERS  ← signatures & bodies unchanged
# ══════════════════════════════════════════════════════════
def add_log(msg: str, level: str = "info"):
    """Append a timestamped entry to the session event log."""
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.event_log.append({"ts": ts, "msg": msg, "level": level})


def build_detection_df(detections: dict) -> pd.DataFrame:
    """Return a full 8-row DataFrame — one row per vehicle class."""
    rows = []
    for cat in VEHICLE_CATEGORIES:
        label = VEHICLE_LABELS.get(cat, cat.replace("_", " ").title())
        if cat in detections:
            info     = detections[cat]
            count    = info["count"]
            conf_pct = f"{info['confidence']:.0%}"
            status   = "Emergency" if cat in EMERGENCY_TYPES else "Normal"
        else:
            count    = 0
            conf_pct = "—"
            status   = "Not detected"
        rows.append({
            "Vehicle Class": label,
            "Count":         count,
            "Confidence":    conf_pct,
            "Status":        status,
        })
    return pd.DataFrame(rows)


# ── UI helpers (CSS class names updated; HTML structure improved) ──

def signal_widget_html(direction: str, state: str) -> str:
    """Render a single traffic-signal panel as HTML."""
    cfg = {
        "RED":    ("lamp-red-on",   "",             ""),
        "YELLOW": ("",              "lamp-amber-on",""),
        "GREEN":  ("",              "",             "lamp-green-on"),
    }
    lane_names = {
        "NORTH": "Northbound Lane",
        "SOUTH": "Southbound Lane",
        "EAST":  "Eastbound Lane",
        "WEST":  "Westbound Lane",
    }
    r_cls, a_cls, g_cls = cfg.get(state, ("", "", ""))
    state_color = {"RED": "s-red", "YELLOW": "s-amber", "GREEN": "s-green"}.get(state, "")
    lane = lane_names.get(direction, "Approach Lane")
    return f"""
    <div class="sig-panel">
        <div class="sig-housing">
            <div class="sig-lamp {r_cls}"></div>
            <div class="sig-lamp {a_cls}"></div>
            <div class="sig-lamp {g_cls}"></div>
        </div>
        <div class="sig-state {state_color}">{state}</div>
        <div class="sig-dir-label">{direction}</div>
        <div class="sig-lane">{lane}</div>
        <div class="sig-note">Simulated</div>
    </div>"""


def event_log_html(log: list) -> str:
    """Render the event log as a timeline-style scrollable block."""
    if not log:
        return (
            '<div class="elog">'
            '<div class="elog-row">'
            '<span class="empty">No events recorded yet.</span>'
            '</div></div>'
        )
    dot_map = {"info": "dot-info", "ok": "dot-ok", "warn": "dot-warn", "error": "dot-err"}
    cls_map  = {"info": "info",    "ok": "ok",     "warn": "warn",     "error": "err"}
    rows = ""
    for entry in reversed(log[-60:]):
        lvl     = entry["level"]
        dot_cls = dot_map.get(lvl, "dot-info")
        txt_cls = cls_map.get(lvl, "info")
        rows += (
            f'<div class="elog-row">'
            f'<span class="elog-dot {dot_cls}"></span>'
            f'<span><span class="ts">[{entry["ts"]}]</span> '
            f'<span class="{txt_cls}">{entry["msg"]}</span></span>'
            f'</div>'
        )
    return f'<div class="elog"><div class="elog-timeline">{rows}</div></div>'


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 4px">
        <div style="font-size:0.98rem;font-weight:700;color:#263445;line-height:1.3;letter-spacing:-0.15px">
            Emergency Vehicle<br>Detection System
        </div>
        <div style="font-size:0.75rem;color:#5A6778;margin-top:5px;line-height:1.5">
            Adaptive Traffic Signal Preemption
        </div>
        <div style="margin-top:12px">
            <span class="sb-demo-badge">Mock / Demo Mode</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Project")
    st.markdown("""
    **Title:** Emergency Vehicle Detection & Adaptive Signal Preemption for Heterogeneous Traffic Conditions

    **Mode:** Mock / Demo — no real model connected

    **Detector:** YOLOv8 *(not connected)*

    **Signal logic:** Simulated output only
    """)

    st.markdown("## Camera Directions")
    st.markdown("NORTH · SOUTH · EAST · WEST")

    st.markdown("## Vehicle Classes")
    for cat in VEHICLE_CATEGORIES:
        label = VEHICLE_LABELS.get(cat, cat)
        note  = "  —  *emergency*" if cat in EMERGENCY_TYPES else ""
        st.markdown(f"- {label}{note}")

    st.markdown("---")
    st.caption(
        "Designed for integration with a real YOLOv8 inference module "
        "and a hardware signal controller. Replace mock_detect() when ready."
    )

# ══════════════════════════════════════════════════════════
#  HERO HEADER BANNER
# ══════════════════════════════════════════════════════════
run_count = st.session_state.run_count
sys_pill  = (
    '<span class="pill pill-active"><span class="pill-dot"></span>Active</span>'
    if run_count > 0 else
    '<span class="pill pill-standby"><span class="pill-dot"></span>Standby</span>'
)

st.markdown(f"""
<div class="hero-banner">
    <div class="hero-inner">
        <div>
            <div class="hero-title">Emergency Vehicle Detection System</div>
            <div class="hero-sub">
                Adaptive Traffic Signal Preemption &nbsp;&middot;&nbsp;
                Heterogeneous Traffic Conditions
            </div>
        </div>
        <div class="hero-pills">
            <span class="pill pill-demo">Mock / Demo Mode</span>
            {sys_pill}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  KPI SUMMARY CARDS
# ══════════════════════════════════════════════════════════
d          = st.session_state.detections
total_v    = d["total_vehicles"] if d else 0
emerg_v    = 1 if st.session_state.emergency_active else 0
active_cam = st.session_state.emergency_dir or (d["camera_direction"] if d else "—")
det_status = (
    "Emergency" if st.session_state.emergency_active else
    ("Clear"    if d else "Idle")
)

kc1, kc2, kc3, kc4 = st.columns(4, gap="medium")

with kc1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-chip">&#128664;</div>
        <div class="kpi-label">Total Vehicles Detected</div>
        <div class="kpi-value">{total_v}</div>
        <div class="kpi-sub">this run</div>
    </div>""", unsafe_allow_html=True)

with kc2:
    cls2 = "kpi-emerg" if emerg_v else ("kpi-clear" if d else "")
    sub2 = "preemption active" if emerg_v else ("none detected" if d else "awaiting")
    chip2 = "&#128680;" if emerg_v else "&#9989;"
    st.markdown(f"""
    <div class="kpi-card {cls2}">
        <div class="kpi-chip">{chip2}</div>
        <div class="kpi-label">Emergency Vehicles</div>
        <div class="kpi-value">{emerg_v}</div>
        <div class="kpi-sub">{sub2}</div>
    </div>""", unsafe_allow_html=True)

with kc3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-chip">&#128247;</div>
        <div class="kpi-label">Active Camera</div>
        <div class="kpi-value" style="font-size:1.55rem;letter-spacing:0">{active_cam}</div>
        <div class="kpi-sub">direction</div>
    </div>""", unsafe_allow_html=True)

with kc4:
    cls4 = "kpi-emerg" if st.session_state.emergency_active else ("kpi-clear" if d else "")
    chip4 = "&#128308;" if st.session_state.emergency_active else ("&#128994;" if d else "&#9898;")
    st.markdown(f"""
    <div class="kpi-card {cls4}">
        <div class="kpi-chip">{chip4}</div>
        <div class="kpi-label">Detection Status</div>
        <div class="kpi-value" style="font-size:1.55rem;letter-spacing:0">{det_status}</div>
        <div class="kpi-sub">run #{st.session_state.run_count}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  MAIN COLUMNS  —  Input (left)  |  Results (right)
# ══════════════════════════════════════════════════════════
input_col, results_col = st.columns([1, 2], gap="large")

# ──────────────────────────────────────────────
#  LEFT  —  Input & Controls
# ──────────────────────────────────────────────
with input_col:
    st.markdown('<div class="sec-label">Image Input</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a camera frame (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Select a traffic-camera frame for detection.",
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(
            img,
            caption=f"{uploaded_file.name}  —  {img.size[0]} \u00d7 {img.size[1]} px",
            use_container_width=True,
        )
        add_log(
            f"Frame loaded: {uploaded_file.name} ({img.size[0]}x{img.size[1]})",
            "info",
        )
    else:
        st.markdown("""
        <div class="upload-empty">
            <div style="font-size:1.5rem;margin-bottom:8px;opacity:0.55">&#128443;</div>
            No frame loaded.<br>
            <span style="font-size:0.74rem;color:#5E6D7E">Upload a JPG, JPEG or PNG image above.</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Camera Direction</div>', unsafe_allow_html=True)

    camera_dir = st.selectbox(
        "Camera Direction",
        DIRECTIONS,
        index=0,
        help="Select the intersection approach being monitored.",
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    btn_run_col, btn_reset_col = st.columns(2, gap="small")
    with btn_run_col:
        run_btn = st.button(
            "Run Detection",
            use_container_width=True,
            type="primary",
            disabled=(uploaded_file is None),
        )
    with btn_reset_col:
        reset_btn = st.button(
            "Reset",
            use_container_width=True,
            type="secondary",
        )

    # Inference info (shown after a run)
    if st.session_state.detections:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Inference Info</div>', unsafe_allow_html=True)
        inf = st.session_state.detections
        mi1, mi2 = st.columns(2)
        with mi1:
            st.metric("Latency (mock)", f"{inf['inference_ms']} ms")
        with mi2:
            st.metric("Classes hit", len(inf["detected"]))

# ══════════════════════════════════════════════════════════
#  BUTTON HANDLERS
# ══════════════════════════════════════════════════════════
if reset_btn:
    st.session_state.detections       = None
    st.session_state.emergency_active = False
    st.session_state.emergency_type   = None
    st.session_state.emergency_conf   = 0.0
    st.session_state.emergency_dir    = None
    st.session_state.signal_states    = {d: "RED" for d in DIRECTIONS}
    st.session_state.run_count        = 0
    add_log("Dashboard reset by user.", "warn")
    st.rerun()

if run_btn and uploaded_file:
    with st.spinner("Running detection pipeline (mock mode)…"):
        time.sleep(0.6)
        result = mock_detect(camera_dir)

    st.session_state.detections       = result
    st.session_state.emergency_active = result["emergency_active"]
    st.session_state.emergency_type   = result["emergency_type"]
    st.session_state.emergency_conf   = result["emergency_conf"]
    st.session_state.emergency_dir    = camera_dir if result["emergency_active"] else None
    st.session_state.signal_states    = compute_signals(
        result["emergency_active"], camera_dir
    )
    st.session_state.run_count       += 1

    add_log(
        f"[Run #{st.session_state.run_count}] Camera: {camera_dir} | "
        f"Vehicles: {result['total_vehicles']} | "
        f"Latency: {result['inference_ms']} ms",
        "info",
    )
    if result["emergency_active"]:
        vtype = result["emergency_type"].replace("_", " ").title()
        add_log(
            f"EMERGENCY — {vtype} detected on {camera_dir} "
            f"(conf {result['emergency_conf']:.0%}) — preemption triggered",
            "error",
        )
    else:
        add_log("All clear — no emergency vehicle detected.", "ok")

    st.rerun()

# ──────────────────────────────────────────────
#  RIGHT  —  Results
# ──────────────────────────────────────────────
with results_col:

    # ── 1. Emergency Detection Status ────────
    st.markdown('<div class="sec-label">Emergency Detection Status</div>',
                unsafe_allow_html=True)

    if st.session_state.detections is None:
        st.markdown("""
        <div class="alert-panel standby">
            <div class="alert-heading" style="color:#5A6778">
                Awaiting Detection
            </div>
            <div style="font-size:0.82rem;color:#5E6D7E">
                Upload a camera frame and click <strong>Run Detection</strong> to begin monitoring.
            </div>
        </div>""", unsafe_allow_html=True)

    elif st.session_state.emergency_active:
        etype = (st.session_state.emergency_type or "").replace("_", " ").title()
        econf = st.session_state.emergency_conf
        edir  = st.session_state.emergency_dir or "—"
        st.markdown(f"""
        <div class="alert-panel emergency">
            <div class="alert-heading" style="color:#9B2C2C">
                <span class="alert-siren" aria-hidden="true">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M7 18v-6a5 5 0 1 1 10 0v6"/>
                        <path d="M5 21h14"/>
                        <path d="M12 2v2"/>
                        <path d="M4.2 5.2l1.4 1.4"/>
                        <path d="M19.8 5.2l-1.4 1.4"/>
                    </svg>
                </span>
                Emergency Vehicle Detected
            </div>
            <div class="alert-row">
                <div>
                    <div class="alert-flabel">Vehicle Type</div>
                    <div class="alert-fvalue emerg">{etype}</div>
                </div>
                <div>
                    <div class="alert-flabel">Confidence</div>
                    <div class="alert-fvalue emerg">{econf:.0%}</div>
                </div>
                <div>
                    <div class="alert-flabel">Camera Direction</div>
                    <div class="alert-fvalue emerg">{edir}</div>
                </div>
                <div>
                    <div class="alert-flabel">Preemption Status</div>
                    <div class="alert-fvalue emerg">Active</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="alert-panel clear">
            <div class="alert-heading" style="color:#1E6B47">
                &#10003; All Clear
            </div>
            <div style="font-size:0.82rem;color:#5A6778">
                No emergency vehicle detected. Normal signal cycle in effect.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── 2. Vehicle Analytics ──────────────────
    st.markdown('<div class="sec-label">Vehicle Analytics — All 8 Classes</div>',
                unsafe_allow_html=True)

    if st.session_state.detections and st.session_state.detections["detected"]:
        df = build_detection_df(st.session_state.detections["detected"])

        def style_zebra(row: pd.Series) -> list[str]:
            bg = "#FFFFFF" if row.name % 2 == 0 else "#F8FAFC"
            return [f"background-color: {bg}; color: #263445;" for _ in row]

        def style_status(val: str) -> str:
            if val == "Emergency":
                return (
                    "color: #9B2C2C; font-weight: 700; "
                    "background-color: #FEE2E2; border-radius: 99px;"
                )
            if val == "Normal":
                return (
                    "color: #1E6B47; font-weight: 600; "
                    "background-color: #DCFCE7; border-radius: 99px;"
                )
            return (
                "color: #8A5E06; font-weight: 500; "
                "background-color: #FEF3C7; border-radius: 99px;"
            )

        styled = (
            df.style
            .apply(style_zebra, axis=1)
            .map(style_status, subset=["Status"])
        )

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Vehicle Class": st.column_config.TextColumn(
                    "Vehicle Class", width="medium"
                ),
                "Count": st.column_config.NumberColumn(
                    "Count", width="small", format="%d"
                ),
                "Confidence": st.column_config.TextColumn(
                    "Confidence", width="small"
                ),
                "Status": st.column_config.TextColumn(
                    "Status", width="medium"
                ),
            },
        )
    else:
        st.info(
            "No detection data yet. "
            "Upload an image and click **Run Detection** to populate this table."
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── 3. Traffic Signal Monitoring ─────────
    st.markdown(
        '<div class="sec-label">Traffic Signal Monitoring — Simulated States</div>',
        unsafe_allow_html=True,
    )

    sig_cols = st.columns(4, gap="small")
    for i, direction in enumerate(DIRECTIONS):
        state = st.session_state.signal_states[direction]
        with sig_cols[i]:
            st.markdown(signal_widget_html(direction, state), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── 4. Event Log ──────────────────────────
    st.markdown('<div class="sec-label">Event Log — Recent Activity</div>',
                unsafe_allow_html=True)
    st.markdown(event_log_html(st.session_state.event_log), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════
st.markdown('<hr style="margin:1.6rem 0 0.8rem">', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-size:0.72rem;color:#5E6D7E;letter-spacing:0.2px">'
    'Mock / Demo Mode &nbsp;&middot;&nbsp; '
    'Emergency Vehicle Detection &amp; Adaptive Signal Preemption &nbsp;&middot;&nbsp; '
    'All detections are simulated. '
    'Replace <code style="font-size:0.7rem;color:#334E68">mock_detect()</code> '
    'with real YOLOv8 inference when the backend is ready.'
    '</div>',
    unsafe_allow_html=True,
)
