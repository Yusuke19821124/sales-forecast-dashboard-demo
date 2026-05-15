# tokens.py
# デザイントークン定数 — ui.py や各ページから import して使います

# ─── カラーパレット ───────────────────────────────────────────
BG           = "#0D1117"   # ページ背景
SURFACE      = "#161B22"   # カード背景
SURFACE2     = "#1C2128"   # 入れ子カード / テーブル行
BORDER       = "#21262D"   # 通常ボーダー
BORDER2      = "#30363D"   # 強調ボーダー

TEXT         = "#E6EDF3"   # 本文テキスト
TEXT_SUB     = "#C9D1D9"   # 補足テキスト
TEXT_MUTED   = "#8B949E"   # ラベル / キャプション

ACCENT       = "#607D8B"   # プライマリアクセント（スレートグレー）
ACCENT_L     = "#90A4AE"   # 明るいアクセント
ACCENT_LL    = "#B0BEC5"   # さらに明るいアクセント

WARNING      = "#F9A825"   # 注意（Amber）
WARNING_BG   = "rgba(249,168,37,0.12)"
SUCCESS      = "#66BB6A"   # 成功（Green）
SUCCESS_BG   = "rgba(102,187,106,0.12)"
DANGER       = "#EF5350"   # 緊急（Red）
DANGER_BG    = "rgba(239,83,80,0.12)"
INFO         = "#607D8B"   # 情報
INFO_BG      = "rgba(96,125,139,0.12)"

# チャート用カラーシーケンス（Plotly に渡す）
CHART_COLORS = [ACCENT_L, WARNING, SUCCESS, ACCENT, "#78909C", ACCENT_LL]

# ─── チャート共通レイアウト（Plotly） ────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(color=TEXT_MUTED, size=11),
    margin        = dict(l=0, r=0, t=30, b=0),
    legend        = dict(
        orientation = "h",
        yanchor     = "bottom",
        y           = 1.02,
        xanchor     = "right",
        x           = 1,
        font        = dict(size=11, color=TEXT_MUTED),
        bgcolor     = "rgba(0,0,0,0)",
    ),
    xaxis = dict(
        gridcolor   = BORDER,
        linecolor   = "rgba(0,0,0,0)",
        tickcolor   = "rgba(0,0,0,0)",
        tickfont    = dict(color=TEXT_MUTED, size=11),
    ),
    yaxis = dict(
        gridcolor   = BORDER,
        linecolor   = "rgba(0,0,0,0)",
        tickcolor   = "rgba(0,0,0,0)",
        tickfont    = dict(color=TEXT_MUTED, size=11),
    ),
    hoverlabel = dict(
        bgcolor     = SURFACE2,
        bordercolor = BORDER2,
        font        = dict(color=TEXT_SUB, size=12),
    ),
)

# ─── セグメント（個人 / 個人事業主 / 法人） ──────────────────
# 全画面で統一する3セグメントの色（Phase2 ドキュメント準拠）
SEGMENT_COLORS = {
    "個人":       "#64748B",  # グレー系
    "個人事業主": "#2D5FA6",  # ブルー系
    "法人":       "#1E3464",  # ネイビー
}
SEGMENT_ORDER = ["個人", "個人事業主", "法人"]

# 新EC ローンチ日（推定→実測の切替点）
import pandas as _pd
LAUNCH_DATE = _pd.Timestamp("2026-06-15")

# 現EC期の推定セグメント閾値（1注文あたり点数）
LEGACY_BUSINESS_THRESHOLD = 5


# ─── スペーシング ─────────────────────────────────────────────
RADIUS_SM = "6px"
RADIUS    = "8px"
RADIUS_LG = "12px"
