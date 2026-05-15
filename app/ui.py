# ui.py
# 使い方: from ui import metric_card, alert_banner, section_header, kpi_row
# ページ先頭で load_css() を呼んでください
#
# Streamlit 1.50 の sanitizer は inline style を剥がすため、
# 全てのカスタムコンポーネントは class ベース（スタイルは style.css に移譲）。

import json
import html as _html
import streamlit as st
import streamlit.components.v1 as components


# ─── CSS 注入 ──────────────────────────────────────────────────
def load_css(path: str = "style.css") -> None:
    """ページ先頭で一度だけ呼ぶ。親ドキュメントの <head> に <style> を注入する。"""
    with open(path, encoding="utf-8") as f:
        css = f.read()
    payload = json.dumps(css)
    components.html(f"""
    <script>
      (function() {{
        const css = {payload};
        const doc = window.parent.document;
        const id = 'jcd-design-css';
        let el = doc.getElementById(id);
        if (!el) {{
          el = doc.createElement('style');
          el.id = id;
          doc.head.appendChild(el);
        }}
        el.textContent = css;
      }})();
    </script>
    """, height=0)


# ─── 色バリアント（CSS class のサフィックスにマップ） ────────────
_COLOR_VARIANT = {
    "accent":  "accent",
    "warning": "warning",
    "success": "success",
    "danger":  "danger",
    "info":    "info",
}

def _variant_from_color_arg(arg: str) -> str:
    """後方互換: 旧 API は色コード(#F9A825 等)を渡していた。色→variant に変換。"""
    if not arg:
        return "accent"
    s = arg.lower()
    if s in _COLOR_VARIANT:
        return _COLOR_VARIANT[s]
    mapping = {
        "#f9a825": "warning",
        "#66bb6a": "success",
        "#ef5350": "danger",
        "#607d8b": "accent",
        "#90a4ae": "accent",
        "#b0bec5": "accent",
    }
    return mapping.get(s, "accent")


# ─── セクションヘッダー ────────────────────────────────────────
def section_header(title: str, sub: str = "") -> None:
    sub_html = f'<div class="jcd-section__sub">{_html.escape(sub)}</div>' if sub else ""
    st.markdown(
        f'<div class="jcd-section">'
        f'<div class="jcd-section__title">{_html.escape(title)}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─── メトリクスカード ─────────────────────────────────────────
def metric_card(
    label: str,
    value: str,
    delta: str = "",
    delta_up: bool = True,
    border_color: str = "accent",
) -> None:
    variant = _variant_from_color_arg(border_color)
    delta_html = ""
    if delta:
        dir_cls = "jcd-metric__delta--up" if delta_up else "jcd-metric__delta--down"
        arrow = "▲" if delta_up else "▼"
        delta_html = (
            f'<div class="jcd-metric__delta {dir_cls}">'
            f'<span class="jcd-metric__arrow">{arrow}</span>'
            f'<span>{_html.escape(delta)}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div class="jcd-metric jcd-metric--{variant}">'
        f'<div class="jcd-metric__label">{_html.escape(label)}</div>'
        f'<div class="jcd-metric__value">{_html.escape(value)}</div>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─── KPI 4列ショートカット ─────────────────────────────────────
def kpi_row(items: list[dict]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            metric_card(
                label        = item["label"],
                value        = item["value"],
                delta        = item.get("delta", ""),
                delta_up     = item.get("up", True),
                border_color = item.get("color", "accent"),
            )


# ─── アラートバナー ────────────────────────────────────────────
_ALERT_ICON = {
    "info": "ℹ️", "warning": "⚠️", "danger": "🚨", "success": "✅",
}

def page_header(title: str, sub: str = "", icon: str = "",
                show_period: bool = True, default_period: str = "6M",
                key_prefix: str = "pg",
                latest_date=None) -> dict:
    """全ページ共通のヘッダー: タイトル(左) + 期間セグメント/YoY/期間表示(右)。
    返り値: {"period": "1M|3M|6M|1Y", "yoy": "前月比|..."}
    """
    hcol_l, hcol_m, hcol_r = st.columns([3, 2, 2])
    with hcol_l:
        icon_html = f'<span class="jcd-page-title__icon">{icon}</span> ' if icon else ""
        sub_html = f'<div class="jcd-page-title__sub">{_html.escape(sub)}</div>' if sub else ""
        st.markdown(
            f'<div class="jcd-page-title">'
            f'<div class="jcd-page-title__main">{icon_html}{_html.escape(title)}</div>'
            f'{sub_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
    result = {"period": default_period, "yoy": "前月比"}
    if show_period:
        with hcol_m:
            result["period"] = st.radio(
                "期間", ["1M", "3M", "6M", "1Y"],
                index=["1M", "3M", "6M", "1Y"].index(default_period),
                horizontal=True, key=f"{key_prefix}_period",
                label_visibility="collapsed",
            )
        with hcol_r:
            rc1, rc2 = st.columns([1, 1])
            with rc1:
                result["yoy"] = st.selectbox(
                    "比較軸", ["前月比", "前年同月比 (YoY)", "比較なし"],
                    index=0, key=f"{key_prefix}_yoy",
                    label_visibility="collapsed",
                )
            with rc2:
                if latest_date is not None:
                    _months = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12}[result["period"]]
                    import pandas as _pd
                    _start = (latest_date - _pd.DateOffset(months=_months)).replace(day=1)
                    chip = f'{_start.strftime("%Y/%m")} ～ {latest_date.strftime("%Y/%m")}'
                else:
                    chip = ""
                st.markdown(
                    f'<div class="jcd-date-range">{_html.escape(chip)}</div>',
                    unsafe_allow_html=True,
                )
    return result


def date_range_chip(text: str) -> None:
    """ヘッダー右端に期間表示を描画。page_header 直後に呼ぶ。"""
    st.markdown(
        f'<div class="jcd-date-range jcd-date-range--floating">{_html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def alert_banner(message: str, level: str = "warning") -> None:
    icon = _ALERT_ICON.get(level, "ℹ️")
    # message は意図的に HTML を許容（<strong> 等）
    st.markdown(
        f'<div class="jcd-alert jcd-alert--{level}">'
        f'<span class="jcd-alert__icon">{icon}</span>'
        f'<span class="jcd-alert__msg">{message}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─── カード枠 ─────────────────────────────────────────────────
def card_begin(padding: str = "20px") -> None:
    st.markdown('<div class="jcd-card">', unsafe_allow_html=True)

def card_end() -> None:
    st.markdown('</div>', unsafe_allow_html=True)


# ─── ステータスバッジ ──────────────────────────────────────────
_BADGE_MAP = {
    "ok":     ("success", "正常"),
    "warn":   ("warning", "注意"),
    "danger": ("danger",  "緊急"),
}

def status_badge_html(status: str) -> str:
    variant, label = _BADGE_MAP.get(status, ("muted", status))
    return f'<span class="jcd-badge jcd-badge--{variant}">{_html.escape(label)}</span>'


# ─── Plotly レイアウトのマージ ────────────────────────────────
def apply_chart_theme(fig, title: str = "") -> None:
    from tokens import CHART_LAYOUT, TEXT_SUB
    fig.update_layout(**CHART_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color=TEXT_SUB, size=14), x=0))
