"""
販売予測モデルアプリケーション（デモ版）
起動: streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
import os
import sys
import re
import hmac


def _html_compact(s: str) -> str:
    """Markdown の 4スペース=コードブロック 誤認を避けるため、各行の先頭空白を除去。"""
    return re.sub(r"^[ \t]+", "", s, flags=re.MULTILINE)


# セグメント名 → CSS セーフなスラッグ
_SEG_SLUG = {"個人": "individual", "個人事業主": "sole", "法人": "corp"}
def _seg_slug(seg: str) -> str:
    return _SEG_SLUG.get(seg, "other")


# 営業管理: 法人顧客のデモ用ダミー社名（customer_id から決定論的に生成）
_CORP_PREFIX = ["株式会社", "有限会社", "合同会社"]
_CORP_STEM = [
    "山田建設", "田中工業", "佐藤製作所", "鈴木商事", "高橋電機", "渡辺ワークス",
    "伊藤産業", "中村製造", "小林組", "加藤工務店", "吉田技研", "山本テック",
    "松本建機", "井上機工", "木村興業", "林物産", "斎藤工芸", "清水設備",
    "森田ロジ", "池田テクノ", "橋本精工", "石川製鉄", "前田ハウス", "藤田運輸",
    "岡田工業", "長谷川商会", "後藤電設", "近藤建材", "坂本エンジ", "遠藤製作",
]
def corp_display_name(customer_id: str) -> str:
    """customer_id → ダミー法人名（デモ用、決定論的）"""
    h = sum(ord(c) for c in str(customer_id))
    return f"{_CORP_PREFIX[h % len(_CORP_PREFIX)]}{_CORP_STEM[h % len(_CORP_STEM)]}"

# ── 設定 ──
st.set_page_config(page_title="販売予測モデル", page_icon="📊", layout="wide")

# ── アクセス制御（Secrets["password"] による合言葉認証）──
def _check_password() -> bool:
    try:
        expected = st.secrets["password"]
    except Exception:
        st.error("⚠️ 認証パスワードが未設定です。Streamlit Cloud の App settings → Secrets で `password = \"...\"` を設定してください。")
        return False

    def _on_submit():
        if hmac.compare_digest(st.session_state.get("_pw_input", ""), str(expected)):
            st.session_state["_pw_ok"] = True
            st.session_state["_pw_input"] = ""
        else:
            st.session_state["_pw_ok"] = False

    if st.session_state.get("_pw_ok", False):
        return True

    st.text_input("パスワードを入力してください", type="password",
                  on_change=_on_submit, key="_pw_input")
    if st.session_state.get("_pw_ok") is False:
        st.error("パスワードが違います")
    return False

if not _check_password():
    st.stop()

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, "data")

# デザインシステム読み込み
sys.path.insert(0, APP_DIR)
from tokens import (
    ACCENT, ACCENT_L, ACCENT_LL, WARNING, SUCCESS, DANGER, INFO,
    TEXT, TEXT_SUB, TEXT_MUTED, SURFACE, BORDER,
    CHART_COLORS, CHART_LAYOUT,
    SEGMENT_COLORS, SEGMENT_ORDER, LAUNCH_DATE, LEGACY_BUSINESS_THRESHOLD,
)
from ui import (load_css, kpi_row, section_header, alert_banner,
                apply_chart_theme, page_header, date_range_chip)


def _page_date_range_text():
    """現在の期間ラベルに基づく `YYYY/MM ～ YYYY/MM` を返す。"""
    _latest = orders["order_date"].max()
    return f"{orders['order_date'].min().strftime('%Y/%m')} ～ {_latest.strftime('%Y/%m')}"

# CSS 注入
_css_path = os.path.join(APP_DIR, "style.css")
if os.path.exists(_css_path):
    load_css(_css_path)

# 後方互換: 旧 COLORS 参照を新トークンにマップ
COLORS = {
    "primary": ACCENT, "accent": ACCENT_L, "light": ACCENT_LL,
    "bg": SURFACE,
    "chart": CHART_COLORS,
}


# ============================================================
# データ読み込み
# ============================================================
@st.cache_data
def load_data():
    d = {}
    for name in ["product_master", "customers", "ec_orders", "wms_inventory",
                 "ga_behavior", "ga_sessions", "ad_performance", "campaigns",
                 "supply_chain_master", "vendors", "monthly_expenses", "ad_plan"]:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if os.path.exists(path):
            d[name] = pd.read_csv(path)
    d["ec_orders"]["order_date"] = pd.to_datetime(d["ec_orders"]["order_date"])
    d["wms_inventory"]["date"]   = pd.to_datetime(d["wms_inventory"]["date"])
    d["ga_behavior"]["date"]     = pd.to_datetime(d["ga_behavior"]["date"])
    if "ga_sessions" in d:
        d["ga_sessions"]["date"] = pd.to_datetime(d["ga_sessions"]["date"])
    d["ad_performance"]["date"]  = pd.to_datetime(d["ad_performance"]["date"])
    d["campaigns"]["start_date"] = pd.to_datetime(d["campaigns"]["start_date"])
    d["campaigns"]["end_date"]   = pd.to_datetime(d["campaigns"]["end_date"])
    return d

data = load_data()
products     = data["product_master"]
customers    = data["customers"]
orders       = data["ec_orders"]
inventory    = data["wms_inventory"].copy()
# 「マザー倉庫」→「EC在庫倉庫」へ名称統一（基幹倉庫→EC在庫倉庫→発送倉庫 のフロー）
inventory["warehouse_type"] = inventory["warehouse_type"].replace({"マザー倉庫": "EC在庫倉庫"})
inventory["warehouse_name"] = inventory["warehouse_name"].replace({"関東マザー倉庫": "関東EC在庫倉庫"})
ga           = data["ga_behavior"]
ads          = data["ad_performance"]
campaigns    = data["campaigns"]
supply_chain = data["supply_chain_master"]
vendors      = data["vendors"]
expenses     = data["monthly_expenses"]
ad_plan      = data.get("ad_plan")
ga_sessions  = data.get("ga_sessions")


# ============================================================
# UU単価 計算関数群（Phase2 KPI ドキュメント準拠）
# ============================================================
def estimate_segment_for_legacy(orders_df: pd.DataFrame,
                                threshold: int = LEGACY_BUSINESS_THRESHOLD) -> pd.DataFrame:
    """現EC期（新ECローンチ前）のセグメント推定。
    1注文あたり点数 >= threshold なら「事業者（推定）」、未満なら「個人（推定）」。
    戻り値: order_id -> estimated_segment のDataFrame
    """
    pts = orders_df.groupby("order_id")["quantity"].sum().rename("total_qty")
    seg = pts.apply(lambda q: "事業者(推定)" if q >= threshold else "個人(推定)")
    return seg.rename("estimated_segment").reset_index()


def calc_segment_uu_unit_price(ga_sessions_df: pd.DataFrame,
                               orders_df: pd.DataFrame,
                               customers_df: pd.DataFrame,
                               start, end) -> pd.DataFrame:
    """セグメント別 UU単価 を算出（案A: 会員ログイン後UUのみ）。
    戻り値 columns = [segment, uu, sales, uu_unit_price, sales_ratio, uu_ratio, orders, cvr, aov]
    """
    if ga_sessions_df is None:
        return pd.DataFrame()

    # ログイン済セッションのみ（匿名訪問は除外）
    s = ga_sessions_df[(ga_sessions_df["is_logged_in"]) &
                       (ga_sessions_df["date"] >= start) &
                       (ga_sessions_df["date"] <= end)].copy()
    # セグメント確定（ga_sessions の customer_type を信頼）
    uu_by_seg = s.groupby("customer_type")["customer_id"].nunique()

    # 売上 & 注文数（セグメント結合）
    o = orders_df[(orders_df["order_date"] >= start) &
                  (orders_df["order_date"] <= end) &
                  (orders_df["status"] == "完了")].merge(
        customers_df[["customer_id", "customer_type"]], on="customer_id", how="left"
    )
    sales_by_seg  = o.groupby("customer_type")["total_amount"].sum()
    orders_by_seg = o.groupby("customer_type")["order_id"].nunique()

    rows = []
    for seg in SEGMENT_ORDER:
        uu    = int(uu_by_seg.get(seg, 0))
        sales = float(sales_by_seg.get(seg, 0))
        ords  = int(orders_by_seg.get(seg, 0))
        rows.append({
            "segment": seg,
            "uu": uu,
            "sales": sales,
            "orders": ords,
            "uu_unit_price": (sales / uu) if uu else 0,
            "cvr": (ords / uu) if uu else 0,
            "aov": (sales / ords) if ords else 0,
        })
    df = pd.DataFrame(rows)
    total_sales = df["sales"].sum()
    total_uu    = df["uu"].sum()
    df["sales_ratio"] = df["sales"] / total_sales if total_sales else 0
    df["uu_ratio"]    = df["uu"]    / total_uu    if total_uu    else 0
    return df


def calc_overall_weighted_uu_unit_price(seg_df: pd.DataFrame) -> dict:
    """セグメント別テーブルから全体加重UU単価を算出。"""
    total_sales = float(seg_df["sales"].sum())
    total_uu    = int(seg_df["uu"].sum())
    return {
        "overall_uu_unit_price": (total_sales / total_uu) if total_uu else 0,
        "total_sales": total_sales,
        "total_uu":    total_uu,
        "note": "これはセグメント別UU単価の加重平均です",
    }


def calc_segment_uu_monthly(ga_sessions_df: pd.DataFrame,
                            orders_df: pd.DataFrame,
                            customers_df: pd.DataFrame) -> pd.DataFrame:
    """月次のセグメント別UU単価の時系列。columns=[month, segment, uu, sales, uu_unit_price]"""
    if ga_sessions_df is None:
        return pd.DataFrame()
    s = ga_sessions_df[ga_sessions_df["is_logged_in"]].copy()
    s["month"] = s["date"].dt.to_period("M").dt.to_timestamp()
    uu = s.groupby(["month", "customer_type"])["customer_id"].nunique().rename("uu")

    o = orders_df[orders_df["status"] == "完了"].merge(
        customers_df[["customer_id", "customer_type"]], on="customer_id", how="left"
    ).copy()
    o["month"] = o["order_date"].dt.to_period("M").dt.to_timestamp()
    sales = o.groupby(["month", "customer_type"])["total_amount"].sum().rename("sales")

    df = pd.concat([uu, sales], axis=1).fillna(0).reset_index()
    df = df[df["customer_type"].isin(SEGMENT_ORDER)]
    df["uu_unit_price"] = df.apply(
        lambda r: (r["sales"] / r["uu"]) if r["uu"] else 0, axis=1)
    df = df.rename(columns={"customer_type": "segment"})
    return df


# ============================================================
# サイドバー
# ============================================================
# ── サイドバー: ブランド ──
st.sidebar.markdown(
    '<div class="jcd-brand">'
    '<div class="jcd-brand__text">'
    '<div class="jcd-brand__title">EC Analytics</div>'
    '<div class="jcd-brand__sub">販売予測ダッシュボード</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

_PAGES = ["ダッシュボード", "収支管理", "広告効果分析",
          "キャンペーン分析", "需要予測", "在庫分析",
          "需要予測×在庫管理", "営業管理"]
# クエリパラメータで初期ページ指定可（キャプチャ用途）: ?p=営業管理 や ?p=2
_qp = st.query_params.get("p", None)
_default_idx = 0
if _qp:
    if _qp in _PAGES:
        _default_idx = _PAGES.index(_qp)
    elif _qp.isdigit() and 0 <= int(_qp) < len(_PAGES):
        _default_idx = int(_qp)
page = st.sidebar.radio(
    "メニュー", _PAGES, index=_default_idx,
    label_visibility="collapsed",
)

# ── サイドバー: フッター（データ期間） ──
_orders_min = orders["order_date"].min()
_orders_max = orders["order_date"].max()
st.sidebar.markdown(
    f'<div class="jcd-side-footer">'
    f'<div class="jcd-side-footer__row"><span class="jcd-side-footer__k">最終更新</span>'
    f'<span class="jcd-side-footer__v">{_orders_max.strftime("%Y/%m/%d")}</span></div>'
    f'<div class="jcd-side-footer__row"><span class="jcd-side-footer__k">データ取得範囲</span>'
    f'<span class="jcd-side-footer__v">{_orders_min.strftime("%Y/%m")} ～ {_orders_max.strftime("%Y/%m")}</span></div>'
    f'</div>',
    unsafe_allow_html=True,
)


# ============================================================
# 共通: 予測関数
# ============================================================
@st.cache_data
def _past_monthly_ad_spend():
    """過去の月次総広告費（全媒体合計）"""
    a = ads.copy()
    a["month"] = a["date"].dt.to_period("M").dt.to_timestamp()
    return a.groupby("month")["cost"].sum().to_dict()


def forecast_category_demand(future_ad_spend_map=None, horizon=12):
    """カテゴリ別月次需要予測。

    future_ad_spend_map: {pd.Timestamp(月初): 広告費} の辞書（未来月のみ）
      None の場合は過去平均を未来月に流用（= 現状維持プラン相当）
    horizon: 予測する先の月数（デフォルト12）
    """
    completed = orders[orders["status"] == "完了"]
    past_spend = _past_monthly_ad_spend()
    avg_past_spend = float(np.mean(list(past_spend.values()))) if past_spend else 0.0
    results = {}
    for cat in products["category"].unique():
        cat_skus = products[products["category"] == cat]["sku_id"].tolist()
        cat_orders = completed[completed["sku_id"].isin(cat_skus)]
        monthly = (cat_orders
                   .assign(month_dt=cat_orders["order_date"].dt.to_period("M").dt.to_timestamp())
                   .groupby("month_dt")["quantity"].sum()
                   .reset_index().rename(columns={"month_dt": "date", "quantity": "qty"}))
        if len(monthly) < 6:
            continue
        monthly["month_num"] = np.arange(len(monthly))
        monthly["month_of_year"] = monthly["date"].dt.month
        for m in range(1, 13):
            monthly[f"m_{m}"] = (monthly["month_of_year"] == m).astype(int)
        # 広告費特徴量（月次総広告費）
        monthly["ad_spend"] = monthly["date"].map(lambda d: past_spend.get(d, avg_past_spend))
        feat = ["month_num", "ad_spend"] + [f"m_{m}" for m in range(1, 13)]
        mdl = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        mdl.fit(monthly[feat].values, monthly["qty"].values)
        last_num = monthly["month_num"].iloc[-1]
        last_date = monthly["date"].iloc[-1]
        future = []
        for i in range(1, horizon + 1):
            fd = last_date + pd.DateOffset(months=i)
            # 未来月の広告費: プラン があればそれ、なければ過去平均
            ad = (future_ad_spend_map or {}).get(fd, avg_past_spend)
            row = {"date": fd, "month_num": last_num + i, "ad_spend": ad}
            for m in range(1, 13):
                row[f"m_{m}"] = 1 if fd.month == m else 0
            future.append(row)
        fdf = pd.DataFrame(future)
        fdf["qty"] = mdl.predict(fdf[feat].values).clip(0).astype(int)
        results[cat] = {"history": monthly, "forecast": fdf}
    return results


# ============================================================
# 📊 ダッシュボード
# ============================================================
if page == "ダッシュボード":
    hdr = page_header("ダッシュボード", "全体の売上状況をひと目で把握できます",
                      default_period="6M", key_prefix="dash",
                      latest_date=orders["order_date"].max())
    period_label = hdr["period"]
    yoy_mode = hdr["yoy"]

    completed = orders[orders["status"] == "完了"]
    _period_months = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12}[period_label]
    _latest = completed["order_date"].max()
    _period_start = (_latest - pd.DateOffset(months=_period_months)).replace(day=1)
    completed = completed[completed["order_date"] >= _period_start]

    # ── KPI 計算 ──
    total_rev = completed["total_amount"].sum()
    total_orders = completed["order_id"].nunique()
    avg_order_value = total_rev / total_orders if total_orders else 0
    unique_customers = completed["customer_id"].nunique()

    # 比較期間の集計
    latest_date = completed["order_date"].max()
    def _period_stats(df):
        return {
            "rev": df["total_amount"].sum(),
            "orders": df["order_id"].nunique(),
            "aov": (df["total_amount"].sum() / df["order_id"].nunique()) if df["order_id"].nunique() else 0,
            "cust": df["customer_id"].nunique(),
        }

    def _pct(cur, prev):
        if prev == 0 or prev is None:
            return None
        return (cur - prev) / prev * 100

    # 当月・前月・前年同月
    cur_mstart = latest_date.replace(day=1)
    cur = _period_stats(completed[completed["order_date"] >= cur_mstart])
    prev_m_end = cur_mstart - pd.Timedelta(days=1)
    prev_m_start = prev_m_end.replace(day=1)
    prev_m = _period_stats(completed[(completed["order_date"] >= prev_m_start) &
                                      (completed["order_date"] <= prev_m_end)])
    prev_y_start = cur_mstart - pd.DateOffset(years=1)
    prev_y_end = prev_y_start + (prev_m_end - prev_m_start)
    prev_y = _period_stats(completed[(completed["order_date"] >= prev_y_start) &
                                      (completed["order_date"] <= prev_y_end)])

    if yoy_mode == "前年同月比 (YoY)":
        base = prev_y
        label_suffix = "YoY"
    elif yoy_mode == "前月比":
        base = prev_m
        label_suffix = "前月比"
    else:
        base = None
        label_suffix = ""

    def _delta_str(cur_v, base_v):
        if base is None or base_v in (0, None):
            return "", True
        pct = _pct(cur_v, base_v)
        if pct is None:
            return "", True
        return f"{pct:+.1f}% {label_suffix}", pct >= 0

    d1, u1 = _delta_str(cur["rev"],    base["rev"]    if base else 0)
    d2, u2 = _delta_str(cur["orders"], base["orders"] if base else 0)
    d3, u3 = _delta_str(cur["aov"],    base["aov"]    if base else 0)
    d4, u4 = _delta_str(cur["cust"],   base["cust"]   if base else 0)

    kpi_row([
        {"label": "累計売上",       "value": f"¥{total_rev:,.0f}",       "delta": d1, "up": u1, "color": "accent"},
        {"label": "累計注文数",     "value": f"{total_orders:,}",        "delta": d2, "up": u2, "color": "success"},
        {"label": "平均注文単価",   "value": f"¥{avg_order_value:,.0f}", "delta": d3, "up": u3, "color": "warning"},
        {"label": "購入UU",         "value": f"{unique_customers:,}",    "delta": d4, "up": u4, "color": "accent"},
    ])

    # ============================================================
    # 📊 UU単価ブロック（セグメント別 + 全体加重）
    # ============================================================
    seg_uu = calc_segment_uu_unit_price(ga_sessions, orders, customers,
                                        _period_start, _latest)
    overall_uu = calc_overall_weighted_uu_unit_price(seg_uu) if len(seg_uu) else None

    if overall_uu and overall_uu["total_uu"] > 0:
        # 3セグメントカード HTML
        seg_cards_html = ""
        for _, r in seg_uu.iterrows():
            seg = r["segment"]
            seg_cards_html += f"""
            <div class="jcd-uu-seg jcd-uu-seg--{_seg_slug(seg)}">
              <div class="jcd-uu-seg__label">{seg}</div>
              <div class="jcd-uu-seg__value">¥{r['uu_unit_price']:,.0f}</div>
              <div class="jcd-uu-seg__meta">
                <span>UU {int(r['uu']):,}</span>
                <span>売上比 {r['sales_ratio']*100:.0f}%</span>
              </div>
            </div>
            """

        uu_block_html = f"""
        <div class="jcd-card jcd-uu-block">
          <div class="jcd-card__title">📊 UU単価（訪問者1人あたり売上）</div>
          <div class="jcd-card__sub">会員ログイン後UUで算出 ／ 匿名訪問は除外</div>
          <div class="jcd-uu-grid">
            <div class="jcd-uu-overall">
              <div class="jcd-uu-overall__label">全体UU単価</div>
              <div class="jcd-uu-overall__value">¥{overall_uu['overall_uu_unit_price']:,.0f}</div>
              <div class="jcd-uu-overall__note">ℹ️ セグメント別UU単価の加重平均です</div>
              <div class="jcd-uu-overall__meta">
                総UU {overall_uu['total_uu']:,} / 売上 ¥{overall_uu['total_sales']/1e6:.1f}M
              </div>
            </div>
            <div class="jcd-uu-seg-row">
              {seg_cards_html}
            </div>
          </div>
        </div>
        """
        st.markdown(_html_compact(uu_block_html), unsafe_allow_html=True)

        # ── セグメント別UU単価 月次推移 ──
        monthly_uu = calc_segment_uu_monthly(ga_sessions, orders, customers)
        if len(monthly_uu):
            fig_uu = go.Figure()
            for seg in SEGMENT_ORDER:
                d = monthly_uu[monthly_uu["segment"] == seg].sort_values("month")
                if d.empty:
                    continue
                fig_uu.add_trace(go.Scatter(
                    x=d["month"], y=d["uu_unit_price"],
                    name=seg, mode="lines+markers",
                    line=dict(color=SEGMENT_COLORS[seg], width=2.2),
                ))
            # 新ECローンチ境界線
            launch_str = LAUNCH_DATE.strftime("%Y-%m-%d")
            fig_uu.add_shape(type="line", x0=launch_str, x1=launch_str,
                             y0=0, y1=1, yref="paper",
                             line=dict(dash="dash", color="#E57373", width=1.6))
            fig_uu.add_annotation(x=launch_str, y=1.04, yref="paper",
                                  text="新ECローンチ（推定→実測）",
                                  showarrow=False,
                                  font=dict(color="#E57373", size=11))
            apply_chart_theme(fig_uu)
            fig_uu.update_layout(height=300, yaxis_tickprefix="¥",
                                 yaxis_tickformat=",.0f",
                                 title=dict(text="セグメント別 UU単価 月次推移",
                                            font=dict(color=TEXT_SUB, size=13), x=0))
            st.plotly_chart(fig_uu, use_container_width=True)

        with st.expander("▶ UU単価の分解（CVR × AOV）"):
            st.markdown(
                "**UU単価 = CVR × AOV** に分解することで、変動要因（訪問→購入の転換か、単価か）を切り分けられます。"
            )
            tbl = seg_uu[["segment", "uu", "orders", "sales",
                          "uu_unit_price", "cvr", "aov"]].copy()
            tbl["uu"]            = tbl["uu"].map(lambda v: f"{v:,}")
            tbl["orders"]        = tbl["orders"].map(lambda v: f"{v:,}")
            tbl["sales"]         = tbl["sales"].map(lambda v: f"¥{v:,.0f}")
            tbl["uu_unit_price"] = tbl["uu_unit_price"].map(lambda v: f"¥{v:,.0f}")
            tbl["cvr"]           = tbl["cvr"].map(lambda v: f"{v*100:.2f}%")
            tbl["aov"]           = tbl["aov"].map(lambda v: f"¥{v:,.0f}")
            tbl.columns = ["セグメント", "UU数", "注文数", "売上",
                           "UU単価", "CVR", "AOV"]
            st.dataframe(tbl, hide_index=True, use_container_width=True)

    st.markdown("---")

    # ── 月次売上推移 + 新規/リピートドーナツ（横並び） ──
    completed_with_cust = completed.copy()
    completed_with_cust["month"] = completed_with_cust["order_date"].dt.to_period("M").astype(str)

    # 各顧客の初回購入月を特定
    first_purchase = (completed_with_cust
                      .groupby("customer_id")["order_date"]
                      .min().dt.to_period("M").astype(str)
                      .reset_index().rename(columns={"order_date": "first_month"}))
    completed_with_cust = completed_with_cust.merge(first_purchase, on="customer_id", how="left")
    completed_with_cust["顧客区分"] = np.where(
        completed_with_cust["month"] == completed_with_cust["first_month"], "新規", "リピート")

    ccol_l, ccol_r = st.columns([3, 1])

    with ccol_l:
        section_header("売上トレンド（月次）", "総売上 / 新規 / リピートの推移")
        monthly_nr = (completed_with_cust
                      .groupby(["month", "顧客区分"])["total_amount"].sum()
                      .reset_index().rename(columns={"total_amount": "売上"}))
        monthly_total = (completed_with_cust.groupby("month")["total_amount"].sum()
                         .reset_index().rename(columns={"total_amount": "総売上"}))
        new_series = monthly_nr[monthly_nr["顧客区分"] == "新規"]
        rep_series = monthly_nr[monthly_nr["顧客区分"] == "リピート"]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly_total["month"], y=monthly_total["総売上"],
            name="総売上", mode="lines+markers",
            line=dict(color=ACCENT_LL, width=2.5),
            fill="tozeroy", fillcolor="rgba(176,190,197,0.08)",
        ))
        fig_trend.add_trace(go.Scatter(
            x=new_series["month"], y=new_series["売上"],
            name="新規", mode="lines+markers",
            line=dict(color=WARNING, width=1.8),
        ))
        fig_trend.add_trace(go.Scatter(
            x=rep_series["month"], y=rep_series["売上"],
            name="リピート", mode="lines+markers",
            line=dict(color=SUCCESS, width=1.8),
        ))
        apply_chart_theme(fig_trend)
        fig_trend.update_layout(height=360, yaxis_tickprefix="¥", yaxis_tickformat=",.0f")
        st.plotly_chart(fig_trend, use_container_width=True)

    with ccol_r:
        section_header("新規 vs リピート", "累計期間の構成比")
        _nr_total = (completed_with_cust.groupby("顧客区分")["total_amount"].sum()
                     .reindex(["新規", "リピート"]).fillna(0))
        _new_amt = float(_nr_total.get("新規", 0))
        _rep_amt = float(_nr_total.get("リピート", 0))
        _sum_amt = _new_amt + _rep_amt
        _new_ratio = (_new_amt / _sum_amt * 100) if _sum_amt else 0
        fig_donut = go.Figure(go.Pie(
            labels=["新規", "リピート"], values=[_new_amt, _rep_amt],
            hole=0.65, sort=False,
            marker=dict(colors=[WARNING, SUCCESS], line=dict(color=SURFACE, width=2)),
            textinfo="none", hovertemplate="%{label}: ¥%{value:,.0f}<extra></extra>",
        ))
        fig_donut.update_layout(
            height=360, showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(text="新規比率", x=0.5, y=0.58, showarrow=False,
                     font=dict(color=TEXT_MUTED, size=12)),
                dict(text=f"{_new_ratio:.0f}%", x=0.5, y=0.42, showarrow=False,
                     font=dict(color=TEXT, size=28, family="Inter")),
            ],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── 📐 売上ロジックツリー + 購入ファネル（横並び） ──
    # 顧客区分別集計
    new_df = completed_with_cust[completed_with_cust["顧客区分"] == "新規"]
    rep_df = completed_with_cust[completed_with_cust["顧客区分"] == "リピート"]
    new_rev = new_df["total_amount"].sum()
    rep_rev = rep_df["total_amount"].sum()
    new_orders = new_df["order_id"].nunique()
    rep_orders = rep_df["order_id"].nunique()
    new_customers_cnt = new_df["customer_id"].nunique()
    rep_customers_cnt = rep_df["customer_id"].nunique()
    new_aov = new_rev / new_orders if new_orders else 0
    rep_aov = rep_rev / rep_orders if rep_orders else 0
    rep_freq = rep_orders / rep_customers_cnt if rep_customers_cnt else 0

    # GA データ
    ga_total_sessions = int(ga["sessions"].sum())
    ga_product_pv = int(ga["product_page_views"].sum())
    ga_atc = int(ga["add_to_cart"].sum())
    ga_bounce = float(ga["bounce_rate"].mean())
    total_orders_all = completed["order_id"].nunique()
    overall_cvr = total_orders_all / ga_total_sessions if ga_total_sessions else 0
    new_sessions = int(ga_total_sessions * (new_orders / total_orders_all)) if total_orders_all else 0
    rep_sessions = ga_total_sessions - new_sessions
    new_cvr = new_orders / new_sessions if new_sessions else 0
    rep_cvr = rep_orders / rep_sessions if rep_sessions else 0

    # F2転換率
    cust_order_count = completed.groupby("customer_id")["order_id"].nunique()
    total_unique_cust = len(cust_order_count)
    f2_converted = (cust_order_count >= 2).sum()
    f2_rate = f2_converted / total_unique_cust if total_unique_cust else 0

    # 1注文あたりの平均商品点数・平均単価
    new_basket_items = new_df.groupby("order_id")["quantity"].sum().mean() if new_orders else 0
    rep_basket_items = rep_df.groupby("order_id")["quantity"].sum().mean() if rep_orders else 0
    new_unit_price = new_df["unit_price"].mean() if len(new_df) else 0
    rep_unit_price = rep_df["unit_price"].mean() if len(rep_df) else 0
    aov_delta = rep_aov - new_aov

    def _fmt_m(v):
        return f"¥{v/1e6:.1f}M" if v >= 1e6 else f"¥{v:,.0f}"

    tot_rev = new_rev + rep_rev

    # ── ロジックツリー HTML ──
    tree_html = f"""
    <div class="jcd-card">
      <div class="jcd-card__title">売上ロジックツリー</div>
      <div class="jcd-card__sub">売上を「新規」「リピート」に分解し、追うべきKPIを整理します</div>
      <div class="jcd-tree">
        <div class="jcd-tree__row jcd-tree__row--1">
          <div class="jcd-tnode jcd-tnode--total">
            <div class="jcd-tnode__label">総売上</div>
            <div class="jcd-tnode__value">{_fmt_m(tot_rev)}</div>
          </div>
        </div>
        <div class="jcd-tree__conn jcd-tree__conn--1to2" aria-hidden="true">
          <svg viewBox="0 0 100 40" preserveAspectRatio="none">
            <path d="M50 0 L50 20 M25 20 L75 20 M25 20 L25 40 M75 20 L75 40"
                  stroke="#30363D" stroke-width="0.6" fill="none"/>
          </svg>
        </div>
        <div class="jcd-tree__row jcd-tree__row--2">
          <div class="jcd-tnode jcd-tnode--new">
            <div class="jcd-tnode__label">新規売上</div>
            <div class="jcd-tnode__value">{_fmt_m(new_rev)}</div>
          </div>
          <div class="jcd-tnode jcd-tnode--rep">
            <div class="jcd-tnode__label">リピート売上</div>
            <div class="jcd-tnode__value">{_fmt_m(rep_rev)}</div>
          </div>
        </div>
        <div class="jcd-tree__conn jcd-tree__conn--2to3" aria-hidden="true">
          <svg viewBox="0 0 100 40" preserveAspectRatio="none">
            <path d="M25 0 L25 20 M10 20 L40 20 M10 20 L10 40 M25 20 L25 40 M40 20 L40 40
                     M75 0 L75 20 M60 20 L90 20 M60 20 L60 40 M75 20 L75 40 M90 20 L90 40"
                  stroke="#30363D" stroke-width="0.6" fill="none"/>
          </svg>
        </div>
        <div class="jcd-tree__row jcd-tree__row--3">
          <div class="jcd-tnode jcd-tnode--leaf jcd-tnode--new-sub">
            <div class="jcd-tnode__label">セッション数（推計）</div>
            <div class="jcd-tnode__value jcd-tnode__value--warn">{new_sessions:,}</div>
          </div>
          <div class="jcd-tnode jcd-tnode--leaf jcd-tnode--new-sub">
            <div class="jcd-tnode__label">新規CVR</div>
            <div class="jcd-tnode__value jcd-tnode__value--warn">{new_cvr*100:.2f}%</div>
          </div>
          <div class="jcd-tnode jcd-tnode--leaf jcd-tnode--new-sub">
            <div class="jcd-tnode__label">新規AOV</div>
            <div class="jcd-tnode__value jcd-tnode__value--warn">¥{new_aov:,.0f}</div>
          </div>
          <div class="jcd-tnode jcd-tnode--leaf jcd-tnode--rep-sub">
            <div class="jcd-tnode__label">リピート顧客数</div>
            <div class="jcd-tnode__value jcd-tnode__value--ok">{rep_customers_cnt:,}人</div>
          </div>
          <div class="jcd-tnode jcd-tnode--leaf jcd-tnode--rep-sub">
            <div class="jcd-tnode__label">購入頻度</div>
            <div class="jcd-tnode__value jcd-tnode__value--ok">{rep_freq:.1f}回/期</div>
          </div>
          <div class="jcd-tnode jcd-tnode--leaf jcd-tnode--rep-sub">
            <div class="jcd-tnode__label">リピートAOV</div>
            <div class="jcd-tnode__value jcd-tnode__value--ok">¥{rep_aov:,.0f}</div>
          </div>
        </div>
      </div>
    </div>
    """

    # ── 購入ファネル HTML ──
    funnel_stages = [
        ("セッション数",       ga_total_sessions, "muted"),
        ("商品ページ閲覧",     ga_product_pv,     "muted2"),
        ("カート投入",         ga_atc,            "warn"),
        ("購入",               total_orders_all,  "ok"),
    ]
    base_sess = max(ga_total_sessions, 1)
    funnel_rows_html = ""
    for label, cnt, variant in funnel_stages:
        pct = cnt / base_sess * 100
        funnel_rows_html += f"""
        <div class="jcd-funnel__row">
          <div class="jcd-funnel__label">{label}</div>
          <div class="jcd-funnel__bar">
            <svg viewBox="0 0 100 12" preserveAspectRatio="none">
              <rect x="0" y="0" width="100" height="12" rx="2" class="jcd-funnel__bg"/>
              <rect x="0" y="0" width="{pct:.2f}" height="12" rx="2" class="jcd-funnel__fill jcd-funnel__fill--{variant}"/>
            </svg>
          </div>
          <div class="jcd-funnel__count">{cnt:,}</div>
          <div class="jcd-funnel__pct">{pct:.1f}%</div>
        </div>
        """

    funnel_html = f"""
    <div class="jcd-card">
      <div class="jcd-card__title">購入ファネル（新規）</div>
      <div class="jcd-card__sub">セッションから購入までの転換率</div>
      <div class="jcd-funnel">{funnel_rows_html}</div>
      <div class="jcd-alert jcd-alert--info jcd-alert--mt">
        <span class="jcd-alert__msg"><strong>CVR {overall_cvr*100:.1f}%</strong> — カートから購入への転換率改善が優先課題です</span>
      </div>
      <div class="jcd-card__title jcd-card__title--sub">🧺 バスケット単価（AOV）比較</div>
      <div class="jcd-aov-compare">
        <div class="jcd-aov-cell">
          <div class="jcd-aov-cell__label">新規 AOV</div>
          <div class="jcd-aov-cell__value jcd-aov-cell__value--warn">¥{new_aov:,.0f}</div>
        </div>
        <div class="jcd-aov-cell">
          <div class="jcd-aov-cell__label">リピート AOV</div>
          <div class="jcd-aov-cell__value jcd-aov-cell__value--ok">¥{rep_aov:,.0f}</div>
        </div>
        <div class="jcd-aov-cell">
          <div class="jcd-aov-cell__label">差分</div>
          <div class="jcd-aov-cell__value jcd-aov-cell__value--muted">{'+' if aov_delta>=0 else ''}¥{aov_delta:,.0f}</div>
        </div>
      </div>
    </div>
    """

    tcol_l, tcol_r = st.columns([3, 2])
    with tcol_l:
        st.markdown(_html_compact(tree_html), unsafe_allow_html=True)
    with tcol_r:
        st.markdown(_html_compact(funnel_html), unsafe_allow_html=True)

    # ── 売上構成: 顧客種別 / カテゴリ別 / チャネル別（3カラム）──
    section_header("売上構成", "顧客種別 / カテゴリ別 / 流入チャネル別")
    col1, col2, col3 = st.columns(3)

    # 顧客種別売上
    cust_type_sales = (completed
                       .merge(customers[["customer_id", "customer_type"]], on="customer_id", how="left")
                       .groupby("customer_type")["total_amount"].sum()
                       .reset_index().rename(columns={"total_amount": "売上"}))
    fig_ctype = px.pie(cust_type_sales, names="customer_type", values="売上",
                       title="顧客種別 売上構成",
                       color_discrete_sequence=COLORS["chart"])
    col1.plotly_chart(fig_ctype, use_container_width=True)

    # カテゴリ別売上
    cat_sales = (completed
                 .merge(products[["sku_id", "category"]], on="sku_id", how="left")
                 .groupby("category")["total_amount"].sum()
                 .reset_index().rename(columns={"total_amount": "売上"}))
    fig_cat = px.pie(cat_sales, names="category", values="売上",
                     title="カテゴリ別売上構成",
                     color_discrete_sequence=COLORS["chart"])
    col2.plotly_chart(fig_cat, use_container_width=True)

    # チャネル別売上（流入チャネルを last-touch で注文に帰属）
    _CH_BUCKET = {
        "自然検索": "自然検索",
        "検索広告": "デジタル広告",
        "ディスプレイ広告": "デジタル広告",
        "リターゲ": "デジタル広告",
        "DM": "メルマガ／DM",
        "ダイレクト": "ダイレクト",
    }
    if ga_sessions is not None and not ga_sessions.empty:
        _o = (completed[["order_id", "customer_id", "order_date", "total_amount"]]
              .sort_values("order_date"))
        _g = ga_sessions[["customer_id", "date", "channel"]].sort_values("date")
        _att = pd.merge_asof(_o, _g, left_on="order_date", right_on="date",
                             by="customer_id", direction="backward")
        _att["チャネル"] = _att["channel"].map(_CH_BUCKET).fillna("ダイレクト")
        ch_sales = (_att.groupby("チャネル")["total_amount"].sum()
                    .reset_index().rename(columns={"total_amount": "売上"})
                    .sort_values("売上", ascending=False))
        fig_ch = px.pie(ch_sales, names="チャネル", values="売上",
                        title="チャネル別売上構成",
                        color_discrete_sequence=COLORS["chart"])
        col3.plotly_chart(fig_ch, use_container_width=True)
    else:
        col3.info("チャネルデータがありません")

    # ── 品番別売上ランキング ──
    _n_sku = completed["sku_id"].nunique()
    section_header("品番別売上ランキング",
                   f"売上上位の品番（完了注文ベース・上位15 / 全{_n_sku}品番）")
    _pm = products[["sku_id", "product_name", "color", "size"]].copy()
    _pm["品番"] = (_pm["product_name"] + " " + _pm["color"].astype(str)
                   + "/" + _pm["size"].astype(str) + "（" + _pm["sku_id"] + "）")
    prod_sales = (completed
                  .merge(_pm[["sku_id", "品番"]], on="sku_id", how="left")
                  .groupby("品番")["total_amount"].sum()
                  .sort_values(ascending=False)
                  .head(15)
                  .reset_index().rename(columns={"total_amount": "売上"}))
    fig_prod = px.bar(prod_sales, x="売上", y="品番", orientation="h",
                      color_discrete_sequence=[ACCENT])
    fig_prod.update_layout(yaxis_title="品番", xaxis_title="売上（円）",
                           yaxis=dict(autorange="reversed"), height=520)
    st.plotly_chart(fig_prod, use_container_width=True)

    # ── 都道府県別売上 ──
    pref_sales = (completed
                  .merge(customers[["customer_id", "prefecture"]], on="customer_id", how="left")
                  .groupby("prefecture")["total_amount"].sum()
                  .sort_values(ascending=False)
                  .reset_index().rename(columns={"total_amount": "売上"}))
    fig_pref = px.bar(pref_sales, x="prefecture", y="売上",
                      title="都道府県別 売上ランキング",
                      color_discrete_sequence=[COLORS["accent"]])
    fig_pref.update_layout(xaxis_title="都道府県", yaxis_title="売上（円）")
    st.plotly_chart(fig_pref, use_container_width=True)

    # ── 曜日別 ──
    dow_map = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
    dow_sales = (completed
                 .assign(dow=completed["order_date"].dt.dayofweek.map(dow_map))
                 .groupby("dow")["total_amount"].mean()
                 .reindex(["月", "火", "水", "木", "金", "土", "日"])
                 .reset_index().rename(columns={"total_amount": "平均売上"}))
    fig_dow = px.bar(dow_sales, x="dow", y="平均売上",
                     title="曜日別 平均売上",
                     color_discrete_sequence=[COLORS["primary"]])
    st.plotly_chart(fig_dow, use_container_width=True)


# ============================================================
# 🔮 需要予測
# ============================================================
elif page == "需要予測":
    page_header("需要予測", "カテゴリ・SKU別の需要予測と広告シミュレーション",
                key_prefix="fcst",
                latest_date=orders["order_date"].max())

    completed = orders[orders["status"] == "完了"]

    # ── 月次売上予測 ──
    monthly = (completed
               .assign(month_dt=completed["order_date"].dt.to_period("M").dt.to_timestamp())
               .groupby("month_dt")["total_amount"].sum()
               .reset_index().rename(columns={"month_dt": "date", "total_amount": "sales"}))
    monthly["month_num"] = np.arange(len(monthly))
    monthly["month_of_year"] = monthly["date"].dt.month
    for m in range(1, 13):
        monthly[f"m_{m}"] = (monthly["month_of_year"] == m).astype(int)
    feature_cols = ["month_num"] + [f"m_{m}" for m in range(1, 13)]
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(monthly[feature_cols].values, monthly["sales"].values)

    last_month_num = monthly["month_num"].iloc[-1]
    last_date = monthly["date"].iloc[-1]  # 現在 = 直近実績月（モック前提: 2026/9）
    FORECAST_MONTHS = 6
    future_rows = []
    for i in range(1, FORECAST_MONTHS + 1):
        fdate = last_date + pd.DateOffset(months=i)
        row = {"date": fdate, "month_num": last_month_num + i, "month_of_year": fdate.month}
        for m in range(1, 13):
            row[f"m_{m}"] = 1 if fdate.month == m else 0
        future_rows.append(row)
    future_df = pd.DataFrame(future_rows)
    future_df["sales"] = model.predict(future_df[feature_cols].values)
    # 下限・上限は予測売上から ±30% の幅
    future_df["lower"] = future_df["sales"] * 0.7
    future_df["upper"] = future_df["sales"] * 1.3

    st.subheader("月次売上の実績と予測")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["date"], y=monthly["sales"],
                             mode="lines+markers", name="実績",
                             line=dict(color=COLORS["primary"], width=2)))
    fig.add_trace(go.Scatter(x=future_df["date"], y=future_df["sales"],
                             mode="lines+markers", name="予測",
                             line=dict(color="#E57373", width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=pd.concat([future_df["date"], future_df["date"][::-1]]),
        y=pd.concat([future_df["upper"], future_df["lower"][::-1]]),
        fill="toself", fillcolor="rgba(229,115,115,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="予測範囲"))
    fig.update_layout(title="売上推移と6ヶ月予測",
                      xaxis_title="月", yaxis_title="売上（円）", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ── 実績（2026/03〜2026/09）＋ 予測（2026/10〜2027/03）の対比表（中央＝現在 2026/9）──
    st.subheader("売上 実績・予測テーブル（中央＝現在 2026/9）")
    st.caption("左＝2026/03〜2026/09の実績、右＝2026/10〜2027/03の予測（下限 −30% ／ 上限 +30%）。"
               "※ 2026/04・05は新ECローンチ前で実績データがありません。")
    # 左側: 2026/03〜2026/09 の実績（データのない月は「—」）
    left_months = pd.date_range("2026-03-01", "2026-09-01", freq="MS")
    _past_s = monthly.set_index("date")["sales"]
    past_tbl = pd.DataFrame({"date": left_months})
    past_tbl["区分"] = "実績"
    past_tbl["売上"] = past_tbl["date"].map(_past_s).apply(
        lambda v: "—" if pd.isna(v) else f"¥{v:,.0f}")
    past_tbl["下限"] = "—"
    past_tbl["上限"] = "—"
    # 右側: 予測6ヶ月（2026/10〜2027/03）
    fut_tbl = future_df[["date", "sales", "lower", "upper"]].copy()
    fut_tbl["区分"] = "予測"
    fut_tbl["売上"] = fut_tbl["sales"].apply(lambda v: f"¥{v:,.0f}")
    fut_tbl["下限"] = fut_tbl["lower"].apply(lambda v: f"¥{v:,.0f}")
    fut_tbl["上限"] = fut_tbl["upper"].apply(lambda v: f"¥{v:,.0f}")
    _cols = ["date", "区分", "売上", "下限", "上限"]
    tbl = pd.concat([past_tbl[_cols], fut_tbl[_cols]], ignore_index=True)
    tbl["月"] = tbl["date"].dt.strftime("%Y/%m")
    tbl_t = tbl.set_index("月")[["区分", "売上", "下限", "上限"]].T
    st.dataframe(tbl_t, use_container_width=True)

    # ── SKU別 予測販売数 ──
    st.markdown("---")

    # ── 📣 広告出稿プラン シミュレーター ──
    st.subheader("📣 広告出稿プラン シミュレーター")
    st.caption("媒体別の投下金額を調整すると、需要予測が連動して変わります（What-if 分析）")

    if ad_plan is None or len(ad_plan) == 0:
        st.warning("広告プラン（ad_plan.csv）が読み込まれていません。")
        future_ad_map = None
    else:
        plan_work = ad_plan.copy()
        plan_work["month_ts"] = pd.to_datetime(plan_work["month"] + "-01")
        # 現在から6ヶ月先（予測期間と同じ月）のプランのみに絞る
        plan_work = plan_work[plan_work["month_ts"].isin(set(future_df["date"]))].copy()

        channels = sorted(plan_work["channel"].unique())
        st.markdown("**媒体別 予算調整スライダー**（1.0 = プラン通り）")
        mult_cols = st.columns(len(channels))
        multipliers = {}
        for i, ch in enumerate(channels):
            obj = plan_work[plan_work["channel"] == ch]["objective"].iloc[0]
            multipliers[ch] = mult_cols[i].slider(
                f"{ch}（{obj}）", min_value=0.0, max_value=2.0, value=1.0, step=0.1,
                key=f"ad_mult_{ch}")

        # 調整後の予算
        plan_work["adjusted_cost"] = plan_work.apply(
            lambda r: int(r["planned_cost"] * multipliers[r["channel"]]), axis=1)

        # ベースライン（プラン通り）と調整後の月次総額
        monthly_plan = plan_work.groupby("month_ts").agg(
            planned=("planned_cost", "sum"),
            adjusted=("adjusted_cost", "sum"),
        ).reset_index()

        # KPI比較
        total_planned = int(monthly_plan["planned"].sum())
        total_adjusted = int(monthly_plan["adjusted"].sum())
        delta_budget = total_adjusted - total_planned

        mk1, mk2, mk3 = st.columns(3)
        mk1.metric("プラン総予算（6ヶ月）", f"¥{total_planned:,}")
        mk2.metric("調整後 総予算", f"¥{total_adjusted:,}",
                   delta=f"¥{delta_budget:+,}")
        mk3.metric("増減率", f"{(delta_budget/total_planned*100 if total_planned else 0):+.1f}%")

        # プラン表示（媒体×月のヒートマップ風）
        with st.expander("📋 月×媒体 広告プラン詳細"):
            pivot = plan_work.pivot(index="channel", columns="month", values="adjusted_cost")
            pivot_disp = pivot.copy().astype(object)
            for idx in pivot.index:
                for c in pivot.columns:
                    pivot_disp.loc[idx, c] = f"¥{pivot.loc[idx, c]:,.0f}"
            st.dataframe(pivot_disp, use_container_width=True)

        # 予測用の辞書（調整後）: {month_ts: adjusted total} — 下の SKU別予測に反映
        future_ad_map = dict(zip(monthly_plan["month_ts"], monthly_plan["adjusted"].astype(float)))
        st.caption("スライダーの調整結果は、下の「SKU別 予測販売数」に反映されます。")

    st.markdown("---")
    st.subheader("SKU別 予測販売数（来月〜6ヶ月先）")
    st.markdown("各SKUがどれだけ売れるかの予測です。下の表は **調整後プラン** に基づきます。")

    demand = forecast_category_demand(future_ad_map, horizon=6)

    sel_cat_sku = st.selectbox("カテゴリを選択", sorted(demand.keys()), key="sku_fc_cat")
    if sel_cat_sku in demand:
        cat_fc = demand[sel_cat_sku]["forecast"]
        cat_skus = products[products["category"] == sel_cat_sku].copy()
        # 直近6ヶ月のSKU別販売比率を算出
        recent_orders = completed[
            (completed["order_date"] >= completed["order_date"].max() - pd.DateOffset(months=6)) &
            (completed["sku_id"].isin(cat_skus["sku_id"]))
        ]
        sku_ratio = (recent_orders.groupby("sku_id")["quantity"].sum()
                     .reset_index().rename(columns={"quantity": "recent_qty"}))
        total_recent = sku_ratio["recent_qty"].sum()
        sku_ratio["ratio"] = sku_ratio["recent_qty"] / total_recent if total_recent > 0 else 0

        # SKUごとに月別予測を計算
        sku_forecast_rows = []
        for _, sku in cat_skus.iterrows():
            r = sku_ratio[sku_ratio["sku_id"] == sku["sku_id"]]
            ratio = r["ratio"].iloc[0] if len(r) > 0 else 1.0 / len(cat_skus)
            row = {
                "商品名": sku["product_name"],
                "色": sku["color"],
                "サイズ": sku["size"],
                "SKU": sku["sku_id"],
            }
            for _, fc_row in cat_fc.iterrows():
                month_label = fc_row["date"].strftime("%Y/%m")
                row[month_label] = max(1, int(fc_row["qty"] * ratio))
            row["合計"] = sum(v for k, v in row.items() if k not in ["商品名", "色", "サイズ", "SKU"])
            sku_forecast_rows.append(row)

        sku_fc_df = pd.DataFrame(sku_forecast_rows).sort_values("合計", ascending=False)
        st.dataframe(sku_fc_df, use_container_width=True, hide_index=True)

        # カテゴリ合計
        total_cat = sku_fc_df["合計"].sum()
        st.info(f"📦 {sel_cat_sku} 全体: 6ヶ月間で **{total_cat:,}個** の販売を予測")

    # ── カテゴリ別推移 ──
    st.subheader("カテゴリ別 月次売上推移")
    cat_monthly = (completed
                   .merge(products[["sku_id", "category"]], on="sku_id", how="left")
                   .assign(month=completed["order_date"].dt.to_period("M").astype(str))
                   .groupby(["month", "category"])["total_amount"].sum()
                   .reset_index())
    fig_cat = px.line(cat_monthly, x="month", y="total_amount", color="category",
                      title="カテゴリ別売上推移", color_discrete_sequence=COLORS["chart"])
    fig_cat.update_layout(xaxis_title="月", yaxis_title="売上（円）", xaxis_tickangle=-45)
    st.plotly_chart(fig_cat, use_container_width=True)


# ============================================================
# 🏭 需要予測×在庫管理
# ============================================================
elif page == "需要予測×在庫管理":
    page_header("需要予測×在庫管理",
                "需要予測をもとに、基幹倉庫→EC在庫倉庫→発送倉庫の在庫を最適化します",
                key_prefix="invmgmt",
                latest_date=orders["order_date"].max())

    completed = orders[orders["status"] == "完了"]
    demand = forecast_category_demand()
    latest_inv = inventory[inventory["date"] == inventory["date"].max()]

    # 横持ち移動（基幹倉庫→EC在庫倉庫）のリードタイム（日）
    TRANSFER_LEAD_DAYS = 7
    # モック前提の「現在」
    NOW = pd.Timestamp("2026-09-30")

    st.markdown("""
    **在庫フロー: 基幹倉庫（本業）→ EC在庫倉庫 → 発送倉庫**

    本業（EC以外）が売上の大半を占めるため、ECで生産発注は行いません。
    需要予測に対して在庫が不足するSKUは、**基幹倉庫から横持ち移動**で補充します。

    > **STEP 1** 需要予測から、発送倉庫で足りないSKUを特定
    > → **STEP 2** 不足分を **EC在庫倉庫** から発送倉庫へ移動
    > → **STEP 3** EC在庫倉庫でも足りない分 → **基幹倉庫から EC在庫倉庫へ移動**（移動時期つき）
    """)

    replenish_rows = []
    for cat, ddata in demand.items():
        if len(ddata["forecast"]) == 0:
            continue
        next_month_qty = ddata["forecast"].iloc[0]["qty"]
        cat_skus = products[products["category"] == cat]["sku_id"].tolist()
        n_skus = max(1, len(cat_skus))
        per_sku_demand = max(1, int(next_month_qty / n_skus))

        for sku_id in cat_skus:
            ship_inv = latest_inv[(latest_inv["sku_id"] == sku_id) &
                                  (latest_inv["warehouse_type"] == "発送倉庫")]
            ec_inv = latest_inv[(latest_inv["sku_id"] == sku_id) &
                                (latest_inv["warehouse_type"] == "EC在庫倉庫")]
            ship_stock = int(ship_inv["stock_quantity"].sum()) if len(ship_inv) > 0 else 0
            ec_stock = int(ec_inv["stock_quantity"].sum()) if len(ec_inv) > 0 else 0
            total_stock = ship_stock + ec_stock
            shortage_from_ship = max(0, per_sku_demand - ship_stock)
            transfer = min(shortage_from_ship, ec_stock)  # EC在庫倉庫 → 発送倉庫
            still_short = max(0, per_sku_demand - total_stock)  # 基幹倉庫から移動が必要な量
            prod = products[products["sku_id"] == sku_id].iloc[0]

            # 基幹倉庫からEC在庫倉庫への移動が必要な時期
            if still_short > 0:
                daily_demand = max(1, per_sku_demand // 30)
                days_until_stockout = max(0, total_stock // daily_demand)
                stockout_date = NOW + pd.Timedelta(days=days_until_stockout)
                move_deadline = stockout_date - pd.Timedelta(days=TRANSFER_LEAD_DAYS)
                deadline_str = move_deadline.strftime("%Y/%m/%d")
                status = "🚨 基幹倉庫から移動"
            elif shortage_from_ship > 0:
                deadline_str = ""
                status = "🔄 EC在庫倉庫から移動"
            else:
                deadline_str = ""
                status = "✅ 在庫十分"

            replenish_rows.append({
                "商品名": prod["product_name"],
                "カテゴリ": cat,
                "色": prod["color"],
                "サイズ": prod["size"],
                "来月予測需要": per_sku_demand,
                "発送倉庫 在庫": ship_stock,
                "EC在庫倉庫 在庫": ec_stock,
                "STEP1 発送不足": shortage_from_ship,
                "STEP2 EC在庫倉庫から移動": transfer,
                "STEP3 基幹倉庫から移動": still_short,
                "基幹倉庫からの移動期限": deadline_str,
                "判定": status,
            })

    if replenish_rows:
        rep_df = pd.DataFrame(replenish_rows)
        needs_core = rep_df[rep_df["判定"] == "🚨 基幹倉庫から移動"]
        needs_transfer = rep_df[rep_df["判定"] == "🔄 EC在庫倉庫から移動"]
        ok_skus = rep_df[rep_df["判定"] == "✅ 在庫十分"]

        # ── KPI ──
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("在庫十分", f"{len(ok_skus)} SKU")
        col2.metric("EC在庫倉庫から移動で対応", f"{len(needs_transfer)} SKU")
        col3.metric("🚨 基幹倉庫から移動が必要", f"{len(needs_core)} SKU")
        total_short = int(needs_core["STEP3 基幹倉庫から移動"].sum()) if len(needs_core) > 0 else 0
        col4.metric("基幹倉庫から移動 総数", f"{total_short:,}個")

        # ── STEP 1: 発送倉庫の不足 ──
        st.subheader("STEP 1: 需要予測に対する発送倉庫の過不足")
        step1 = rep_df[rep_df["STEP1 発送不足"] > 0].sort_values("STEP1 発送不足", ascending=False)
        if len(step1) > 0:
            st.warning(f"⚠️ 発送倉庫だけでは **{len(step1)} SKU** が不足します（総不足: {int(step1['STEP1 発送不足'].sum()):,}個）")
        else:
            st.success("発送倉庫の在庫で来月の需要を充足できます。")

        # ── STEP 2: EC在庫倉庫からの移動 ──
        st.subheader("STEP 2: EC在庫倉庫から発送倉庫への移動")
        if len(needs_transfer) > 0:
            st.info(f"🔄 **{len(needs_transfer)} SKU** はEC在庫倉庫からの移動で対応可能です")
            with st.expander(f"EC在庫倉庫から移動で対応（{len(needs_transfer)} SKU）— 詳細を見る"):
                st.dataframe(needs_transfer[["商品名", "カテゴリ", "色", "サイズ",
                             "来月予測需要", "発送倉庫 在庫", "STEP1 発送不足",
                             "EC在庫倉庫 在庫", "STEP2 EC在庫倉庫から移動"]],
                             use_container_width=True, hide_index=True)

        # ── STEP 3: EC在庫倉庫でも不足 → 基幹倉庫から移動 ──
        st.subheader("STEP 3: EC在庫倉庫でも不足 → 基幹倉庫から移動")
        if len(needs_core) > 0:
            st.error(f"""
            🚨 **{len(needs_core)} SKU** がEC在庫倉庫の在庫でも賄えません。
            合計 **{total_short:,}個** を基幹倉庫からEC在庫倉庫へ移動する必要があります。
            """)

            # カテゴリ別サマリー（移動が必要な順）
            cat_summary = (needs_core.groupby("カテゴリ")
                           .agg(SKU数=("商品名", "count"),
                                不足合計=("STEP3 基幹倉庫から移動", "sum"),
                                最短移動期限=("基幹倉庫からの移動期限", "min"))
                           .reset_index()
                           .sort_values("最短移動期限"))
            cat_summary["不足合計"] = cat_summary["不足合計"].apply(lambda v: f"{int(v):,}個")

            st.markdown("**カテゴリ別 移動アラート（移動が必要な順）**")
            for _, row in cat_summary.iterrows():
                deadline = row["最短移動期限"]
                if deadline and pd.Timestamp(deadline) <= NOW + pd.Timedelta(days=14):
                    st.error(f"🚨 **{row['カテゴリ']}**: {row['不足合計']}不足 / "
                             f"{row['SKU数']} SKU / **{deadline} までに基幹倉庫から移動**")
                elif deadline and pd.Timestamp(deadline) <= NOW + pd.Timedelta(days=30):
                    st.warning(f"⚠️ **{row['カテゴリ']}**: {row['不足合計']}不足 / "
                               f"{row['SKU数']} SKU / {deadline} までに移動")
                else:
                    st.info(f"📋 **{row['カテゴリ']}**: {row['不足合計']}不足 / "
                            f"{row['SKU数']} SKU / {deadline} までに移動")

            st.markdown("**詳細一覧（移動が必要な順）**")
            move_disp = needs_core.sort_values("基幹倉庫からの移動期限")
            st.dataframe(move_disp[["商品名", "カテゴリ", "色", "サイズ",
                         "来月予測需要", "発送倉庫 在庫", "EC在庫倉庫 在庫",
                         "STEP3 基幹倉庫から移動", "基幹倉庫からの移動期限"]],
                         use_container_width=True, hide_index=True)
        else:
            st.success("✅ すべてのSKUがEC在庫倉庫からの移動で対応可能です。基幹倉庫からの移動は不要です。")
    else:
        st.success("来月の予測需要に対し、発送倉庫の在庫は十分です。")


# ============================================================
# 📢 広告効果分析
# ============================================================
elif page == "広告効果分析":
    page_header("広告効果分析", "媒体別の広告効果・ROAS・CVRを分析します",
                key_prefix="ads",
                latest_date=orders["order_date"].max())

    ch_summary = (ads.groupby("channel")
                  .agg(費用=("cost", "sum"), クリック=("clicks", "sum"),
                       CV数=("conversions", "sum"), 売上=("revenue", "sum"))
                  .reset_index())
    ch_summary["CPA"] = (ch_summary["費用"] / ch_summary["CV数"]).round(0)
    ch_summary["ROAS"] = (ch_summary["売上"] / ch_summary["費用"] * 100).round(1)

    st.subheader("チャネル別サマリー")
    disp = ch_summary.copy()
    for c in ["費用", "売上"]:
        disp[c] = disp[c].apply(lambda v: f"¥{v:,.0f}")
    disp["CPA"] = disp["CPA"].apply(lambda v: f"¥{v:,.0f}")
    disp["ROAS"] = disp["ROAS"].apply(lambda v: f"{v}%")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    col_l, col_r = st.columns(2)
    fig_roas = px.bar(ch_summary, x="channel", y="ROAS", title="チャネル別 ROAS（%）",
                      color="channel", color_discrete_sequence=COLORS["chart"])
    fig_roas.update_layout(showlegend=False)
    col_l.plotly_chart(fig_roas, use_container_width=True)

    cv_breakdown = ads.groupby(["channel", "cv_type"])["conversions"].sum().reset_index()
    fig_cv = px.bar(cv_breakdown, x="channel", y="conversions", color="cv_type", barmode="stack",
                    title="チャネル別 CV種別内訳", color_discrete_sequence=COLORS["chart"])
    col_r.plotly_chart(fig_cv, use_container_width=True)

    st.subheader("月次 広告費 vs 売上")
    ads_monthly = (ads.assign(month=ads["date"].dt.to_period("M").astype(str))
                   .groupby("month").agg(広告費=("cost", "sum"), 広告売上=("revenue", "sum")).reset_index())
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(x=ads_monthly["month"], y=ads_monthly["広告費"],
                               name="広告費", marker_color=COLORS["light"]))
    fig_trend.add_trace(go.Scatter(x=ads_monthly["month"], y=ads_monthly["広告売上"],
                                   name="広告売上", mode="lines+markers",
                                   line=dict(color=COLORS["primary"], width=2), yaxis="y2"))
    fig_trend.update_layout(yaxis=dict(title="広告費（円）"),
                            yaxis2=dict(title="広告売上（円）", overlaying="y", side="right"),
                            xaxis_tickangle=-45, hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── 広告経由の注文を特定（流入チャネル last-touch）→ 品番別 売上 / 利益 ──
    _AD_CHANNELS = {"検索広告", "ディスプレイ広告", "リターゲ"}
    _comp_ad = orders[orders["status"] == "完了"].copy()
    if ga_sessions is not None and not ga_sessions.empty:
        _oo = (_comp_ad[["order_id", "customer_id", "order_date", "sku_id", "quantity", "total_amount"]]
               .sort_values("order_date"))
        _gg = ga_sessions[["customer_id", "date", "channel"]].sort_values("date")
        _m = pd.merge_asof(_oo, _gg, left_on="order_date", right_on="date",
                           by="customer_id", direction="backward")
        ad_orders = _m[_m["channel"].isin(_AD_CHANNELS)].copy()

        # 品番ラベル + 原価から粗利を算出
        _pm = products[["sku_id", "product_name", "color", "size", "cost_price"]].copy()
        _pm["品番"] = (_pm["product_name"] + " " + _pm["color"].astype(str)
                       + "/" + _pm["size"].astype(str) + "（" + _pm["sku_id"] + "）")
        ad_orders = ad_orders.merge(_pm[["sku_id", "品番", "cost_price"]], on="sku_id", how="left")
        ad_orders["利益"] = ad_orders["total_amount"] - ad_orders["cost_price"] * ad_orders["quantity"]
        ad_orders["month"] = ad_orders["order_date"].dt.to_period("M").astype(str)

        # (1) 広告経由 売上品番ランキング
        st.subheader("広告経由 売上品番ランキング")
        st.caption("流入チャネルが広告（検索広告／ディスプレイ広告／リターゲ）の完了注文を品番別に集計・上位15")
        ad_prod_rev = (ad_orders.groupby("品番")["total_amount"].sum()
                       .sort_values(ascending=False).head(15)
                       .reset_index().rename(columns={"total_amount": "売上"}))
        fig_ad_rev = px.bar(ad_prod_rev, x="売上", y="品番", orientation="h",
                            color_discrete_sequence=[COLORS["accent"]])
        fig_ad_rev.update_layout(yaxis_title="品番", xaxis_title="広告経由 売上（円）",
                                 yaxis=dict(autorange="reversed"), height=520)
        st.plotly_chart(fig_ad_rev, use_container_width=True)

        # (2) 月次 広告費 vs 利益
        st.subheader("月次 広告費 vs 利益")
        st.caption("広告費は媒体実績、利益は広告経由注文の粗利（売上−原価）")
        ad_cost_monthly = (ads.assign(month=ads["date"].dt.to_period("M").astype(str))
                           .groupby("month")["cost"].sum().reset_index(name="広告費"))
        ad_profit_monthly = ad_orders.groupby("month")["利益"].sum().reset_index(name="広告経由利益")
        pm_merge = (ad_cost_monthly.merge(ad_profit_monthly, on="month", how="outer")
                    .fillna(0).sort_values("month"))
        fig_ap = go.Figure()
        fig_ap.add_trace(go.Bar(x=pm_merge["month"], y=pm_merge["広告費"],
                                name="広告費", marker_color=COLORS["light"]))
        fig_ap.add_trace(go.Scatter(x=pm_merge["month"], y=pm_merge["広告経由利益"],
                                    name="広告経由利益", mode="lines+markers",
                                    line=dict(color=COLORS["primary"], width=2), yaxis="y2"))
        fig_ap.update_layout(yaxis=dict(title="広告費（円）"),
                             yaxis2=dict(title="広告経由利益（円）", overlaying="y", side="right"),
                             xaxis_tickangle=-45, hovermode="x unified")
        st.plotly_chart(fig_ap, use_container_width=True)

        # (3) 広告経由 利益品番ランキング
        st.subheader("広告経由 利益品番ランキング")
        st.caption("広告経由注文の粗利（売上−原価）を品番別に集計・上位15")
        ad_prod_profit = (ad_orders.groupby("品番")["利益"].sum()
                          .sort_values(ascending=False).head(15).reset_index())
        fig_ad_profit = px.bar(ad_prod_profit, x="利益", y="品番", orientation="h",
                               color_discrete_sequence=[COLORS["primary"]])
        fig_ad_profit.update_layout(yaxis_title="品番", xaxis_title="広告経由 利益（円）",
                                    yaxis=dict(autorange="reversed"), height=520)
        st.plotly_chart(fig_ad_profit, use_container_width=True)
    else:
        st.info("流入チャネルデータ（ga_sessions）がないため、広告経由の品番分析は表示できません。")


# ============================================================
# 📦 在庫分析
# ============================================================
elif page == "在庫分析":
    page_header("在庫分析", "在庫回転率・SKU別在庫状況を確認します",
                key_prefix="inv",
                latest_date=orders["order_date"].max())

    col_wh, col_cat = st.columns(2)
    wh_types = ["全体"] + sorted(inventory["warehouse_type"].unique().tolist())
    sel_wh = col_wh.selectbox("倉庫種別", wh_types)
    inv_filtered = inventory if sel_wh == "全体" else inventory[inventory["warehouse_type"] == sel_wh]

    inv_cat = (inv_filtered.merge(products[["sku_id", "category"]], on="sku_id", how="left")
               .groupby(["date", "category"])
               .agg(在庫数=("stock_quantity", "sum"), 入庫=("inbound", "sum"), 出庫=("outbound", "sum"))
               .reset_index())

    sel_cat = col_cat.selectbox("カテゴリを選択", sorted(inv_cat["category"].unique()))
    cat_data = inv_cat[inv_cat["category"] == sel_cat]

    fig_inv = go.Figure()
    fig_inv.add_trace(go.Bar(x=cat_data["date"], y=cat_data["入庫"], name="入庫", marker_color=COLORS["accent"]))
    fig_inv.add_trace(go.Bar(x=cat_data["date"], y=-cat_data["出庫"], name="出庫", marker_color="#E57373"))
    fig_inv.add_trace(go.Scatter(x=cat_data["date"], y=cat_data["在庫数"], name="在庫数",
                                 mode="lines+markers", line=dict(color=COLORS["primary"], width=2)))
    fig_inv.update_layout(title=f"{sel_cat} — 在庫推移（{sel_wh}）", barmode="relative",
                          xaxis_title="月", yaxis_title="数量")
    st.plotly_chart(fig_inv, use_container_width=True)

    recent = inv_filtered[inv_filtered["date"] >= inv_filtered["date"].max() - pd.DateOffset(months=5)]
    recent_cat = (recent.merge(products[["sku_id", "category"]], on="sku_id", how="left")
                  .groupby("category")
                  .agg(平均在庫=("stock_quantity", "mean"), 総出庫=("outbound", "sum")).reset_index())
    recent_cat["在庫回転率"] = (recent_cat["総出庫"] / recent_cat["平均在庫"]).round(2)
    fig_turn = px.bar(recent_cat, x="category", y="在庫回転率",
                      title="カテゴリ別 在庫回転率（直近6ヶ月）",
                      color_discrete_sequence=[COLORS["accent"]])
    st.plotly_chart(fig_turn, use_container_width=True)

    st.subheader("在庫アラート（発送倉庫）")
    ship_latest = inventory[(inventory["date"] == inventory["date"].max()) &
                            (inventory["warehouse_type"] == "発送倉庫")]
    ship_detail = ship_latest.merge(products[["sku_id", "product_name", "category", "color", "size"]],
                                    on="sku_id", how="left")
    low_stock = ship_detail[ship_detail["stock_quantity"] < 15].sort_values("stock_quantity")
    if len(low_stock) > 0:
        st.warning(f"⚠️ 発送倉庫で在庫が少ない商品: {len(low_stock)} SKU")
        st.dataframe(low_stock[["product_name", "category", "color", "size",
                                "stock_quantity", "warehouse_name"]].head(15),
                     use_container_width=True, hide_index=True)
    else:
        st.success("発送倉庫で在庫が極端に少ない商品はありません。")


# ============================================================
# 🎯 キャンペーン分析
# ============================================================
elif page == "キャンペーン分析":
    page_header("キャンペーン分析", "過去・予定キャンペーンの効果を比較します",
                key_prefix="camp",
                latest_date=orders["order_date"].max())

    completed = orders[orders["status"] == "完了"]
    camp_options = campaigns["campaign_name"].tolist()
    sel_camp = st.selectbox("キャンペーンを選択", camp_options)
    camp_row = campaigns[campaigns["campaign_name"] == sel_camp].iloc[0]

    st.markdown(f"""
    | 項目 | 内容 |
    |------|------|
    | **期間** | {camp_row['start_date'].strftime('%Y/%m/%d')} 〜 {camp_row['end_date'].strftime('%Y/%m/%d')} |
    | **割引率** | {camp_row['discount_rate']}% |
    | **種別** | {camp_row['coupon_type']} |
    | **対象** | {camp_row['target_category']} |
    | **予算** | ¥{camp_row['budget']:,} |
    | **実績費用** | ¥{camp_row['actual_spend']:,} |
    """)

    dur = (camp_row["end_date"] - camp_row["start_date"]).days
    before_start = camp_row["start_date"] - pd.Timedelta(days=dur)
    after_end = camp_row["end_date"] + pd.Timedelta(days=dur)

    def period_sales(start, end):
        mask = (completed["order_date"] >= start) & (completed["order_date"] <= end)
        return completed.loc[mask, "total_amount"].sum()

    sales_before = period_sales(before_start, camp_row["start_date"] - pd.Timedelta(days=1))
    sales_during = period_sales(camp_row["start_date"], camp_row["end_date"])
    sales_after = period_sales(camp_row["end_date"] + pd.Timedelta(days=1), after_end)

    comp = pd.DataFrame({"期間": ["前", "中", "後"], "売上": [sales_before, sales_during, sales_after]})
    col_l, col_r = st.columns(2)
    fig_comp = px.bar(comp, x="期間", y="売上", title="前後比較（同期間）", color="期間",
                      color_discrete_sequence=[COLORS["light"], COLORS["accent"], COLORS["primary"]])
    fig_comp.update_layout(showlegend=False)
    col_l.plotly_chart(fig_comp, use_container_width=True)

    if sales_before > 0:
        during_lift = (sales_during - sales_before) / sales_before * 100
        after_lift = (sales_after - sales_before) / sales_before * 100
    else:
        during_lift = after_lift = 0
    fig_lift = go.Figure(go.Bar(
        x=["キャンペーン中", "キャンペーン後"], y=[during_lift, after_lift],
        marker_color=[COLORS["accent"], COLORS["primary"]],
        text=[f"{during_lift:+.1f}%", f"{after_lift:+.1f}%"], textposition="outside"))
    fig_lift.update_layout(title="売上増減率（前比）", yaxis_title="増減率（%）")
    col_r.plotly_chart(fig_lift, use_container_width=True)

    st.subheader("日次売上推移")
    daily_range = completed[
        (completed["order_date"] >= before_start) & (completed["order_date"] <= after_end)
    ].groupby("order_date")["total_amount"].sum().reset_index()
    fig_daily = px.line(daily_range, x="order_date", y="total_amount",
                        title="日次売上推移（キャンペーン前後）",
                        color_discrete_sequence=[COLORS["primary"]])
    fig_daily.add_vrect(x0=camp_row["start_date"], x1=camp_row["end_date"],
                        fillcolor=COLORS["accent"], opacity=0.15, line_width=0,
                        annotation_text="キャンペーン期間", annotation_position="top left")
    fig_daily.update_layout(xaxis_title="日付", yaxis_title="売上（円）")
    st.plotly_chart(fig_daily, use_container_width=True)

    # ── キャンペーン期間中の 売上品番ランキング ──
    st.subheader("売上品番ランキング（キャンペーン期間中）")
    st.caption(f"{sel_camp} の期間"
               f"（{camp_row['start_date'].strftime('%Y/%m/%d')}〜"
               f"{camp_row['end_date'].strftime('%Y/%m/%d')}）の完了注文を品番別に集計・上位15")
    _camp_mask = ((completed["order_date"] >= camp_row["start_date"])
                  & (completed["order_date"] <= camp_row["end_date"]))
    _camp_orders = completed.loc[_camp_mask].copy()
    if _camp_orders.empty:
        st.info("この期間に完了注文がありません。")
    else:
        _pm = products[["sku_id", "product_name", "color", "size"]].copy()
        _pm["品番"] = (_pm["product_name"] + " " + _pm["color"].astype(str)
                       + "/" + _pm["size"].astype(str) + "（" + _pm["sku_id"] + "）")
        camp_prod = (_camp_orders.merge(_pm[["sku_id", "品番"]], on="sku_id", how="left")
                     .groupby("品番")["total_amount"].sum()
                     .sort_values(ascending=False).head(15)
                     .reset_index().rename(columns={"total_amount": "売上"}))
        fig_camp_prod = px.bar(camp_prod, x="売上", y="品番", orientation="h",
                               color_discrete_sequence=[COLORS["accent"]])
        fig_camp_prod.update_layout(yaxis_title="品番", xaxis_title="売上（円）",
                                    yaxis=dict(autorange="reversed"), height=520)
        st.plotly_chart(fig_camp_prod, use_container_width=True)

    st.subheader("全キャンペーン一覧")
    camp_disp = campaigns.copy()
    camp_disp["start_date"] = camp_disp["start_date"].dt.strftime("%Y/%m/%d")
    camp_disp["end_date"] = camp_disp["end_date"].dt.strftime("%Y/%m/%d")
    camp_disp["budget"] = camp_disp["budget"].apply(lambda v: f"¥{v:,}")
    camp_disp["actual_spend"] = camp_disp["actual_spend"].apply(lambda v: f"¥{v:,}")
    st.dataframe(camp_disp, use_container_width=True, hide_index=True)


# ============================================================
# 💰 収支管理
# ============================================================
elif page == "収支管理":
    page_header("収支管理", "月次PL（売上・コスト・利益）を管理します",
                key_prefix="pl",
                latest_date=orders["order_date"].max())

    pl = expenses.copy()
    # 広告費はクライアント側で別管理のため、このPLからは除外
    cost_cols = ["原価", "倉庫代", "外注費", "物流費", "決済手数料", "システム費"]
    pl["総コスト"] = pl[cost_cols].sum(axis=1)
    pl["営業利益"] = pl["売上"] - pl["総コスト"]
    pl["利益率"] = (pl["営業利益"] / pl["売上"] * 100).round(1)
    st.caption("※ 広告費は別管理のため、本シートのコスト・利益には含めていません。")

    # ── KPI（直近月） ──
    latest = pl.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("売上（直近月）", f"¥{latest['売上']:,.0f}")
    col2.metric("総コスト", f"¥{latest['総コスト']:,.0f}")
    col3.metric("営業利益", f"¥{latest['営業利益']:,.0f}",
                delta=f"{latest['利益率']}%")
    col4.metric("原価率", f"{latest['原価'] / latest['売上'] * 100:.1f}%")

    st.markdown("---")

    # ── 月次PL推移 ──
    st.subheader("月次 売上 / コスト / 利益 推移")
    fig_pl = go.Figure()
    fig_pl.add_trace(go.Bar(x=pl["month"], y=pl["売上"], name="売上",
                            marker_color=COLORS["accent"]))
    fig_pl.add_trace(go.Bar(x=pl["month"], y=-pl["総コスト"], name="総コスト",
                            marker_color="#E57373"))
    fig_pl.add_trace(go.Scatter(x=pl["month"], y=pl["営業利益"], name="営業利益",
                                mode="lines+markers",
                                line=dict(color=COLORS["primary"], width=3)))
    fig_pl.update_layout(barmode="relative", xaxis_title="月",
                         yaxis_title="金額（円）", xaxis_tickangle=-45,
                         hovermode="x unified")
    st.plotly_chart(fig_pl, use_container_width=True)

    # ── コスト内訳（積み上げ棒グラフ） ──
    st.subheader("コスト内訳 推移")
    cost_melt = pl.melt(id_vars="month", value_vars=cost_cols,
                        var_name="費目", value_name="金額")
    fig_cost = px.bar(cost_melt, x="month", y="金額", color="費目",
                      barmode="stack", title="月次コスト内訳",
                      color_discrete_sequence=["#36454F", "#455A64", "#546E7A",
                                               "#607D8B", "#78909C", "#90A4AE", "#B0BEC5"])
    fig_cost.update_layout(xaxis_title="月", yaxis_title="金額（円）", xaxis_tickangle=-45)
    st.plotly_chart(fig_cost, use_container_width=True)

    # ── 利益率推移 ──
    col_l, col_r = st.columns(2)
    fig_margin = px.line(pl, x="month", y="利益率", title="営業利益率 推移",
                         markers=True, color_discrete_sequence=[COLORS["primary"]])
    fig_margin.update_layout(xaxis_title="月", yaxis_title="利益率（%）", xaxis_tickangle=-45)
    fig_margin.add_hline(y=0, line_dash="dash", line_color="red", line_width=1)
    col_l.plotly_chart(fig_margin, use_container_width=True)

    # ── コスト構成比（直近月） ──
    latest_costs = pd.DataFrame({
        "費目": cost_cols,
        "金額": [latest[c] for c in cost_cols]
    })
    fig_pie = px.pie(latest_costs, names="費目", values="金額",
                     title=f"コスト構成比（{latest['month']}）",
                     color_discrete_sequence=COLORS["chart"] + ["#CFD8DC", "#ECEFF1"])
    col_r.plotly_chart(fig_pie, use_container_width=True)

    # ── PLテーブル（横=月、縦=科目）──
    st.subheader("月次PL一覧")

    row_items = ["売上"] + cost_cols + ["総コスト", "営業利益", "利益率"]
    pl_sorted = pl.sort_values("month").reset_index(drop=True)  # 古い→新しい（左→右）

    # 表示月数をスライダーで選択（デフォルト: 直近12ヶ月）
    n_months = len(pl_sorted)
    default_window = min(12, n_months)
    if n_months > default_window:
        window = st.slider("表示月数", min_value=3, max_value=n_months,
                            value=default_window, step=1,
                            help="直近から何ヶ月分を表示するか")
    else:
        window = n_months

    # 横スクロール用の開始位置スライダー（左=過去、右=直近）
    max_start = max(0, n_months - window)
    if max_start > 0:
        start_idx = st.slider("表示開始月（左＝過去）",
                               min_value=0, max_value=max_start,
                               value=max_start, step=1,
                               help="左にスライドするほど過去の月を表示")
    else:
        start_idx = 0

    pl_window = pl_sorted.iloc[start_idx:start_idx + window].copy()

    # 転置: 行=科目, 列=月
    pl_t = pl_window.set_index("month")[row_items].T
    pl_t.index.name = "科目"

    # フォーマット
    def _fmt(val, item):
        if item == "利益率":
            return f"{val}%"
        return f"¥{val:,.0f}"

    pl_display = pl_t.copy().astype(object)
    for item in pl_t.index:
        for col in pl_t.columns:
            pl_display.loc[item, col] = _fmt(pl_t.loc[item, col], item)

    st.dataframe(pl_display, use_container_width=True)


# ============================================================
# 営業管理（法人EC購入モニタリング）
# ============================================================
elif page == "営業管理":
    page_header("営業管理", "法人顧客のEC購入状況を営業部向けに可視化します",
                key_prefix="sales",
                latest_date=orders["order_date"].max())

    # ── 期間クランプ: 2026/6/15（新ECローンチ）以降のみ ──
    _latest = orders["order_date"].max()
    _period_label = st.session_state.get("sales_period", "6M")
    _months = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12}[_period_label]
    _window_start = (_latest - pd.DateOffset(months=_months)).replace(day=1)
    _period_start = max(_window_start, LAUNCH_DATE)  # 新ECローンチで下限クランプ

    if _latest < LAUNCH_DATE:
        alert_banner(
            f"新ECローンチ（{LAUNCH_DATE.strftime('%Y/%m/%d')}）以降の注文データが"
            f"まだありません。ローンチ後に法人購入状況がここに表示されます。",
            level="info",
        )
        st.stop()

    # ── 法人顧客のみ抽出 ──
    corp_ids = customers[customers["customer_type"] == "法人"]["customer_id"].tolist()
    corp_orders = orders[
        (orders["customer_id"].isin(corp_ids)) &
        (orders["order_date"] >= _period_start) &
        (orders["order_date"] <= _latest) &
        (orders["status"] == "完了")
    ].copy()

    # ── サマリーKPI ──
    n_corp      = corp_orders["customer_id"].nunique()
    corp_sales  = corp_orders["total_amount"].sum()
    avg_per_corp = (corp_sales / n_corp) if n_corp else 0
    n_orders_total = corp_orders["order_id"].nunique()
    avg_orders_per_corp = (n_orders_total / n_corp) if n_corp else 0

    kpi_row([
        {"label": "購入法人数",     "value": f"{n_corp:,}",              "color": "accent"},
        {"label": "法人売上",       "value": f"¥{corp_sales:,.0f}",      "color": "success"},
        {"label": "法人平均購入額", "value": f"¥{avg_per_corp:,.0f}",    "color": "warning"},
        {"label": "平均購入回数",   "value": f"{avg_orders_per_corp:.1f} 回", "color": "accent"},
    ])

    st.markdown("---")

    if n_corp == 0:
        alert_banner("選択期間内に法人の完了注文がありません。", level="info")
        st.stop()

    # ── 検索 + 並び替えコントロール ──
    ctrl_l, ctrl_r = st.columns([2, 3])
    with ctrl_l:
        search_q = st.text_input("法人名検索", "", key="sales_search",
                                  placeholder="例: 建設 / 工業 …")
    with ctrl_r:
        sort_by = st.radio(
            "並び替え",
            ["直近購入日", "累計売上", "購入回数", "法人名"],
            horizontal=True, key="sales_sort",
            label_visibility="collapsed",
        )

    # ── 法人別集計 ──
    # 商品マスタを結合（カテゴリ取得）
    oc = corp_orders.merge(
        products[["product_id", "product_name", "category"]],
        on="product_id", how="left",
    )

    agg_by_cust = oc.groupby("customer_id").agg(
        last_order    = ("order_date",  "max"),
        order_count   = ("order_id",    "nunique"),
        total_qty     = ("quantity",    "sum"),
        total_amount  = ("total_amount", "sum"),
    ).reset_index()
    agg_by_cust["法人名"] = agg_by_cust["customer_id"].map(corp_display_name)

    # 検索フィルタ
    if search_q.strip():
        q = search_q.strip()
        agg_by_cust = agg_by_cust[agg_by_cust["法人名"].str.contains(q, na=False)]

    # ソート
    sort_map = {
        "直近購入日": ("last_order",   False),
        "累計売上":   ("total_amount", False),
        "購入回数":   ("order_count",  False),
        "法人名":     ("法人名",       True),
    }
    col, asc = sort_map[sort_by]
    agg_by_cust = agg_by_cust.sort_values(col, ascending=asc).reset_index(drop=True)

    st.markdown(
        f'<div class="jcd-card__sub">該当 {len(agg_by_cust):,} 社 ／ '
        f'期間: {_period_start.strftime("%Y/%m/%d")} ～ '
        f'{_latest.strftime("%Y/%m/%d")}</div>',
        unsafe_allow_html=True,
    )

    # ── 法人行 + アコーディオン ──
    # ヘッダー行
    st.markdown(
        '<div class="jcd-sales-head">'
        '<div class="jcd-sales-head__cell jcd-sales-head__name">法人名</div>'
        '<div class="jcd-sales-head__cell">直近購入</div>'
        '<div class="jcd-sales-head__cell">購入回数</div>'
        '<div class="jcd-sales-head__cell">累計点数</div>'
        '<div class="jcd-sales-head__cell jcd-sales-head__amt">累計売上</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    for _, row in agg_by_cust.iterrows():
        cid   = row["customer_id"]
        cname = row["法人名"]
        summary_label = (
            f"**{cname}**　"
            f"｜ 直近: {row['last_order'].strftime('%Y/%m/%d')}　"
            f"｜ {int(row['order_count'])}回　"
            f"｜ {int(row['total_qty']):,}点　"
            f"｜ ¥{row['total_amount']:,.0f}"
        )
        with st.expander(summary_label):
            cust_orders = oc[oc["customer_id"] == cid].copy()

            # カテゴリ別サマリ
            cat = cust_orders.groupby("category").agg(
                点数=("quantity", "sum"),
                金額=("total_amount", "sum"),
            ).reset_index().sort_values("金額", ascending=False)

            st.markdown("**カテゴリ別 購入サマリ**")
            cat_disp = cat.copy()
            cat_disp["点数"] = cat_disp["点数"].map(lambda v: f"{int(v):,}")
            cat_disp["金額"] = cat_disp["金額"].map(lambda v: f"¥{v:,.0f}")
            cat_disp.columns = ["カテゴリ", "点数", "金額"]
            st.dataframe(cat_disp, hide_index=True, use_container_width=True)

            # 注文明細（日付降順）
            st.markdown("**注文明細**")
            detail = cust_orders.sort_values("order_date", ascending=False).copy()
            detail_disp = pd.DataFrame({
                "注文日":   detail["order_date"].dt.strftime("%Y/%m/%d"),
                "注文ID":   detail["order_id"],
                "商品名":   detail["product_name"],
                "カテゴリ": detail["category"],
                "点数":     detail["quantity"].map(lambda v: f"{int(v):,}"),
                "単価":     detail["unit_price"].map(lambda v: f"¥{v:,.0f}"),
                "小計":     detail["total_amount"].map(lambda v: f"¥{v:,.0f}"),
            })
            st.dataframe(detail_disp, hide_index=True, use_container_width=True)
