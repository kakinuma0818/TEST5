# main.py
import streamlit as st
import pandas as pd
import numpy as np
import itertools
import math
from datetime import datetime

# ---------------------
# 設定
# ---------------------
st.set_page_config(page_title="最終完全統合版 競馬投資アプリ", layout="wide")
st.markdown("<style>body{font-family: Helvetica, Arial, sans-serif;}</style>", unsafe_allow_html=True)

MIN_BET_UNIT = 100  # 最小掛け金単位

# ---------------------
# ユーティリティ関数
# ---------------------

def round_unit(x, unit=MIN_BET_UNIT):
    """100円単位に切り上げ（0 -> 0）"""
    if x <= 0:
        return 0
    return int(math.ceil(x / unit) * unit)

def geom_mean(vals):
    a = 1.0
    for v in vals:
        a *= max(0.001, v)
    return a ** (1.0 / len(vals))

def estimate_combo_odds(names, horses_df, bet_type):
    """
    組み合わせの推定オッズを返す（近似）。
    実運用では実オッズソースに置換してください。
    ロジック:
      - 単勝/複勝: 単純に馬のオッズ
      - 馬連/ワイド/馬単: 馬の幾何平均 * ペアファクタ
      - 3連複/3連単: 幾何平均 * トリプルファクタ
    """
    odds_list = []
    for n in names:
        row = horses_df[horses_df["馬名"] == n]
        if len(row)==0:
            odds_list.append(10.0)
        else:
            odds_list.append(float(row.iloc[0]["オッズ"]))
    gm = geom_mean(odds_list)
    if bet_type in ["単勝","複勝"]:
        # single picks only: use individual avg (but names is tuple of single)
        return float(odds_list[0])
    if bet_type in ["馬連","ワイド","馬単","枠連"]:
        # pair factor
        return max(1.0, gm * 1.6)
    if bet_type in ["3連複","3連単"]:
        return max(1.0, gm * 2.5)
    # fallback
    return gm

def generate_combinations_by_method(selected_names, bet_type, method, axis1=None, axis2=None, formation=None):
    """
    selected_names: list of horse names selected by user
    bet_type: string
    method: "単純"/"軸"/"軸2"/"ボックス"/"フォーメーション"
    axis1, axis2: for axis selections (single names or None)
    formation: dict with keys col1, col2, col3 lists for formation mode
    returns: list of tuples (combination)
    """
    names = selected_names[:]
    combos = []

    # helper
    def unique_tuples(lst):
        # ensure deterministic ordering
        seen = set()
        res = []
        for t in lst:
            tt = tuple(t)
            if tt not in seen:
                seen.add(tt)
                res.append(tuple(t))
        return res

    if bet_type in ["単勝","複勝"]:
        combos = [(n,) for n in names]

    elif bet_type in ["馬連","ワイド","枠連"]:
        if method == "ボックス":
            combos = list(itertools.combinations(names,2))
        elif method.startswith("軸"):
            if not axis1:
                return []
            combos = [(axis1, opponent) for opponent in names if opponent!=axis1]
        elif method == "フォーメーション" and formation:
            # formation for pair bets: col1 x col2
            col1 = formation.get("col1", [])
            col2 = formation.get("col2", [])
            combos = [(a,b) for a in col1 for b in col2 if a!=b]

    elif bet_type == "馬単":
        if method == "ボックス":
            pairs = list(itertools.permutations(names,2))
            combos = pairs
        elif method.startswith("軸"):
            if not axis1:
                return []
            combos = [(axis1, opponent) for opponent in names if opponent!=axis1]
        elif method == "フォーメーション" and formation:
            col1 = formation.get("col1", [])
            col2 = formation.get("col2", [])
            combos = [(a,b) for a in col1 for b in col2 if a!=b]

    elif bet_type in ["3連複","3連単"]:
        if method == "ボックス":
            if bet_type=="3連複":
                combos = list(itertools.combinations(names,3))
            else:
                combos = list(itertools.permutations(names,3))
        elif method == "軸":
            # axis1 single
            if not axis1:
                return []
            others = [n for n in names if n!=axis1]
            if bet_type=="3連複":
                combos = [tuple(sorted((axis1, *c))) for c in itertools.combinations(others,2)]
            else:  # 3連単: axis in first position
                combos = [(axis1, b, c) for (b,c) in itertools.permutations(others,2)]
        elif method == "軸2" or method=="軸2頭":
            if not axis1 or not axis2:
                return []
            others = [n for n in names if n not in (axis1, axis2)]
            if bet_type=="3連複":
                # axis1+axis2+one of others
                combos = [tuple(sorted((axis1, axis2, o))) for o in others]
            else:
                # for 3連単 with two-axis: permutations of axis positions with last from others
                combos = []
                for o in others:
                    # permutations of axis1,axis2 order then o
                    combos.append((axis1, axis2, o))
                    combos.append((axis2, axis1, o))
        elif method == "フォーメーション" and formation:
            # formation expects col1, col2, col3 lists
            c1 = formation.get("col1",[])
            c2 = formation.get("col2",[])
            c3 = formation.get("col3",[])
            combos = [(a,b,c) for a in c1 for b in c2 for c in c3 if len({a,b,c})==3]
            if bet_type=="3連複":
                # to make unique combos (orderless)
                combos = [tuple(sorted(c)) for c in combos]
                combos = unique_tuples([tuple(c) for c in combos])
    # dedupe and return
    combos = [tuple(x) for x in combos]
    # For 3連複 we want combinations (orderless) unique
    if bet_type=="3連複":
        uniq = set()
        new = []
        for c in combos:
            key = tuple(sorted(c))
            if key not in uniq:
                uniq.add(key)
                new.append(tuple(key))
        combos = new
    return combos

def allocate_by_target(combos, horses_df, total_investment, desired_odds, bet_type, tolerance=0.1):
    """
    combos: list of tuples of horse names
    horses_df: df with '馬名' and 'オッズ'
    desired_odds: multiplier (e.g.,1.5)
    Returns allocation list [(combo, bet_amount, estimated_return, note), ...] and totals
    Strategy:
      - Target payout H = total_investment * desired_odds
      - Initial idea: give each combo share = H / (estimated_odds(combo) * len(combos))
        -> gives per-combo bet that if that combo wins, payout ~ H
      - Round to MIN_BET_UNIT
      - Check total bet > total_investment, adjust proportionally downward (but ensure at least MIN_BET_UNIT)
      - If after adjustment some combos payout < H*(1-tolerance) then mark warning
    """
    results = []
    if not combos:
        return results, 0, 0

    H = total_investment * desired_odds
    # estimate odds for each combo
    est_odds = []
    for c in combos:
        odds_est = estimate_combo_odds(c, horses_df, bet_type)
        est_odds.append(max(0.01, odds_est))

    N = len(combos)
    # initial bet per combo that aims to return H if that combo hits
    raw_bets = []
    for o in est_odds:
        raw = H / (o * N)
        raw_bets.append(raw)

    # round to unit
    rounded = [round_unit(b) for b in raw_bets]

    total_bet = sum(rounded)
    # if total_bet > total_investment, scale down proportionally
    if total_bet > total_investment and total_bet>0:
        scale = total_investment / total_bet
        scaled = [max(MIN_BET_UNIT, round_unit(b*scale)) for b in rounded]
        # re-total
        rounded = scaled
        total_bet = sum(rounded)

    # final results with expected returns
    for combo, amt, o in zip(combos, rounded, est_odds):
        expected = amt * o
        note = ""
        if expected < H * (1 - tolerance):
            note = f"下回り（{expected:.0f} < {H*(1-tolerance):.0f}）"
        results.append({
            "組合せ": combo,
            "掛け金": int(amt),
            "推定オッズ": round(o,2),
            "期待払い戻し": round(expected,1),
            "注記": note
        })

    return results, total_bet, H

# ---------------------
# データ読み込み（CSV or デモ）
# ---------------------
@st.cache_data(ttl=300)
def load_data_from_csv(path=None):
    if path:
        try:
            df = pd.read_csv(path)
            # expect columns: 枠,馬番,馬名,性齢,斤量,体重,騎手,脚質,オッズ,人気順,スコア,AIスコア,印
            return df
        except Exception as e:
            st.warning(f"CSV読み込み失敗: {e} → デモデータ使用")
    # demo data
    demo = [
        {"枠":1,"馬番":1,"馬名":"馬A","性齢":"牡4","斤量":57,"体重":"500kg","騎手":"川田","脚質":"差し","オッズ":3.5,"人気順":1,"スコア":85,"AIスコア":88,"印":"◎"},
        {"枠":1,"馬番":2,"馬名":"馬B","性齢":"牝3","斤量":55,"体重":"480kg","騎手":"武豊","脚質":"逃げ","オッズ":5.0,"人気順":2,"スコア":78,"AIスコア":80,"印":"○"},
        {"枠":2,"馬番":3,"馬名":"馬C","性齢":"牡5","斤量":57,"体重":"510kg","騎手":"ルメール","脚質":"差し","オッズ":8.0,"人気順":3,"スコア":70,"AIスコア":75,"印":"▲"},
        {"枠":3,"馬番":4,"馬名":"馬D","性齢":"牡6","斤量":58,"体重":"490kg","騎手":"福永","脚質":"先行","オッズ":12.0,"人気順":4,"スコア":66,"AIスコア":65,"印":"×"},
    ]
    return pd.DataFrame(demo)

# try load CSV path from sidebar input
st.sidebar.markdown("#### (任意) CSVデータを指定")
csv_path = st.sidebar.text_input("race CSV path (or leave blank to use demo)", value="")
df_horses = load_data_from_csv(csv_path if csv_path.strip()!="" else None)

# ---------------------
# UI: レース選択（上部）
# ---------------------
col1, col2, col3, col4 = st.columns([2,2,2,1])
with col1:
    race_date = st.date_input("日付", datetime.today())
with col2:
    race_course = st.selectbox("競馬場", ["東京","中山","京都","阪神","小倉","中京","福島","新潟"])
with col3:
    race_number = st.selectbox("レース番号", list(range(1,13)))
with col4:
    if st.button("更新 🔄"):
        st.experimental_rerun()

st.markdown(f"### {race_date} {race_course} {race_number}R")
st.markdown("芝/ダート・距離・Gレース等はCSVまたは手動で追記してください。")

# ---------------------
# Tabs: 出馬表, スコア, 馬券, 基本情報, 成績, AI
# ---------------------
tabs = st.tabs(["出馬表","スコア","馬券","基本情報","成績","AI"])

# ---------------------
# 出馬表タブ
# ---------------------
with tabs[0]:
    st.subheader("出馬表")
    # ensure display order
    display_cols = ["枠","馬番","馬名","性齢","斤量","体重","騎手","脚質","オッズ","人気順","スコア","AIスコア","印"]
    df_show = df_horses.copy()
    # fill missing columns
    for c in display_cols:
        if c not in df_show.columns:
            df_show[c] = ""
    # reorder
    df_show = df_show[display_cols]
    # highlight top6 scoring and bolding small: streamlit doesn't support bold in dataframe easily
    st.write("（表示：枠・馬番・馬名は左に固定イメージ。実装環境で横スクロールします）")
    st.dataframe(df_show, use_container_width=True)

# ---------------------
# スコアタブ
# ---------------------
with tabs[1]:
    st.subheader("スコア（手動スコア -3〜+3 を設定）")
    df_score = df_horses.copy()
    # add expected score cols if not exist
    for col in ["年齢","血統","馬主","生産者","調教師","成績","競馬場","距離","脚質","枠","馬場"]:
        if col not in df_score.columns:
            df_score[col] = ""
    if "手動" not in df_score.columns:
        df_score["手動"] = 0
    manual_vals = []
    for i, row in df_score.iterrows():
        # selectbox for manual score
        v = st.selectbox(f"{row['馬名']} 手動", options=[-3,-2,-1,0,1,2,3], index=3, key=f"manual_{i}")
        manual_vals.append(v)
    df_score["手動"] = manual_vals
    # 合計スコア = 既存 'スコア' + 手動 + AI補正(任意)
    df_score["合計スコア"] = df_score.get("スコア",0) + df_score["手動"] + df_score.get("AIスコア",0)*0.0
    st.dataframe(df_score[["馬名","合計スコア","年齢","血統","騎手","馬主","生産者","調教師","成績","競馬場","距離","脚質","枠","馬場","手動"]], use_container_width=True)

# reflect manual sum back to df_horses (for main display)
for i, r in df_score.iterrows():
    df_horses.loc[df_horses["馬名"]==r["馬名"], "スコア"] = r["合計スコア"]

# ---------------------
# 馬券タブ
# ---------------------
with tabs[2]:
    st.subheader("馬券（BE） — 種類・買い方を選び、馬を指定して自動配分")
    # bet choices
    bet_types = ["単勝","複勝","馬連","馬単","ワイド","3連複","3連単","枠連"]
    bet_type = st.selectbox("馬券種類", bet_types, index=0)
    method_map = ["単純","軸","軸2","ボックス","フォーメーション"]
    method = st.selectbox("買い方", method_map, index=0)

    st.markdown("**馬を選択（チェック）**")
    # show as a table of checkboxes
    selected_names = []
    cols = st.columns(2)
    for idx, row in df_horses.iterrows():
        c = cols[idx % 2].checkbox(f"{row['馬番']}: {row['馬名']} (オッズ:{row['オッズ']})", key=f"sel_{idx}")
        if c:
            selected_names.append(row["馬名"])

    # axis selectors for axis/axis2/formation
    axis1 = None; axis2 = None; formation = None
    if method in ("軸","軸2","フォーメーション"):
        st.markdown("**軸 / フォーメーション指定**")
        if method in ("軸","軸2"):
            axis1 = st.selectbox("軸1を選択", [""] + selected_names, index=0, key="axis1")
            if method=="軸2":
                axis2 = st.selectbox("軸2を選択", [""] + selected_names, index=0, key="axis2")
        else:
            # formation: allow user to pick columns (col1, col2, col3) depending on bet_type
            st.markdown("フォーメーション列を作成してください（空にすると全選択とみなします）")
            if bet_type in ["3連複","3連単"]:
                col1 = st.multiselect("1列目（軸候補）", options=selected_names)
                col2 = st.multiselect("2列目（中位）", options=selected_names)
                col3 = st.multiselect("3列目（最終）", options=selected_names)
                formation = {"col1": col1 or selected_names, "col2": col2 or selected_names, "col3": col3 or selected_names}
            else:
                col1 = st.multiselect("1列目", options=selected_names)
                col2 = st.multiselect("2列目", options=selected_names)
                formation = {"col1": col1 or selected_names, "col2": col2 or selected_names}

    # investment inputs
    total_investment = st.number_input("総投資金額 (円)", min_value=100, step=100, value=1000, key="invest_total")
    desired_odds = st.number_input("希望払い戻し倍率 (例 1.5)", min_value=1.0, step=0.1, value=1.5, key="invest_target")
    tolerance_pct = st.slider("下回り許容率 (%)", min_value=0, max_value=50, value=10, step=1)/100.0

    # perform allocation
    if st.button("自動配分計算"):
        if len(selected_names) == 0:
            st.warning("購入する馬を少なくとも1頭選択してください。")
        else:
            combos = generate_combinations_by_method(selected_names, bet_type, method, axis1=axis1 if axis1!="" else None, axis2=axis2 if axis2!="" else None, formation=formation)
            if not combos:
                st.warning("指定方法により組合せが作れません。馬の選択/軸/フォーメーションを確認してください。")
            else:
                allocs, total_bet, target_payout = allocate_by_target(combos, df_horses, total_investment, desired_odds, bet_type, tolerance=tolerance_pct)
                df_alloc = pd.DataFrame(allocs)
                st.write(f"目標払い戻し額 H = 投資 {total_investment} × {desired_odds} = {target_payout:.0f} 円")
                st.write(f"合計掛け金（計算結果）: {total_bet} 円 (入力投資: {total_investment} 円)")
                # show warnings if total_bet > input
                if total_bet > total_investment:
                    st.warning("注意：計算上の合計掛け金が入力投資額を超えています。投資額を増やすか希望倍率を下げてください。")
                # show allocation table
                st.dataframe(df_alloc, use_container_width=True)

# ---------------------
# 基本情報タブ
# ---------------------
with tabs[3]:
    st.subheader("基本情報（PR）")
    cols = ["馬名","性齢","血統","馬主","生産者","調教師"]
    for c in cols:
        if c not in df_horses.columns:
            df_horses[c] = ""
    st.dataframe(df_horses[cols], use_container_width=True)

# ---------------------
# 成績タブ
# ---------------------
with tabs[4]:
    st.subheader("成績（GR）")
    if "過去成績" not in df_horses.columns:
        df_horses["過去成績"] = ""
    st.dataframe(df_horses[["馬名","過去成績"]], use_container_width=True)

# ---------------------
# AIタブ
# ---------------------
with tabs[5]:
    st.subheader("AI予想")
    # ensure AI factors exist
    factors = ["スピード","スタミナ","パワー","展開適正","距離適正","馬場適正"]
    for f in factors:
        if f not in df_horses.columns:
            df_horses[f] = np.random.randint(50,90,len(df_horses))
    # calc AI score
    df_horses["AI合計"] = df_horses[factors].sum(axis=1)
    df_ai = df_horses.sort_values("AI合計", ascending=False).reset_index(drop=True)
    df_ai["AI順位"] = df_ai.index + 1
    st.dataframe(df_ai[["馬名","AI合計","AI順位"] + factors], use_container_width=True)

# ---------------------
# フッターメッセージ
# ---------------------
st.markdown("---")
st.markdown("**注意**：組合せオッズ・払い戻し額は推定値です。実際のオッズを取得するAPI/スクレイピング接続を推奨します。")
st.markdown("データの永続化・自動更新・オッズのリアルタイム反映は次ステップで追加できます。")
