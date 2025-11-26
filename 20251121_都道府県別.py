import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np  # ★ 数値変換などで使用
import re           # ★ 滞在日数の文字列パースに使用

# ★ このダッシュボードで使っているデータ期間
DATA_PERIOD = "2024年4〜6月期インバウンド個票データ"

st.set_page_config(
    page_title=f"観光客分析ダッシュボード（{DATA_PERIOD}）",
    layout="wide"
)

# =========================================================
# ヘルパー：滞在日数の文字列 → 日数(float)
# =========================================================
def parse_stay_days(x):
    """
    例:
      '3日'        -> 3
      '3日間'      -> 3
      '3〜4日'     -> 3.5
      '1か月未満'  -> 30
      '2か月'      -> 60
      '1年未満'    -> 365
      '7'          -> 7
      '日間'       -> NaN
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if s == "":
        return np.nan

    # 全て数字だけならそのまま
    if s.isdigit():
        return float(s)

    # 数字を全部抜き出す
    nums = re.findall(r"\d+", s)
    nums_f = [float(n) for n in nums] if nums else []

    # 「年」が含まれる → おおざっぱに 1年=365日で換算
    if "年" in s:
        if nums_f:
            return nums_f[0] * 365.0
        else:
            return 365.0

    # 「月」が含まれる → 1か月=30日で換算
    if "月" in s:
        if nums_f:
            return nums_f[0] * 30.0
        else:
            return 30.0

    # 「〜」や「-」でレンジ表現がある場合は平均を取る
    if "〜" in s or "～" in s or "-" in s:
        if nums_f:
            return sum(nums_f) / len(nums_f)

    # 「日」が含まれていて数字がある → 最初の数字を採用
    if "日" in s and nums_f:
        return nums_f[0]

    # それ以外は一旦 NaN にしておく
    if nums_f:
        return nums_f[0]

    return np.nan


# =========================================================
# ヘルパー：居住地カラム名を柔軟に取得
#   - 「居住地」
#   - 「居住地・地域」
#   - それ以外でも「居住」という文字を含む列
# =========================================================
def get_res_col(df: pd.DataFrame):
    if "居住地" in df.columns:
        return "居住地"
    if "居住地・地域" in df.columns:
        return "居住地・地域"
    for c in df.columns:
        if "居住" in c:
            return c
    return None


# =========================================================
# データ読み込み
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("df_route2.csv")

    # 不要列削除
    for col in ["Unnamed: 0", "Unnamed: 0.1"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 都道府県そろえる
    if "都道府県" not in df.columns:
        cand = None
        for c in df.columns:
            if c != "都道府県コード" and "都道府県" in c:
                cand = c
                break
        if cand is not None:
            df = df.rename(columns={cand: "都道府県"})
        else:
            st.error("`都道府県` 列が見つかりませんでした。現在の列名: " + ", ".join(df.columns))
            return df

    # （任意）居住地カラムを「居住地」に揃える試み
    #   → うまく行かなくても get_res_col が拾うのでここはオマケ
    res_col = get_res_col(df)
    if res_col is not None and res_col != "居住地":
        df = df.rename(columns={res_col: "居住地"})

    # 総支出（円／人）を作る
    big_exp_cols = [
        "宿泊費（円／人）",
        "飲食費（円／人）",
        "交通費（円／人）",
        "娯楽等サービス費（円／人）",
        "買物代（円／人）",
        "その他費目（円／人）",
    ]

    # 金額列の前処理：カンマ除去 → 数値化
    for c in big_exp_cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace(["", "nan", "NaN"], np.nan)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = 0.0

    if "総支出（円／人）" in df.columns:
        df["総支出（円／人）"] = (
            df["総支出（円／人）"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace(["", "nan", "NaN"], np.nan)
        )
        df["総支出（円／人）"] = pd.to_numeric(df["総支出（円／人）"], errors="coerce")
    else:
    # big_exp_cols に入っている6つの費目を行方向に合計して「総支出」を作る
        df["総支出（円／人）"] = df[big_exp_cols].sum(axis=1)



    # 滞在日数をクレンジングして数値列を作る
    if "滞在日数" in df.columns:
        df["滞在日数_clean"] = df["滞在日数"].apply(parse_stay_days)
    else:
        df["滞在日数_clean"] = np.nan

    # ここから：1日あたりの金額を作る（円／人・日）
    # 日数のベース：滞在日数_clean → それがなければ泊数
    days = None
    if "滞在日数_clean" in df.columns:
        days = df["滞在日数_clean"]
    elif "泊数" in df.columns:
        days = pd.to_numeric(df["泊数"], errors="coerce")

    if days is not None:
        days = days.replace(0, np.nan)

        target_cols = big_exp_cols + ["総支出（円／人）"]
        for col in target_cols:
            if col in df.columns:
                new_col = col.replace("（円／人）", "（円／人・日）")
                df[new_col] = df[col] / days

    return df


# =========================================================
# ① 都道府県別ダッシュボード
# =========================================================
def show_pref_dashboard(df: pd.DataFrame):
    st.title("都道府県別ダッシュボード")
    st.caption(f"データ期間：{DATA_PERIOD}")

    pref_list = sorted(df["都道府県"].dropna().unique())
    target_pref = st.selectbox("都道府県を選んでください", pref_list)

    df_pref = df[df["都道府県"] == target_pref].copy()

    st.subheader(f"{target_pref} を訪れた観光客の傾向")

    # -----------------------------
    # 中段：国籍・性年代・同行者
    # -----------------------------
    st.subheader("国籍・性年代・同行者")

    col_a, col_b = st.columns(2)

    if "国籍・地域" in df_pref.columns:
        nat = (
            df_pref.groupby("国籍・地域")
            .size()
            .reset_index(name="件数")
            .sort_values("件数", ascending=False)
        )
        fig_nat = px.bar(
            nat.head(15),
            x="国籍・地域",
            y="件数",
            title="国籍別 延べ訪問件数（上位15）"
        )
        col_a.plotly_chart(fig_nat, use_container_width=True)

    if "性年代" in df_pref.columns:
        age = (
            df_pref.groupby("性年代")
            .size()
            .reset_index(name="件数")
            .sort_values("件数", ascending=False)
        )
        fig_age = px.bar(
            age,
            x="性年代",
            y="件数",
            title="性年代別 延べ訪問件数"
        )
        col_b.plotly_chart(fig_age, use_container_width=True)

    if "同行者" in df_pref.columns:
        comp = (
            df_pref.groupby("同行者")
            .size()
            .reset_index(name="件数")
            .sort_values("件数", ascending=False)
        )
        fig_comp = px.bar(
            comp,
            x="同行者",
            y="件数",
            title="同行者タイプ別 延べ訪問件数"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # 行程 & 支出構成
    # -----------------------------
    st.subheader("行程のタイミング（この都道府県が何箇所目で訪問されているか）")
    if "行程" in df_pref.columns:
        route_order = [
            "入国",
            "1箇所目", "2箇所目", "3箇所目", "4箇所目", "5箇所目",
            "6箇所目", "7箇所目", "8箇所目", "9箇所目", "10箇所目",
            "11箇所目", "12箇所目", "13箇所目",
            "出国",
        ]
        df_pref["行程"] = pd.Categorical(df_pref["行程"], categories=route_order, ordered=True)

        route = (
            df_pref.groupby("行程")
            .size()
            .reset_index(name="件数")
            .sort_values("行程")
        )
        fig_route = px.bar(
            route,
            x="行程",
            y="件数",
            title="行程別 延べ訪問件数"
        )
        st.plotly_chart(fig_route, use_container_width=True)

    st.subheader("平均支出内訳（1人あたり）")

    use_per_day = st.checkbox("1日あたり（円／人・日）で見る", value=False)

    exp_map = {
        "宿泊費（円／人）": "宿泊",
        "飲食費（円／人）": "飲食",
        "交通費（円／人）": "交通",
        "娯楽等サービス費（円／人）": "娯楽",
        "買物代（円／人）": "買物",
        "その他費目（円／人）": "その他",
    }

    rows = []
    actually_used_cols = []

    for base_col, label in exp_map.items():
        per_day_col = base_col.replace("（円／人）", "（円／人・日）")

        if use_per_day and per_day_col in df_pref.columns:
            s = pd.to_numeric(df_pref[per_day_col], errors="coerce")
            if not s.dropna().empty:
                rows.append((label, s.mean()))
                actually_used_cols.append(per_day_col)
                continue

        if base_col in df_pref.columns:
            s = pd.to_numeric(df_pref[base_col], errors="coerce")
            if not s.dropna().empty:
                rows.append((label, s.mean()))
                actually_used_cols.append(base_col)

    if rows:
        exp_df = pd.DataFrame(rows, columns=["カテゴリ", "平均額"])

        if any("・日" in c for c in actually_used_cols):
            unit = "円／人・日"
        else:
            unit = "円／人"

        fig_exp = px.bar(
            exp_df,
            x="カテゴリ",
            y="平均額",
            title=f"平均支出内訳（{unit}）"
        )
        fig_exp.update_yaxes(tickformat=",")
        fig_exp.update_traces(hovertemplate="%{x}: %{y:,.0f} 円")
        st.plotly_chart(fig_exp, use_container_width=True)

        if use_per_day and unit == "円／人":
            st.info("この都道府県では、1日あたりの支出が計算できなかったため、総額（円／人）で表示しています。")
    else:
        st.write("支出関連の列が見つかりませんでした。")

    st.markdown("---")

    st.subheader("行程 × 利用宿泊施設：何箇所目でどの宿に泊まっているか")

    if "行程" in df_pref.columns and "利用宿泊施設" in df_pref.columns:
        tmp = df_pref[["行程", "利用宿泊施設"]].dropna().copy()

        route_order = [
            "入国",
            "1箇所目", "2箇所目", "3箇所目", "4箇所目", "5箇所目",
            "6箇所目", "7箇所目", "8箇所目", "9箇所目", "10箇所目",
            "11箇所目", "12箇所目", "13箇所目",
            "出国",
        ]
        tmp["行程"] = pd.Categorical(tmp["行程"], categories=route_order, ordered=True)

        combo = (
            tmp
            .groupby(["行程", "利用宿泊施設"])
            .size()
            .reset_index(name="件数")
            .sort_values(["行程", "件数"], ascending=[True, False])
        )

        st.caption("この都道府県を訪れたときの『何箇所目 × 利用宿泊施設』の組み合わせ頻度")
        st.dataframe(combo, use_container_width=True)

        top_n = st.slider("グラフに表示する組み合わせ数（件数上位）", 5, 50, 20)
        combo_top = combo.head(top_n)

        fig_combo = px.bar(
            combo_top,
            x="行程",
            y="件数",
            color="利用宿泊施設",
            title="行程 × 利用宿泊施設（件数上位）",
            hover_data=["利用宿泊施設"],
        )
        fig_combo.update_xaxes(categoryorder="array", categoryarray=route_order)
        st.plotly_chart(fig_combo, use_container_width=True)
    else:
        st.info("行程 または 利用宿泊施設 の列が見つかりませんでした。")


# =========================================================
# ② 国籍×属性ダッシュボード
# （ここは変更なし・そのまま）
# =========================================================
def show_segment_dashboard(df: pd.DataFrame):
    st.title("国籍×属性ダッシュボード")
    st.caption(f"データ期間：{DATA_PERIOD}")

    st.sidebar.markdown("### セグメント条件")

    nat = st.sidebar.selectbox(
        "国籍・地域",
        sorted(df["国籍・地域"].dropna().unique())
    )
    df_seg = df[df["国籍・地域"] == nat].copy()

    age_options = ["すべて"] + sorted(df_seg["性年代"].dropna().unique())
    age = st.sidebar.selectbox("性年代", age_options)
    if age != "すべて":
        df_seg = df_seg[df_seg["性年代"] == age]

    comp_options = ["すべて"] + sorted(df_seg["同行者"].dropna().unique())
    comp = st.sidebar.selectbox("同行者", comp_options)
    if comp != "すべて":
        df_seg = df_seg[df_seg["同行者"] == comp]

    pref_options = ["すべて"] + sorted(df_seg["都道府県"].dropna().unique())
    pref = st.sidebar.selectbox("都道府県（任意）", pref_options)
    if pref != "すべて":
        df_seg = df_seg[df_seg["都道府県"] == pref]

    st.subheader("現在のセグメント（国籍ベース）")
    st.write(f"- 国籍・地域：**{nat}**")
    st.write(f"- 性年代　　：**{age}**")
    st.write(f"- 同行者　　：**{comp}**")
    st.write(f"- 都道府県　：**{pref}**")

    if df_seg.empty:
        st.warning("この条件に一致するデータがありません。条件を少し緩めてみてください。")
        return

    n = len(df_seg)

    avg_nights = None
    if "泊数" in df_seg.columns:
        nights = pd.to_numeric(df_seg["泊数"], errors="coerce")
        if not nights.dropna().empty:
            avg_nights = nights.mean()

    avg_stay = None
    if "滞在日数_clean" in df_seg.columns:
        stay = pd.to_numeric(df_seg["滞在日数_clean"], errors="coerce")
        if not stay.dropna().empty:
            avg_stay = stay.mean()
    elif "滞在日数" in df_seg.columns:
        stay = pd.to_numeric(df_seg["滞在日数"], errors="coerce")
        if not stay.dropna().empty:
            avg_stay = stay.mean()

    avg_total = None
    if "総支出（円／人）" in df_seg.columns:
        total = pd.to_numeric(df_seg["総支出（円／人）"], errors="coerce")
        if not total.dropna().empty:
            avg_total = total.mean()

    st.markdown("### このセグメントの概要")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("サンプル数", f"{n:,} 件")
    if avg_nights is not None:
        c2.metric("平均泊数", f"{avg_nights:.1f} 泊")
    if avg_stay is not None:
        c3.metric("平均滞在日数", f"{avg_stay:.1f} 日")
    if avg_total is not None:
        c4.metric("平均総支出", f"{avg_total:,.0f} 円／人")

    st.markdown("---")

    # ここから：平均支出内訳（1人あたり） --------------------
    st.subheader("平均支出内訳（1人あたり, 国籍ベース）")

    use_per_day = st.checkbox(
        "1日あたり（円／人・日）で見る（国籍ベース）",
        value=False,
        key="seg_use_per_day_nat"
    )

    exp_map = {
        "宿泊費（円／人）": "宿泊",
        "飲食費（円／人）": "飲食",
        "交通費（円／人）": "交通",
        "娯楽等サービス費（円／人）": "娯楽",
        "買物代（円／人）": "買物",
        "その他費目（円／人）": "その他",
    }

    rows = []
    actually_used_cols = []

    for base_col, label in exp_map.items():
        per_day_col = base_col.replace("（円／人）", "（円／人・日）")

        # 1日あたりを優先
        if use_per_day and per_day_col in df_seg.columns:
            s = pd.to_numeric(df_seg[per_day_col], errors="coerce")
            if not s.dropna().empty:
                rows.append((label, s.mean()))
                actually_used_cols.append(per_day_col)
                continue

        # なければ総額（円／人）
        if base_col in df_seg.columns:
            s = pd.to_numeric(df_seg[base_col], errors="coerce")
            if not s.dropna().empty:
                rows.append((label, s.mean()))
                actually_used_cols.append(base_col)

    if rows:
        exp_df = pd.DataFrame(rows, columns=["カテゴリ", "平均額"])

        if any("・日" in c for c in actually_used_cols):
            unit = "円／人・日"
        else:
            unit = "円／人"

        fig_exp = px.bar(
            exp_df,
            x="カテゴリ",
            y="平均額",
            title=f"平均支出内訳（{unit}, このセグメント, 国籍ベース）",
        )
        fig_exp.update_yaxes(tickformat=",")
        fig_exp.update_traces(hovertemplate="%{x}: %{y:,.0f} 円")
        st.plotly_chart(fig_exp, use_container_width=True)

        if use_per_day and unit == "円／人":
            st.info("このセグメントでは 1日あたり列が見つからなかったため、総額（円／人）で表示しています。")

        with st.expander("支出内訳の数値も見る（国籍ベース）"):
            st.dataframe(exp_df)
    else:
        st.write("支出関連の列が見つかりませんでした。")

    st.markdown("---")

    # ここから下は元のまま（都道府県・泊数などの分布） --------------------
    st.subheader("このセグメントが訪れている都道府県")

    pref_counts = (
        df_seg["都道府県"]
        .value_counts()
        .reset_index()
    )
    pref_counts.columns = ["都道府県", "件数"]

    fig_pref = px.bar(
        pref_counts,
        x="都道府県",
        y="件数",
        title="訪問都道府県ランキング（このセグメント, 国籍ベース）"
    )
    st.plotly_chart(fig_pref, use_container_width=True)
    st.dataframe(pref_counts)

    st.markdown("---")

    st.subheader("宿泊日数・来訪回数・来訪目的の分布（国籍ベース）")

    # 泊数の分布（ヒストグラム）
    if "泊数" in df_seg.columns:
        nights = pd.to_numeric(df_seg["泊数"], errors="coerce").dropna()
        if not nights.empty:
            bins_nights = st.slider(
                "泊数ヒストグラムの階級数（bins, 国籍ベース）",
                min_value=5,
                max_value=40,
                value=15
            )
            fig_nights = px.histogram(
                x=nights,
                nbins=bins_nights,
                title=f"泊数の分布（bins={bins_nights}, 国籍ベース）"
            )
            fig_nights.update_xaxes(title="泊数")
            st.plotly_chart(fig_nights, use_container_width=True)

            freq_nights = (
                nights.value_counts()
                      .sort_index()
                      .reset_index()
            )
            freq_nights.columns = ["泊数", "件数"]
            with st.expander("泊数の度数分布表を見る（国籍ベース）"):
                st.dataframe(freq_nights, use_container_width=True)

    # 滞在日数の分布（滞在日数_clean）
    if "滞在日数_clean" in df_seg.columns:
        stay = pd.to_numeric(df_seg["滞在日数_clean"], errors="coerce").dropna()
        if not stay.empty:
            bins_stay = st.slider(
                "滞在日数ヒストグラムの階級数（bins, 国籍ベース）",
                min_value=5,
                max_value=40,
                value=15
            )
            fig_stay = px.histogram(
                x=stay,
                nbins=bins_stay,
                title=f"滞在日数の分布（bins={bins_stay}, 国籍ベース）"
            )
            fig_stay.update_xaxes(title="滞在日数（日）")
            st.plotly_chart(fig_stay, use_container_width=True)

            freq_stay = (
                stay.value_counts()
                    .sort_index()
                    .reset_index()
            )
            freq_stay.columns = ["滞在日数（日）", "件数"]
            with st.expander("滞在日数の度数分布表を見る（国籍ベース）"):
                st.dataframe(freq_stay, use_container_width=True)

    col_left, col_right = st.columns(2)

    if "日本への来訪回数" in df_seg.columns:
        vc = (
            df_seg["日本への来訪回数"]
            .value_counts(dropna=False)
            .reset_index()
        )
        vc.columns = ["日本への来訪回数", "件数"]
        fig_jp_visit = px.bar(
            vc,
            x="日本への来訪回数",
            y="件数",
            title="日本への来訪回数の分布（国籍ベース）"
        )
        col_left.plotly_chart(fig_jp_visit, use_container_width=True)

    if "年間来訪回数" in df_seg.columns:
        vc = (
            df_seg["年間来訪回数"]
            .value_counts(dropna=False)
            .reset_index()
        )
        vc.columns = ["年間来訪回数", "件数"]
        fig_year_visit = px.bar(
            vc,
            x="年間来訪回数",
            y="件数",
            title="年間来訪回数の分布（国籍ベース）"
        )
        col_right.plotly_chart(fig_year_visit, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    if "前回来訪時期" in df_seg.columns:
        vc = (
            df_seg["前回来訪時期"]
            .value_counts(dropna=False)
            .reset_index()
        )
        vc.columns = ["前回来訪時期", "件数"]
        fig_prev = px.bar(
            vc,
            x="前回来訪時期",
            y="件数",
            title="前回来訪時期の分布（国籍ベース）"
        )
        col_left2.plotly_chart(fig_prev, use_container_width=True)

    if "主な来訪目的" in df_seg.columns:
        vc = (
            df_seg["主な来訪目的"]
            .value_counts(dropna=False)
            .reset_index()
        )
        vc.columns = ["主な来訪目的", "件数"]
        fig_purpose = px.bar(
            vc,
            x="主な来訪目的",
            y="件数",
            title="主な来訪目的の分布（国籍ベース）"
        )
        col_right2.plotly_chart(fig_purpose, use_container_width=True)

    st.markdown("---")

    with st.expander("このセグメントの生データを見る（国籍ベース）"):
        st.dataframe(df_seg)


# =========================================================
# ②' 居住地×属性ダッシュボード（新規・ここを修正）
# =========================================================
def show_residence_segment_dashboard(df: pd.DataFrame):
    # ★ 居住地カラム名を柔軟に取得
    res_col = get_res_col(df)
    if res_col is None:
        st.error("居住地（または居住地・地域）に相当する列が見つかりません。")
        return

    st.title("居住地×属性ダッシュボード")
    st.caption(f"データ期間：{DATA_PERIOD}")

    st.sidebar.markdown("### セグメント条件（居住地ベース）")

    res = st.sidebar.selectbox(
        "居住地",
        sorted(df[res_col].dropna().unique())
    )
    df_seg = df[df[res_col] == res].copy()

    age_options = ["すべて"] + sorted(df_seg["性年代"].dropna().unique())
    age = st.sidebar.selectbox("性年代", age_options)
    if age != "すべて":
        df_seg = df_seg[df_seg["性年代"] == age]

    comp_options = ["すべて"] + sorted(df_seg["同行者"].dropna().unique())
    comp = st.sidebar.selectbox("同行者", comp_options)
    if comp != "すべて":
        df_seg = df_seg[df_seg["同行者"] == comp]

    pref_options = ["すべて"] + sorted(df_seg["都道府県"].dropna().unique())
    pref = st.sidebar.selectbox("都道府県（任意）", pref_options)
    if pref != "すべて":
        df_seg = df_seg[df_seg["都道府県"] == pref]

    st.subheader("現在のセグメント（居住地ベース）")
    st.write(f"- 居住地　　：**{res}**")
    st.write(f"- 性年代　　：**{age}**")
    st.write(f"- 同行者　　：**{comp}**")
    st.write(f"- 都道府県　：**{pref}**")

    if df_seg.empty:
        st.warning("この条件に一致するデータがありません。条件を少し緩めてみてください。")
        return

    n = len(df_seg)

    avg_nights = None
    if "泊数" in df_seg.columns:
        nights = pd.to_numeric(df_seg["泊数"], errors="coerce")
        if not nights.dropna().empty:
            avg_nights = nights.mean()

    avg_stay = None
    if "滞在日数_clean" in df_seg.columns:
        stay = pd.to_numeric(df_seg["滞在日数_clean"], errors="coerce")
        if not stay.dropna().empty:
            avg_stay = stay.mean()
    elif "滞在日数" in df_seg.columns:
        stay = pd.to_numeric(df_seg["滞在日数"], errors="coerce")
        if not stay.dropna().empty:
            avg_stay = stay.mean()

    avg_total = None
    if "総支出（円／人）" in df_seg.columns:
        total = pd.to_numeric(df_seg["総支出（円／人）"], errors="coerce")
        if not total.dropna().empty:
            avg_total = total.mean()

    st.markdown("### このセグメントの概要（居住地ベース）")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("サンプル数", f"{n:,} 件")
    if avg_nights is not None:
        c2.metric("平均泊数", f"{avg_nights:.1f} 泊")
    if avg_stay is not None:
        c3.metric("平均滞在日数", f"{avg_stay:.1f} 日")
    if avg_total is not None:
        c4.metric("平均総支出", f"{avg_total:,.0f} 円／人")

    st.markdown("---")

    # ここから：平均支出内訳（1人あたり, 居住地ベース） ---------
    st.subheader("平均支出内訳（1人あたり, 居住地ベース）")

    use_per_day = st.checkbox(
        "1日あたり（円／人・日）で見る（居住地ベース）",
        value=False,
        key="seg_use_per_day_res"
    )

    exp_map = {
        "宿泊費（円／人）": "宿泊",
        "飲食費（円／人）": "飲食",
        "交通費（円／人）": "交通",
        "娯楽等サービス費（円／人）": "娯楽",
        "買物代（円／人）": "買物",
        "その他費目（円／人）": "その他",
    }

    rows = []
    actually_used_cols = []

    for base_col, label in exp_map.items():
        per_day_col = base_col.replace("（円／人）", "（円／人・日）")

        if use_per_day and per_day_col in df_seg.columns:
            s = pd.to_numeric(df_seg[per_day_col], errors="coerce")
            if not s.dropna().empty:
                rows.append((label, s.mean()))
                actually_used_cols.append(per_day_col)
                continue

        if base_col in df_seg.columns:
            s = pd.to_numeric(df_seg[base_col], errors="coerce")
            if not s.dropna().empty:
                rows.append((label, s.mean()))
                actually_used_cols.append(base_col)

    if rows:
        exp_df = pd.DataFrame(rows, columns=["カテゴリ", "平均額"])

        if any("・日" in c for c in actually_used_cols):
            unit = "円／人・日"
        else:
            unit = "円／人"

        fig_exp = px.bar(
            exp_df,
            x="カテゴリ",
            y="平均額",
            title=f"平均支出内訳（{unit}, このセグメント, 居住地ベース）",
        )
        fig_exp.update_yaxes(tickformat=",")
        fig_exp.update_traces(hovertemplate="%{x}: %{y:,.0f} 円")
        st.plotly_chart(fig_exp, use_container_width=True)

        if use_per_day and unit == "円／人":
            st.info("このセグメントでは 1日あたり列が見つからなかったため、総額（円／人）で表示しています。")

        with st.expander("支出内訳の数値も見る（居住地ベース）"):
            st.dataframe(exp_df)
    else:
        st.write("支出関連の列が見つかりませんでした。")

    st.markdown("---")

    # ここから下は元のまま --------------------------
    st.subheader("このセグメントが訪れている都道府県（居住地ベース）")

    pref_counts = (
        df_seg["都道府県"]
        .value_counts()
        .reset_index()
    )
    pref_counts.columns = ["都道府県", "件数"]

    fig_pref = px.bar(
        pref_counts,
        x="都道府県",
        y="件数",
        title="訪問都道府県ランキング（このセグメント, 居住地ベース）"
    )
    st.plotly_chart(fig_pref, use_container_width=True)
    st.dataframe(pref_counts)

    st.markdown("---")

    st.subheader("宿泊日数・来訪回数・来訪目的の分布（居住地ベース）")

    # 泊数の分布
    if "泊数" in df_seg.columns:
        nights = pd.to_numeric(df_seg["泊数"], errors="coerce").dropna()
        if not nights.empty:
            bins_nights = st.slider(
                "泊数ヒストグラムの階級数（bins, 居住地ベース）",
                min_value=5,
                max_value=40,
                value=15
            )
            fig_nights = px.histogram(
                x=nights,
                nbins=bins_nights,
                title=f"泊数の分布（bins={bins_nights}, 居住地ベース）"
            )
            fig_nights.update_xaxes(title="泊数")
            st.plotly_chart(fig_nights, use_container_width=True)

            freq_nights = (
                nights.value_counts()
                      .sort_index()
                      .reset_index()
            )
            freq_nights.columns = ["泊数", "件数"]
            with st.expander("泊数の度数分布表を見る（居住地ベース）"):
                st.dataframe(freq_nights, use_container_width=True)

    # 滞在日数の分布（滞在日数_clean）
    if "滞在日数_clean" in df_seg.columns:
        stay = pd.to_numeric(df_seg["滞在日数_clean"], errors="coerce").dropna()
        if not stay.empty:
            bins_stay = st.slider(
                "滞在日数ヒストグラムの階級数（bins, 居住地ベース）",
                min_value=5,
                max_value=40,
                value=15
            )
            fig_stay = px.histogram(
                x=stay,
                nbins=bins_stay,
                title=f"滞在日数の分布（bins={bins_stay}, 居住地ベース）"
            )
            fig_stay.update_xaxes(title="滞在日数（日）")
            st.plotly_chart(fig_stay, use_container_width=True)

            freq_stay = (
                stay.value_counts()
                    .sort_index()
                    .reset_index()
            )
            freq_stay.columns = ["滞在日数（日）", "件数"]
            with st.expander("滞在日数の度数分布表を見る（居住地ベース）"):
                st.dataframe(freq_stay, use_container_width=True)

    col_left, col_right = st.columns(2)

    if "日本への来訪回数" in df_seg.columns:
        vc = (
            df_seg["日本への来訪回数"]
            .value_counts(dropna=False)
            .reset_index()
        )
        vc.columns = ["日本への来訪回数", "件数"]
        fig_jp_visit = px.bar(
            vc,
            x="日本への来訪回数",
            y="件数",
            title="日本への来訪回数の分布（居住地ベース）"
        )
        col_left.plotly_chart(fig_jp_visit, use_container_width=True)

    if "年間来訪回数" in df_seg.columns:
        vc = (
            df_seg["年間来訪回数"]
            .value_counts(dropna=False)
            .reset_index()
        )
        vc.columns = ["年間来訪回数", "件数"]
        fig_year_visit = px.bar(
            vc,
            x="年間来訪回数",
            y="件数",
            title="年間来訪回数の分布（居住地ベース）"
        )
        col_right.plotly_chart(fig_year_visit, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    if "前回来訪時期" in df_seg.columns:
        vc = (
            df_seg["前回来訪時期"]
            .value_counts(dropna=False)
            .reset_index()
        )
        vc.columns = ["前回来訪時期", "件数"]
        fig_prev = px.bar(
            vc,
            x="前回来訪時期",
            y="件数",
            title="前回来訪時期の分布（居住地ベース）"
        )
        col_left2.plotly_chart(fig_prev, use_container_width=True)

    if "主な来訪目的" in df_seg.columns:
        vc = (
            df_seg["主な来訪目的"]
            .value_counts(dropna=False)
            .reset_index()
        )
        vc.columns = ["主な来訪目的", "件数"]
        fig_purpose = px.bar(
            vc,
            x="主な来訪目的",
            y="件数",
            title="主な来訪目的の分布（居住地ベース）"
        )
        col_right2.plotly_chart(fig_purpose, use_container_width=True)

    st.markdown("---")

    with st.expander("このセグメントの生データを見る（居住地ベース）"):
        st.dataframe(df_seg)


# =========================================================
# ③ 国籍比較ダッシュボード（元のまま）
# =========================================================
def show_nationality_compare_dashboard(df: pd.DataFrame):
    st.title("国籍比較ダッシュボード")
    st.caption(f"データ期間：{DATA_PERIOD}")

    st.sidebar.markdown("### 比較条件（国籍ベース）")

    if "国籍・地域" not in df.columns:
        st.error("国籍・地域 列がデータに存在しません。")
        return

    # ★ 比較対象の選択肢は「（円／人）」だけにしておく
    base_exp_cols = [
        c for c in df.columns
        if "（円／人）" in c
    ]
    if not base_exp_cols:
        st.error("「（円／人）」を含む支出列が見つかりませんでした。")
        return

    base_exp_col = st.sidebar.selectbox("比較したい支出項目", base_exp_cols)

    # ★ 1日あたりで見るかどうかを選択
    use_per_day = st.sidebar.checkbox(
        "1日あたり（円／人・日）に換算して比較する",
        value=False
    )

    # 実際に集計に使う列名と単位を決定
    target_exp_col = base_exp_col
    unit = "円／人"

    if use_per_day:
        per_day_col = base_exp_col.replace("（円／人）", "（円／人・日）")
        if per_day_col in df.columns:
            target_exp_col = per_day_col
            unit = "円／人・日"
        else:
            st.sidebar.warning("この項目の 1日あたり列（円／人・日）が見つからなかったため、総額（円／人）で表示します。")

    nat_list = sorted(df["国籍・地域"].dropna().unique())
    default_nats = []
    for cand in ["アメリカ合衆国", "アメリカ", "米国", "カナダ"]:
        if cand in nat_list:
            default_nats.append(cand)

    nat_selected = st.sidebar.multiselect(
        "国籍・地域（複数選択可）",
        nat_list,
        default=default_nats or nat_list[:5]
    )

    pref_selected = []
    if "都道府県" in df.columns:
        all_prefs = sorted(df["都道府県"].dropna().unique())
        pref_selected = st.sidebar.multiselect(
            "都道府県（任意・複数選択可）",
            all_prefs,
            default=[]
        )

    df_filtered = df.copy()
    if nat_selected:
        df_filtered = df_filtered[df_filtered["国籍・地域"].isin(nat_selected)]
    if pref_selected:
        df_filtered = df_filtered[df_filtered["都道府県"].isin(pref_selected)]

    st.subheader("現在の条件（国籍ベース）")
    st.write(f"- 比較項目：**{base_exp_col}**")
    st.write(f"- 単位　　：**{unit}**")
    st.write(f"- 国籍・地域：**{', '.join(nat_selected) if nat_selected else '（未選択）'}**")
    if pref_selected:
        st.write(f"- 都道府県：**{', '.join(pref_selected)}**")

    if df_filtered.empty:
        st.warning("この条件に一致するデータがありません。条件を見直してください。")
        return

    y_col_name = f"平均額（{unit}）"

    summary = (
        df_filtered.groupby("国籍・地域")[target_exp_col]
        .mean()
        .reset_index()
        .rename(columns={target_exp_col: y_col_name})
        .sort_values(y_col_name, ascending=False)
    )

    st.subheader(f"国籍別 平均支出（{unit}）")

    fig = px.bar(
        summary,
        x="国籍・地域",
        y=y_col_name,
        title=f"{base_exp_col} の国籍別平均（{unit}）"
    )
    fig.update_yaxes(tickformat=",")
    fig.update_traces(hovertemplate="%{x}: %{y:,.0f} 円")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(summary, use_container_width=True)


# =========================================================
# ④ 居住地比較ダッシュボード（新規・修正版）
# =========================================================
def show_residence_compare_dashboard(df: pd.DataFrame):
    st.title("居住地比較ダッシュボード")
    st.caption(f"データ期間：{DATA_PERIOD}")

    st.sidebar.markdown("### 比較条件（居住地ベース）")

    # ★ 居住地カラム名を柔軟に取得
    res_col = get_res_col(df)
    if res_col is None:
        st.error("居住地（または居住地・地域）に相当する列が見つかりません。")
        return

    # ★ 比較対象の選択肢は「（円／人）」だけにする
    base_exp_cols = [
        c for c in df.columns
        if "（円／人）" in c
    ]
    if not base_exp_cols:
        st.error("「（円／人）」を含む支出列が見つかりませんでした。")
        return

    base_exp_col = st.sidebar.selectbox("比較したい支出項目", base_exp_cols)

    # ★ 1日あたりで見るかどうか
    use_per_day = st.sidebar.checkbox(
        "1日あたり（円／人・日）に換算して比較する",
        value=False
    )

    # 実際に使う列名と単位を決定
    target_exp_col = base_exp_col
    unit = "円／人"

    if use_per_day:
        per_day_col = base_exp_col.replace("（円／人）", "（円／人・日）")
        if per_day_col in df.columns:
            target_exp_col = per_day_col
            unit = "円／人・日"
        else:
            st.sidebar.warning("この項目の 1日あたり列（円／人・日）が見つからなかったため、総額（円／人）で表示します。")

    res_list = sorted(df[res_col].dropna().unique())
    res_selected = st.sidebar.multiselect(
        "居住地（複数選択可）",
        res_list,
        default=res_list[:5]
    )

    pref_selected = []
    if "都道府県" in df.columns:
        all_prefs = sorted(df["都道府県"].dropna().unique())
        pref_selected = st.sidebar.multiselect(
            "都道府県（任意・複数選択可）",
            all_prefs,
            default=[]
        )

    df_filtered = df.copy()
    if res_selected:
        df_filtered = df_filtered[df_filtered[res_col].isin(res_selected)]
    if pref_selected:
        df_filtered = df_filtered[df_filtered["都道府県"].isin(pref_selected)]

    st.subheader("現在の条件（居住地ベース）")
    st.write(f"- 比較項目：**{base_exp_col}**")
    st.write(f"- 単位　　：**{unit}**")
    st.write(f"- 居住地　：**{', '.join(res_selected) if res_selected else '（未選択）'}**")
    if pref_selected:
        st.write(f"- 都道府県：**{', '.join(pref_selected)}**")

    if df_filtered.empty:
        st.warning("この条件に一致するデータがありません。条件を見直してください。")
        return

    y_col_name = f"平均額（{unit}）"

    summary = (
        df_filtered.groupby(res_col)[target_exp_col]
        .mean()
        .reset_index()
        .rename(columns={target_exp_col: y_col_name})
        .sort_values(y_col_name, ascending=False)
    )

    st.subheader(f"居住地別 平均支出（{unit}）")

    fig = px.bar(
        summary,
        x=res_col,
        y=y_col_name,
        title=f"{base_exp_col} の居住地別平均（{unit}）"
    )
    fig.update_yaxes(tickformat=",")
    fig.update_traces(hovertemplate="%{x}: %{y:,.0f} 円")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(summary, use_container_width=True)


# =========================================================
# main
# =========================================================
def main():
    df = load_data()

    # サイドバーにも期間を表示
    st.sidebar.markdown(f"**データ期間**：{DATA_PERIOD}")

    mode = st.sidebar.radio(
        "分析モードを選択",
        [
            "都道府県別ダッシュボード",
            "国籍×属性ダッシュボード",
            "居住地×属性ダッシュボード",
            "国籍比較ダッシュボード",
            "居住地比較ダッシュボード",
        ]
    )

    if mode == "都道府県別ダッシュボード":
        show_pref_dashboard(df)
    elif mode == "国籍×属性ダッシュボード":
        show_segment_dashboard(df)
    elif mode == "居住地×属性ダッシュボード":
        show_residence_segment_dashboard(df)
    elif mode == "国籍比較ダッシュボード":
        show_nationality_compare_dashboard(df)
    else:
        show_residence_compare_dashboard(df)


if __name__ == "__main__":
    main()
