# ==============================
# Copilot_Fast_Pipeline_v2 - PART 1: Foundation
# ==============================
from __future__ import annotations

from datetime import date, timedelta
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import DBSCAN

# ---- Join key ----
KEY = "experian_consumer_key"

# ---- Income categories (same as your pipeline) ----
INCOME_CATS = [
    "INC-OTH-001","INC-OTH-002","INC-OTH-003","INC-OTH-004",
    "INC-SAL-000","INC-SAL-002","INC-SAL-003","INC-SAL-005"
]

# ---- Override numeric codes (match your current logic) ----
OVR = {
    "NO_DDA":          999_999_999.0,
    "NO_INCOME_ALL":   999_999_998.0,
    "NO_INCOME_180":   999_999_997.0,
    "NUM0":            999_999_996.0,
    "DEN0":            999_999_995.0,
    "BOTH0":           999_999_994.0,
    "NO_TXN":          999_999_990.0,
    "ONE_TXN":         999_999_991.0,
    "ZERO":            0.0,
}
# ---- Date sentinels ----
OVR_TS           = date(2099, 12, 31)   # general ts override
OVR_EST_ONE      = date(2100, 1, 1)     # only 1 txn case (181–198)
OVR_EST_BEFORE   = date(2100, 12, 31)   # cascade for est dates < ref

# ---- A11 typing rules ----
A11_STRING = {"A11_PDAY1050","A11_PDAY2050","A11_PDAY3050"}
A11_DATE = {
    "A11_PDAY1403","A11_PDAY2403","A11_PDAY3403",
    "A11_PDAY1414","A11_PDAY2414","A11_PDAY3414",
    "A11_PDAY1413","A11_PDAY2413","A11_PDAY3413",
    "A11_PDAY1710","A11_PDAY2710","A11_PDAY3710",
    "A11_PDAY1720","A11_PDAY2720","A11_PDAY3720",
    "A11_PDAY1730","A11_PDAY2730","A11_PDAY3730",
    "A11_PDAY1740","A11_PDAY2740","A11_PDAY3740",
    "A11_PDAY1750","A11_PDAY2750","A11_PDAY3750",
    "A11_PDAY1760","A11_PDAY2760","A11_PDAY3760",
    "A11_PDAY4403","A11_PDAY4414","A11_PDAY4413",
}

def cast_all_A11_features(df: pl.DataFrame) -> pl.DataFrame:
    """Cast 3 string, listed date, and all other A11_* to Float64. Keep KEY Int64."""
    casts = []
    schema = df.schema
    if KEY in df.columns and schema.get(KEY) != pl.Int64:
        casts.append(pl.col(KEY).cast(pl.Int64))
    for c in df.columns:
        if not c.startswith("A11_"):
            continue
        if c in A11_STRING:
            casts.append(pl.col(c).cast(pl.Utf8))
        elif c in A11_DATE:
            cur = schema.get(c)
            if cur == pl.Date:
                continue
            elif cur == pl.Datetime:
                casts.append(pl.col(c).dt.date().alias(c))
            elif cur == pl.Utf8:
                casts.append(pl.col(c).str.strptime(pl.Date, strict=False).alias(c))
            else:
                casts.append(pl.col(c).cast(pl.Date))
        else:
            if schema.get(c) != pl.Float64:
                casts.append(pl.col(c).cast(pl.Float64))
    return df.with_columns(casts) if casts else df

def cast_and_concat_customer_list(all_customer_feats: list[pl.DataFrame]) -> pl.DataFrame:
    """Ensure consistent columns/dtypes across many customers before concat."""
    if not all_customer_feats:
        return pl.DataFrame()
    union = sorted({c for d in all_customer_feats for c in d.columns if c.startswith("A11_")})
    out = []
    for d in all_customer_feats:
        miss = [c for c in union if c not in d.columns]
        dd = d.with_columns([pl.lit(None).alias(c) for c in miss]) if miss else d
        out.append(cast_all_A11_features(dd))
    return pl.concat(out, how="vertical", coerce=True)

# ---- Override helpers (semantics identical to your current code) ----
def ratio_override(expr: pl.Expr, num: pl.Expr, den: pl.Expr, *, is_count_amount: bool) -> pl.Expr:
    return (
        pl.when((num == 0) & (den == 0)).then(OVR["BOTH0"])
         .when(num == 0).then(OVR["NUM0"] if not is_count_amount else 0.0)
         .when(den == 0).then(OVR["DEN0"])
         .otherwise(expr)
    )

def interval_override(expr: pl.Expr, count: pl.Expr) -> pl.Expr:
    return (
        pl.when(count == 0).then(OVR["NO_TXN"])
         .when(count == 1).then(OVR["ONE_TXN"])
         .otherwise(expr)
    )

def ensure_key_i64(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col(KEY).cast(pl.Int64)) if df.schema.get(KEY) != pl.Int64 else df

# ---- Category to int (same mapping as your script) ----
def cat_to_int_expr() -> pl.Expr:
    return (
        pl.when(pl.col("enriched_category") == "INC-OTH-001").then(1)
         .when(pl.col("enriched_category") == "INC-OTH-002").then(2)
         .when(pl.col("enriched_category") == "INC-OTH-003").then(3)
         .when(pl.col("enriched_category") == "INC-OTH-004").then(4)
         .when(pl.col("enriched_category") == "INC-SAL-000").then(5)
         .when(pl.col("enriched_category") == "INC-SAL-002").then(6)
         .when(pl.col("enriched_category") == "INC-SAL-003").then(7)
         .when(pl.col("enriched_category") == "INC-SAL-005").then(8)
         .otherwise(0)
    )

# ==============================
# Copilot_Fast_Pipeline_v2 - PART 2: Windows, Grouping, Ranks
# ==============================
def refdate_from_customer(df_eck: pl.DataFrame) -> date:
    # make sure D_appMonth is Date; derive per-customer ref_date as max(D_appMonth)
    if df_eck.schema.get("D_appMonth") != pl.Date:
        df_eck = df_eck.with_columns(pl.col("D_appMonth").str.slice(0,10).str.strptime(pl.Date, strict=False))
    return df_eck.select(pl.col("D_appMonth").max().alias("ref_date")).item()

def build_windows(ref_date: date) -> dict[str, pl.Expr]:
    cut = {
        "ref":  ref_date,
        "d30":  ref_date - timedelta(days=30),
        "d60":  ref_date - timedelta(days=60),
        "d90":  ref_date - timedelta(days=90),
        "d180": ref_date - timedelta(days=180),
        "d365": ref_date - timedelta(days=365),
    }
    return {
        "0_30":  (pl.col("txn_timestamp") > cut["d30"])  & (pl.col("txn_timestamp") <= cut["ref"]),
        "31_60": (pl.col("txn_timestamp") > cut["d60"])  & (pl.col("txn_timestamp") <= cut["d30"]),
        "61_90": (pl.col("txn_timestamp") > cut["d90"])  & (pl.col("txn_timestamp") <= cut["d60"]),
        "0_90":  (pl.col("txn_timestamp") > cut["d90"])  & (pl.col("txn_timestamp") <= cut["ref"]),
        "0_180": (pl.col("txn_timestamp") > cut["d180"]) & (pl.col("txn_timestamp") <= cut["ref"]),
        "0_365": (pl.col("txn_timestamp") > cut["d365"]) & (pl.col("txn_timestamp") <= cut["ref"]),
    }

def group_type_and_override(df_txn: pl.DataFrame, ref_date: date) -> tuple[str, float | None]:
    has_dda = df_txn.filter(pl.col("account_type_code") == "DDA").height > 0
    has_income_all = df_txn.filter(pl.col("enriched_category").is_in(INCOME_CATS)).height > 0
    has_income_180 = df_txn.filter(
        (pl.col("txn_timestamp") > ref_date - timedelta(days=180))
        & (pl.col("enriched_category").is_in(INCOME_CATS))
    ).height > 0
    if not has_dda:
        return "A", OVR["NO_DDA"]
    if not has_income_all:
        return "A", OVR["NO_INCOME_ALL"]
    if not has_income_180:
        return "A", OVR["NO_INCOME_180"]
    return "B", None

# --- light clustering + top-3 (same semantics; hashing not a bottleneck) ---
def cluster_and_rank_top3(df_eck: pl.DataFrame,
                          hashing: HashingVectorizer | None = None,
                          clustering: DBSCAN | None = None,
                          ref_date: date | None = None) -> pl.DataFrame:
    if df_eck.schema.get("txn_timestamp") != pl.Date:
        df_eck = df_eck.with_columns(pl.col("txn_timestamp").str.slice(0,10).str.strptime(pl.Date, strict=False))
    # clean description
    df_eck = df_eck.with_columns(
        pl.col("cleaned_description").fill_null("").str.to_lowercase()
           .str.replace_all(r"[^a-zA-Z\s]","").str.strip_chars()
    )
    if hashing is None:
        hashing = HashingVectorizer(n_features=1000, ngram_range=(1,2), alternate_sign=False, norm="l2")
    if clustering is None:
        clustering = DBSCAN(eps=0.5, min_samples=3)

    # per (account_vid, enriched_category)
    chunks = []
    for (_, _), grp in df_eck.group_by(["account_vid","enriched_category"], maintain_order=True):
        X = hashing.transform(grp["cleaned_description"].to_list()).toarray()
        labels = clustering.fit_predict(X) if X.shape[1] > 0 else np.full(X.shape[0], -1)
        chunks.append(
            grp.select([KEY,"account_vid","txn_amount","txn_timestamp","cleaned_description",
                        "account_type_code","enriched_category"])
               .hstack(pl.DataFrame({"cluster_label": labels}))
        )
    df_income = pl.concat(chunks, how="vertical")
    # acct_cluster_id = account_vid*1000 + cat_int + cluster_label
    df_income = df_income.with_columns(
        (pl.col("account_vid") * 1000 + cat_to_int_expr() + pl.col("cluster_label")).alias("acct_cluster_id")
    )

    # last 180d, not noise
    if ref_date is None:
        ref_date = refdate_from_customer(df_eck)
    cut180 = ref_date - timedelta(days=180)
    df180 = (
        df_income
        .filter((pl.col("txn_timestamp") >= cut180) & (pl.col("txn_timestamp") <= ref_date))
        .filter(pl.col("cluster_label") != -1)
    )

    # sums per window → rank within customer
    cut30, cut60, cut90 = ref_date - timedelta(days=30), ref_date - timedelta(days=60), ref_date - timedelta(days=90)
    agg = (
        df180.group_by([KEY,"acct_cluster_id"])
             .agg([
                 pl.when(pl.col("txn_timestamp") > cut30).then(pl.col("txn_amount")).otherwise(0).sum().alias("sum_30"),
                 pl.when(pl.col("txn_timestamp") > cut60).then(pl.col("txn_amount")).otherwise(0).sum().alias("sum_60"),
                 pl.when(pl.col("txn_timestamp") > cut90).then(pl.col("txn_amount")).otherwise(0).sum().alias("sum_90"),
                 pl.col("txn_amount").sum().alias("sum_180"),
             ])
             .sort([KEY,"sum_30","sum_60","sum_90","sum_180"], descending=[False,True,True,True,True])
             .with_row_count("rc")
             .with_columns((pl.col("rc") - pl.col("rc").min().over(KEY) + 1).alias("rank"))
             .drop("rc")
             .filter(pl.col("rank") <= 3)
             .select([KEY,"acct_cluster_id","rank"])
    )
    return df_income.join(agg, on=[KEY,"acct_cluster_id"], how="left")\
                    .with_columns(pl.col("enriched_category").is_in(INCOME_CATS).alias("is_income_txn"))

# ==============================
# Copilot_Fast_Pipeline_v2 - PART 3: Bases (ranked & agnostic)
# ==============================
def _prepare_txn(df_txn: pl.DataFrame) -> pl.DataFrame:
    """Ensure timestamp/date columns exist; add dt/dow for later use."""
    df = df_txn
    if df.schema.get("txn_timestamp") != pl.Date:
        df = df.with_columns(pl.col("txn_timestamp").str.slice(0,10).str.strptime(pl.Date, strict=False))
    return df.with_columns([
        pl.col("txn_timestamp").dt.date().alias("dt"),
        pl.col("txn_timestamp").dt.weekday().alias("dow"),
    ])

def _interval_cols_from_dtlist(df: pl.DataFrame, key_cols: list[str]) -> pl.DataFrame:
    """Given grouped dt_list, add intervals/recent/max/min/avg and mode stats in one go."""
    # intervals as list of ints (days)
    df = df.with_columns(
        (pl.col("dt_list").list.diff().list.eval(pl.element().dt.total_days())).alias("intervals")
    )

    # explode intervals to compute mode (largest on tie) + dominance
    iv = df.select(key_cols + ["intervals"]).explode("intervals").drop_nulls()
    if iv.height > 0:
        iv_cnt = iv.group_by(key_cols + ["intervals"]).agg(pl.count().alias("freq"))
        mode_tbl = (
            iv_cnt.sort(["freq","intervals"], descending=[True,True])
                  .group_by(key_cols)
                  .agg([
                      pl.first("intervals").alias("mode_interval"),
                      pl.first("freq").alias("mode_freq"),
                      pl.sum("freq").alias("ivals_count"),
                  ])
                  .with_columns((pl.col("mode_freq")/pl.col("ivals_count")).alias("mode_dom"))
                  .drop("mode_freq","ivals_count")
        )
        df = df.join(mode_tbl, on=key_cols, how="left")
        # large and missed counts
        miss_large = (
            iv.join(mode_tbl.select(key_cols+["mode_interval"]), on=key_cols, how="left")
              .with_columns([
                  (pl.col("intervals") > pl.col("mode_interval")).alias("missed"),
                  (pl.col("intervals") > 1.5*pl.col("mode_interval")).alias("is_large"),
              ])
              .group_by(key_cols).agg([
                  pl.col("missed").sum().alias("missed_count"),
                  pl.col("is_large").sum().alias("large_count"),
              ])
        )
        df = df.join(miss_large, on=key_cols, how="left")
    else:
        # no intervals → nulls/zeros where appropriate
        df = df.with_columns([
            pl.lit(None).alias("mode_interval"),
            pl.lit(0.0).alias("mode_dom"),
            pl.lit(0).alias("missed_count"),
            pl.lit(0).alias("large_count"),
        ])

    # recent/max/min/avg intervals (from the list column)
    df = df.with_columns([
        pl.col("intervals").list.get(-1).alias("recent_interval"),
        pl.col("intervals").list.max().alias("max_interval"),
        pl.col("intervals").list.min().alias("min_interval"),
        pl.col("intervals").list.mean().alias("avg_interval"),
    ])
    return df

def _dow_mode_and_dom(df: pl.DataFrame, key_cols: list[str]) -> pl.DataFrame:
    """Mode DOW (largest weekday on tie) and dominance."""
    dow = df.select(key_cols + ["dow"])
    if dow.height == 0:
        return df.with_columns([
            pl.lit(None).alias("mode_dow"),
            pl.lit(0.0).alias("mode_dow_dom"),
        ])
    dow_cnt = (
        dow.group_by(key_cols+["dow"]).agg(pl.count().alias("freq"))
           .sort(["freq","dow"], descending=[True,True])
           .group_by(key_cols).agg(pl.first("dow").alias("mode_dow"))
    )
    dom_tbl = (
        dow.group_by(key_cols).agg(pl.count().alias("txn_cnt"))
           .join(
               dow.join(dow_cnt, on=key_cols, how="left")
                  .with_columns((pl.col("dow")==pl.col("mode_dow")).alias("is_mode"))
                  .group_by(key_cols).agg(pl.col("is_mode").sum().alias("mode_hits")),
               on=key_cols, how="left"
           )
           .with_columns((pl.col("mode_hits")/pl.col("txn_cnt")).alias("mode_dow_dom"))
           .select(key_cols + ["mode_dow_dom"])
    )
    return df.join(dow_cnt, on=key_cols, how="left")\
             .join(dom_tbl, on=key_cols, how="left")

def build_bases_v2(df_txn_with_ranks: pl.DataFrame,
                   windows: dict[str, pl.Expr]) -> dict[str, pl.DataFrame]:
    """
    Build compact bases once per window for:
      - Ranked:   (0_30, 31_60, 61_90, 0_90) grouped by [KEY, rank]
      - Agnostic: (0_30, 31_60, 61_90, 0_90, 0_180) grouped by [KEY]
    Each base includes:
      n_txn,sum_amt,mean_amt,std_amt,max_amt,min_amt,latest_amt,
      earliest_dt,latest_dt,dt_list,intervals (list), recent_interval,
      mode_interval (largest-tie), mode_dom, missed_count, large_count,
      mode_dow, mode_dow_dom
    """
    df = _prepare_txn(df_txn_with_ranks)
    res: dict[str, pl.DataFrame] = {}

    # ---------- Ranked bases ----------
    def ranked(win_key: str) -> pl.DataFrame:
        w = windows[win_key]
        gcols = [KEY, "rank"]
        base = (
            df.filter(w & (pl.col("rank").is_in([1,2,3])))
              .sort([KEY,"rank","txn_timestamp"])
              .group_by(gcols)
              .agg([
                  pl.count().alias("n_txn"),
                  pl.col("txn_amount").sum().alias("sum_amt"),
                  pl.col("txn_amount").mean().alias("mean_amt"),
                  pl.col("txn_amount").std().fill_null(0.0).alias("std_amt"),
                  pl.col("txn_amount").max().alias("max_amt"),
                  pl.col("txn_amount").min().alias("min_amt"),
                  pl.col("txn_amount").sort_by("dt").last().alias("latest_amt"),
                  pl.col("dt").min().alias("earliest_dt"),
                  pl.col("dt").max().alias("latest_dt"),
                  pl.col("dt").sort().alias("dt_list"),
              ])
        )
        base = _interval_cols_from_dtlist(base, gcols)
        base = _dow_mode_and_dom(df.filter(w & (pl.col("rank").is_in([1,2,3]))), gcols).join(base, on=gcols, how="right")
        return base

    res["rank_0_30"]  = ranked("0_30")
    res["rank_31_60"] = ranked("31_60")
    res["rank_61_90"] = ranked("61_90")
    res["rank_0_90"]  = ranked("0_90")

    # ---------- Agnostic bases ----------
    def agn(win_key: str) -> pl.DataFrame:
        w = windows[win_key]
        gcols = [KEY]
        base = (
            df.filter(w & (pl.col("rank") > 0))
              .sort([KEY,"txn_timestamp"])
              .group_by(gcols)
              .agg([
                  pl.count().alias("n_txn"),
                  pl.col("txn_amount").sum().alias("sum_amt"),
                  pl.col("txn_amount").mean().alias("mean_amt"),
                  pl.col("txn_amount").std().fill_null(0.0).alias("std_amt"),
                  pl.col("txn_amount").max().alias("max_amt"),
                  pl.col("txn_amount").min().alias("min_amt"),
                  pl.col("txn_amount").sort_by("dt").last().alias("latest_amt"),
                  pl.col("dt").min().alias("earliest_dt"),
                  pl.col("dt").max().alias("latest_dt"),
                  pl.col("dt").sort().alias("dt_list"),
              ])
        )
        base = _interval_cols_from_dtlist(base, gcols)
        base = _dow_mode_and_dom(df.filter(w & (pl.col("rank") > 0)), gcols).join(base, on=gcols, how="right")
        return base

    res["agn_0_30"]  = agn("0_30")
    res["agn_31_60"] = agn("31_60")
    res["agn_61_90"] = agn("61_90")
    res["agn_0_90"]  = agn("0_90")
    res["agn_0_180"] = agn("0_180")

    # Keep key typed
    for k,v in res.items():
        res[k] = ensure_key_i64(v)
    return res
# ==============================
# Copilot_Fast_Pipeline_v2 - PART 4A: Features 1–21
# ==============================
def features_1_9_v2(
    df_txn_with_ranks: pl.DataFrame,
    windows: dict[str, pl.Expr],
    override_value: float | None,
) -> pl.DataFrame:
    """
    Features 1–9:
      1  -> A11_PDAY0010 : count of unique paycheck sources in last 180d
      2  -> A11_PDAY0020 : sources(0–30) / sources(31–60)
      3  -> A11_PDAY0030 : sources(0–30) / sources(61–90)
      4–6  (A11_PDAY1040/2040/3040): account_vid of rank 1/2/3
      7–9  (A11_PDAY1050/2050/3050): enriched_category of rank 1/2/3 (string in final cast)
    Group A → all nine use the same override value (then cast by schema guard). 
    """
    key_val = int(df_txn_with_ranks[0, KEY])
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})

    # Group A short-circuit
    if override_value is not None:
        return base.with_columns([
            pl.lit(override_value).alias("A11_PDAY0010"),
            pl.lit(override_value).alias("A11_PDAY0020"),
            pl.lit(override_value).alias("A11_PDAY0030"),
            pl.lit(override_value).alias("A11_PDAY1040"),
            pl.lit(override_value).alias("A11_PDAY2040"),
            pl.lit(override_value).alias("A11_PDAY3040"),
            pl.lit(override_value).alias("A11_PDAY1050"),
            pl.lit(override_value).alias("A11_PDAY2050"),
            pl.lit(override_value).alias("A11_PDAY3050"),
        ])

    # --- Unique source counts by window (rank-agnostic: rank>0) ---
    def src_nuniq(win_key: str, out: str) -> pl.DataFrame:
        dfw = df_txn_with_ranks.filter(windows[win_key] & (pl.col("rank") > 0))
        return dfw.group_by(KEY).agg(pl.col("acct_cluster_id").n_unique().alias(out))

    # 180d unique
    u180 = src_nuniq("0_180", "U180")
    # ratios 0–30 / 31–60 and 0–30 / 61–90
    u30   = src_nuniq("0_30", "U30")
    u31_60 = src_nuniq("31_60", "U31_60")
    u61_90 = src_nuniq("61_90", "U61_90")

    out = (
        base.join(u180,   on=KEY, how="left")
            .join(u30,    on=KEY, how="left")
            .join(u31_60, on=KEY, how="left")
            .join(u61_90, on=KEY, how="left")
            # A11_PDAY0010
            .with_columns(pl.col("U180").fill_null(0).alias("A11_PDAY0010"))
            # A11_PDAY0020 and A11_PDAY0030 with ratio override (count-ratio: num=0 -> 0)
            .with_columns([
                ratio_override(
                    (pl.col("U30").fill_null(0) / pl.col("U31_60").fill_null(0)),
                    pl.col("U30").fill_null(0),
                    pl.col("U31_60").fill_null(0),
                    is_count_amount=True
                ).alias("A11_PDAY0020"),
                ratio_override(
                    (pl.col("U30").fill_null(0) / pl.col("U61_90").fill_null(0)),
                    pl.col("U30").fill_null(0),
                    pl.col("U61_90").fill_null(0),
                    is_count_amount=True
                ).alias("A11_PDAY0030"),
            ])
            .drop(["U180","U30","U31_60","U61_90"])
    )

    # --- Top-3 by rank: account_vid & enriched_category ---
    # account_vids
    for r, col in [(1,"A11_PDAY1040"), (2,"A11_PDAY2040"), (3,"A11_PDAY3040")]:
        acc = (df_txn_with_ranks.filter(pl.col("rank")==r)
                                .group_by(KEY).agg(pl.col("account_vid").first().alias(col)))
        out = out.join(acc, on=KEY, how="left").with_columns(pl.col(col).fill_null(OVR["NO_TXN"]))
    # enriched_category (string in final cast)
    for r, col in [(1,"A11_PDAY1050"), (2,"A11_PDAY2050"), (3,"A11_PDAY3050")]:
        cat = (df_txn_with_ranks.filter(pl.col("rank")==r)
                                .group_by(KEY).agg(pl.col("enriched_category").first().alias(col)))
        out = out.join(cat, on=KEY, how="left").with_columns(pl.col(col).fill_null(OVR["NO_TXN"]))

    # Float casts for numeric features (strings will be converted later by cast_all_A11_features)
    return out.with_columns([
        pl.col("A11_PDAY0010").cast(pl.Float64),
        pl.col("A11_PDAY0020").cast(pl.Float64),
        pl.col("A11_PDAY0030").cast(pl.Float64),
    ])


def features_10_15_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    Features 10–15: per-rank txn counts
      30d -> A11_PDAY1104 / 2104 / 3104
      90d -> A11_PDAY1103 / 2103 / 3103
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})

    if override_value is not None:
        return base.with_columns([
            pl.lit(override_value).alias("A11_PDAY1104"),
            pl.lit(override_value).alias("A11_PDAY2104"),
            pl.lit(override_value).alias("A11_PDAY3104"),
            pl.lit(override_value).alias("A11_PDAY1103"),
            pl.lit(override_value).alias("A11_PDAY2103"),
            pl.lit(override_value).alias("A11_PDAY3103"),
        ])

    r30 = bases["rank_0_30"].filter(pl.col(KEY)==key_val)
    r90 = bases["rank_0_90"].filter(pl.col(KEY)==key_val)

    def cnt(df: pl.DataFrame, rk: int) -> float:
        d = df.filter(pl.col("rank")==rk)
        return float(d["n_txn"][0]) if d.height else 0.0

    return base.with_columns([
        pl.lit(cnt(r30,1)).alias("A11_PDAY1104"),
        pl.lit(cnt(r30,2)).alias("A11_PDAY2104"),
        pl.lit(cnt(r30,3)).alias("A11_PDAY3104"),
        pl.lit(cnt(r90,1)).alias("A11_PDAY1103"),
        pl.lit(cnt(r90,2)).alias("A11_PDAY2103"),
        pl.lit(cnt(r90,3)).alias("A11_PDAY3103"),
    ]).with_columns([pl.col(c).cast(pl.Float64) for c in [
        "A11_PDAY1104","A11_PDAY2104","A11_PDAY3104",
        "A11_PDAY1103","A11_PDAY2103","A11_PDAY3103"
    ]])

def features_16_21_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    Features 16–21: per-rank count ratios
      0–30 / 31–60 -> A11_PDAY1113 / 2113 / 3113
      0–30 / 61–90 -> A11_PDAY1123 / 2123 / 3123
    (count-ratio semantics: BOTH0, DEN0, and num=0->0)
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})

    if override_value is not None:
        return base.with_columns([
            pl.lit(override_value).alias("A11_PDAY1113"),
            pl.lit(override_value).alias("A11_PDAY2113"),
            pl.lit(override_value).alias("A11_PDAY3113"),
            pl.lit(override_value).alias("A11_PDAY1123"),
            pl.lit(override_value).alias("A11_PDAY2123"),
            pl.lit(override_value).alias("A11_PDAY3123"),
        ])

    r30   = bases["rank_0_30"].filter(pl.col(KEY)==key_val)
    r31_60= bases["rank_31_60"].filter(pl.col(KEY)==key_val)
    r61_90= bases["rank_61_90"].filter(pl.col(KEY)==key_val)

    def get_cnt(df: pl.DataFrame, rk: int) -> float:
        d = df.filter(pl.col("rank")==rk)
        return float(d["n_txn"][0]) if d.height else 0.0

    def ratio(cnt_num: float, cnt_den: float) -> float:
        # count-ratio overrides
        if cnt_num == 0 and cnt_den == 0: return OVR["BOTH0"]
        if cnt_den == 0: return OVR["DEN0"]
        if cnt_num == 0: return 0.0
        return cnt_num / cnt_den

    vals = {}
    for rk, ncol, d60col, d90col in [
        (1, "A11_PDAY1113","A11_PDAY1113","A11_PDAY1123"),
        (2, "A11_PDAY2113","A11_PDAY2113","A11_PDAY2123"),
        (3, "A11_PDAY3113","A11_PDAY3113","A11_PDAY3123"),
    ]:
        n = get_cnt(r30, rk)
        d60 = get_cnt(r31_60, rk)
        d90 = get_cnt(r61_90, rk)
        vals[(rk, "31_60")] = ratio(n, d60)
        vals[(rk, "61_90")] = ratio(n, d90)

    return base.with_columns([
        pl.lit(vals[(1,"31_60")]).alias("A11_PDAY1113"),
        pl.lit(vals[(2,"31_60")]).alias("A11_PDAY2113"),
        pl.lit(vals[(3,"31_60")]).alias("A11_PDAY3113"),
        pl.lit(vals[(1,"61_90")]).alias("A11_PDAY1123"),
        pl.lit(vals[(2,"61_90")]).alias("A11_PDAY2123"),
        pl.lit(vals[(3,"61_90")]).alias("A11_PDAY3123"),
    ]).with_columns([pl.col(c).cast(pl.Float64) for c in [
        "A11_PDAY1113","A11_PDAY2113","A11_PDAY3113",
        "A11_PDAY1123","A11_PDAY2123","A11_PDAY3123"
    ]])

def build_features_block_1_21(
    df_txn_with_ranks: pl.DataFrame,
    bases: dict[str, pl.DataFrame],
    windows: dict[str, pl.Expr],
    override_value: float | None,
) -> pl.DataFrame:
    """Convenience wrapper that returns a single frame with A11_PDAY0010 … A11_PDAY3123."""
    key_val = int(df_txn_with_ranks[0, KEY])

    f1_9  = features_1_9_v2(df_txn_with_ranks, windows, override_value)
    f10_15= features_10_15_v2(bases, key_val, override_value)
    f16_21= features_16_21_v2(bases, key_val, override_value)

    out = ensure_key_i64(f1_9)
    for b in (f10_15, f16_21):
        out = out.join(ensure_key_i64(b), on=KEY, how="left")

    return out

# ==============================
# Copilot_Fast_Pipeline_v2 - PART 4B: Features 22–45
# ==============================
def _pick_rank_row(base: pl.DataFrame, key_val: int, rk: int) -> pl.DataFrame:
    """Return the single row for (key, rank) from a ranked base (or 0 rows if absent)."""
    return base.filter((pl.col(KEY) == key_val) & (pl.col("rank") == rk))

# -------- 22–27: Mode interval (largest on ties) --------
def features_22_27_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    A11_PDAY1204/2204/3204  (30d)  &  A11_PDAY1203/2203/3203  (90d)
    Semantics: if n_txn==0 -> NO_TXN; if n_txn==1 -> ONE_TXN; else -> mode_interval
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        return base.with_columns([
            pl.lit(override_value).alias("A11_PDAY1204"),
            pl.lit(override_value).alias("A11_PDAY2204"),
            pl.lit(override_value).alias("A11_PDAY3204"),
            pl.lit(override_value).alias("A11_PDAY1203"),
            pl.lit(override_value).alias("A11_PDAY2203"),
            pl.lit(override_value).alias("A11_PDAY3203"),
        ])

    r30 = bases["rank_0_30"]
    r90 = bases["rank_0_90"]

    def mode_val(B: pl.DataFrame, rk: int) -> float:
        row = _pick_rank_row(B, key_val, rk)
        if row.height == 0: return OVR["NO_TXN"]
        n = int(row["n_txn"][0] or 0)
        if n == 0: return OVR["NO_TXN"]
        if n == 1: return OVR["ONE_TXN"]
        mv = row["mode_interval"][0]
        return float(mv) if mv is not None else OVR["NO_TXN"]

    return base.with_columns([
        pl.lit(mode_val(r30,1)).alias("A11_PDAY1204"),
        pl.lit(mode_val(r30,2)).alias("A11_PDAY2204"),
        pl.lit(mode_val(r30,3)).alias("A11_PDAY3204"),
        pl.lit(mode_val(r90,1)).alias("A11_PDAY1203"),
        pl.lit(mode_val(r90,2)).alias("A11_PDAY2203"),
        pl.lit(mode_val(r90,3)).alias("A11_PDAY3203"),
    ]).with_columns([pl.col(c).cast(pl.Float64) for c in [
        "A11_PDAY1204","A11_PDAY2204","A11_PDAY3204",
        "A11_PDAY1203","A11_PDAY2203","A11_PDAY3203"
    ]])

# -------- 28–33: Mode dominance (%) --------
def features_28_33_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    A11_PDAY1214/2214/3214  (30d)  &  A11_PDAY1213/2213/3213  (90d)
    Semantics: if n_txn==0 -> NO_TXN; if n_txn==1 -> ONE_TXN; else -> mode_dom*100
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        return base.with_columns([
            pl.lit(override_value).alias("A11_PDAY1214"),
            pl.lit(override_value).alias("A11_PDAY2214"),
            pl.lit(override_value).alias("A11_PDAY3214"),
            pl.lit(override_value).alias("A11_PDAY1213"),
            pl.lit(override_value).alias("A11_PDAY2213"),
            pl.lit(override_value).alias("A11_PDAY3213"),
        ])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def dom_val(B: pl.DataFrame, rk: int) -> float:
        row = _pick_rank_row(B, key_val, rk)
        if row.height == 0: return OVR["NO_TXN"]
        n = int(row["n_txn"][0] or 0)
        if n == 0: return OVR["NO_TXN"]
        if n == 1: return OVR["ONE_TXN"]
        md = row["mode_dom"][0]
        return float((md or 0.0) * 100.0)

    return base.with_columns([
        pl.lit(dom_val(r30,1)).alias("A11_PDAY1214"),
        pl.lit(dom_val(r30,2)).alias("A11_PDAY2214"),
        pl.lit(dom_val(r30,3)).alias("A11_PDAY3214"),
        pl.lit(dom_val(r90,1)).alias("A11_PDAY1213"),
        pl.lit(dom_val(r90,2)).alias("A11_PDAY2213"),
        pl.lit(dom_val(r90,3)).alias("A11_PDAY3213"),
    ]).with_columns([pl.col(c).cast(pl.Float64) for c in [
        "A11_PDAY1214","A11_PDAY2214","A11_PDAY3214",
        "A11_PDAY1213","A11_PDAY2213","A11_PDAY3213"
    ]])

# -------- 34–39: Count of intervals > 1.5×mode --------
def features_34_39_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    A11_PDAY1224/2224/3224  (30d)  &  A11_PDAY1223/2223/3223  (90d)
    Semantics: if n_txn==0 -> NO_TXN; if n_txn==1 -> ONE_TXN; else -> large_count
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        return base.with_columns([
            pl.lit(override_value).alias("A11_PDAY1224"),
            pl.lit(override_value).alias("A11_PDAY2224"),
            pl.lit(override_value).alias("A11_PDAY3224"),
            pl.lit(override_value).alias("A11_PDAY1223"),
            pl.lit(override_value).alias("A11_PDAY2223"),
            pl.lit(override_value).alias("A11_PDAY3223"),
        ])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def large_val(B: pl.DataFrame, rk: int) -> float:
        row = _pick_rank_row(B, key_val, rk)
        if row.height == 0: return OVR["NO_TXN"]
        n = int(row["n_txn"][0] or 0)
        if n == 0: return OVR["NO_TXN"]
        if n == 1: return OVR["ONE_TXN"]
        return float(row["large_count"][0] or 0.0)

    return base.with_columns([
        pl.lit(large_val(r30,1)).alias("A11_PDAY1224"),
        pl.lit(large_val(r30,2)).alias("A11_PDAY2224"),
        pl.lit(large_val(r30,3)).alias("A11_PDAY3224"),
        pl.lit(large_val(r90,1)).alias("A11_PDAY1223"),
        pl.lit(large_val(r90,2)).alias("A11_PDAY2223"),
        pl.lit(large_val(r90,3)).alias("A11_PDAY3223"),
    ]).with_columns([pl.col(c).cast(pl.Float64) for c in [
        "A11_PDAY1224","A11_PDAY2224","A11_PDAY3224",
        "A11_PDAY1223","A11_PDAY2223","A11_PDAY3223"
    ]])

# -------- 40–45: Recent interval / Mode interval --------
def features_40_45_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    A11_PDAY1234/2234/3234  (30d)  &  A11_PDAY1233/2233/3233  (90d)
    Interval overrides: if n_txn==0 -> NO_TXN; if n_txn==1 -> ONE_TXN
    Ratio overrides applied to (recent_interval / mode_interval):
      BOTH0 (num=0 & den=0), DEN0 (den=0), NUM0 (num=0)
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        return base.with_columns([
            pl.lit(override_value).alias("A11_PDAY1234"),
            pl.lit(override_value).alias("A11_PDAY2234"),
            pl.lit(override_value).alias("A11_PDAY3234"),
            pl.lit(override_value).alias("A11_PDAY1233"),
            pl.lit(override_value).alias("A11_PDAY2233"),
            pl.lit(override_value).alias("A11_PDAY3233"),
        ])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def ratio_val(B: pl.DataFrame, rk: int) -> float:
        row = _pick_rank_row(B, key_val, rk)
        if row.height == 0: return OVR["NO_TXN"]
        n = int(row["n_txn"][0] or 0)
        if n == 0: return OVR["NO_TXN"]
        if n == 1: return OVR["ONE_TXN"]
        num = row["recent_interval"][0]
        den = row["mode_interval"][0]
        # ratio overrides
        if (num is None or num == 0) and (den is None or den == 0): return OVR["BOTH0"]
        if den in (None, 0): return OVR["DEN0"]
        if num in (None, 0): return OVR["NUM0"]
        return float(num) / float(den)

    return base.with_columns([
        pl.lit(ratio_val(r30,1)).alias("A11_PDAY1234"),
        pl.lit(ratio_val(r30,2)).alias("A11_PDAY2234"),
        pl.lit(ratio_val(r30,3)).alias("A11_PDAY3234"),
        pl.lit(ratio_val(r90,1)).alias("A11_PDAY1233"),
        pl.lit(ratio_val(r90,2)).alias("A11_PDAY2233"),
        pl.lit(ratio_val(r90,3)).alias("A11_PDAY3233"),
    ]).with_columns([pl.col(c).cast(pl.Float64) for c in [
        "A11_PDAY1234","A11_PDAY2234","A11_PDAY3234",
        "A11_PDAY1233","A11_PDAY2233","A11_PDAY3233"
    ]])

# -------- Wrapper: combine 22–45 into one small frame --------
def build_features_block_22_45(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    f22_27 = features_22_27_v2(bases, key_val, override_value)
    f28_33 = features_28_33_v2(bases, key_val, override_value)
    f34_39 = features_34_39_v2(bases, key_val, override_value)
    f40_45 = features_40_45_v2(bases, key_val, override_value)

    out = ensure_key_i64(f22_27)
    for b in (f28_33, f34_39, f40_45):
        out = out.join(ensure_key_i64(b), on=KEY, how="left")
    return out

# ==============================
# Copilot_Fast_Pipeline_v2 - PART 4C: Features 46–75 (corrected)
# ==============================

def _pick_rank_row(base: pl.DataFrame, key_val: int, rk: int) -> pl.DataFrame:
    """Return the single row for (key, rank) from a ranked base (or 0 rows if absent)."""
    return base.filter((pl.col(KEY) == key_val) & (pl.col("rank") == rk))

# -------- 46–63: Interval stats (max/min/avg) --------
def features_46_63_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    30d:
      Rank1 -> A11_PDAY1244 (max), A11_PDAY1254 (min), A11_PDAY1264 (avg)
      Rank2 -> A11_PDAY2244, A11_PDAY2254, A11_PDAY2264
      Rank3 -> A11_PDAY3244, A11_PDAY3254, A11_PDAY3264
    90d:
      Rank1 -> A11_PDAY1243 (max), A11_PDAY1253 (min), A11_PDAY1263 (avg)
      Rank2 -> A11_PDAY2243, A11_PDAY2253, A11_PDAY2263
      Rank3 -> A11_PDAY3243, A11_PDAY3253, A11_PDAY3263
    Interval overrides:
      n_txn==0 -> NO_TXN; n_txn==1 -> ONE_TXN; else value
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        cols = [
            # 30d
            "A11_PDAY1244","A11_PDAY1254","A11_PDAY1264",
            "A11_PDAY2244","A11_PDAY2254","A11_PDAY2264",
            "A11_PDAY3244","A11_PDAY3254","A11_PDAY3264",
            # 90d
            "A11_PDAY1243","A11_PDAY1253","A11_PDAY1263",
            "A11_PDAY2243","A11_PDAY2253","A11_PDAY2263",
            "A11_PDAY3243","A11_PDAY3253","A11_PDAY3263",
        ]
        return base.with_columns([pl.lit(override_value).alias(c) for c in cols])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def stat_val(B: pl.DataFrame, rk: int, which: str) -> float:
        row = _pick_rank_row(B, key_val, rk)
        if row.height == 0: return OVR["NO_TXN"]
        n = int(row["n_txn"][0] or 0)
        if n == 0: return OVR["NO_TXN"]
        if n == 1: return OVR["ONE_TXN"]
        v = row[which][0]
        return float(v) if v is not None else OVR["NO_TXN"]

    out = base.with_columns([
        # 30d
        pl.lit(stat_val(r30,1,"max_interval")).alias("A11_PDAY1244"),
        pl.lit(stat_val(r30,1,"min_interval")).alias("A11_PDAY1254"),
        pl.lit(stat_val(r30,1,"avg_interval")).alias("A11_PDAY1264"),
        pl.lit(stat_val(r30,2,"max_interval")).alias("A11_PDAY2244"),
        pl.lit(stat_val(r30,2,"min_interval")).alias("A11_PDAY2254"),
        pl.lit(stat_val(r30,2,"avg_interval")).alias("A11_PDAY2264"),
        pl.lit(stat_val(r30,3,"max_interval")).alias("A11_PDAY3244"),
        pl.lit(stat_val(r30,3,"min_interval")).alias("A11_PDAY3254"),
        pl.lit(stat_val(r30,3,"avg_interval")).alias("A11_PDAY3264"),
        # 90d
        pl.lit(stat_val(r90,1,"max_interval")).alias("A11_PDAY1243"),
        pl.lit(stat_val(r90,1,"min_interval")).alias("A11_PDAY1253"),
        pl.lit(stat_val(r90,1,"avg_interval")).alias("A11_PDAY1263"),
        pl.lit(stat_val(r90,2,"max_interval")).alias("A11_PDAY2243"),
        pl.lit(stat_val(r90,2,"min_interval")).alias("A11_PDAY2253"),
        pl.lit(stat_val(r90,2,"avg_interval")).alias("A11_PDAY2263"),
        pl.lit(stat_val(r90,3,"max_interval")).alias("A11_PDAY3243"),
        pl.lit(stat_val(r90,3,"min_interval")).alias("A11_PDAY3253"),
        pl.lit(stat_val(r90,3,"avg_interval")).alias("A11_PDAY3263"),
    ])
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in out.columns if c != KEY])

# -------- 64–69: Range ratio of intervals (max-min)/avg --------
def features_64_69_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    30d: A11_PDAY1274 / 2274 / 3274
    90d: A11_PDAY1273 / 2273 / 3273
    Interval overrides first (n_txn==0 -> NO_TXN, n_txn==1 -> ONE_TXN),
    then ratio overrides on (num=(max-min), den=avg): BOTH0, DEN0 (NUM0 not applied).
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        return base.with_columns([
            pl.lit(override_value).alias("A11_PDAY1274"),
            pl.lit(override_value).alias("A11_PDAY2274"),
            pl.lit(override_value).alias("A11_PDAY3274"),
            pl.lit(override_value).alias("A11_PDAY1273"),
            pl.lit(override_value).alias("A11_PDAY2273"),
            pl.lit(override_value).alias("A11_PDAY3273"),
        ])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def rr(B: pl.DataFrame, rk: int) -> float:
        row = _pick_rank_row(B, key_val, rk)
        if row.height == 0: return OVR["NO_TXN"]
        n = int(row["n_txn"][0] or 0)
        if n == 0: return OVR["NO_TXN"]
        if n == 1: return OVR["ONE_TXN"]
        maxv = float(row["max_interval"][0] or 0.0)
        minv = float(row["min_interval"][0] or 0.0)
        avgv = float(row["avg_interval"][0] or 0.0)
        num = maxv - minv
        den = avgv
        if num == 0.0 and den == 0.0: return OVR["BOTH0"]
        if den == 0.0: return OVR["DEN0"]
        return num / den

    return base.with_columns([
        pl.lit(rr(r30,1)).alias("A11_PDAY1274"),
        pl.lit(rr(r30,2)).alias("A11_PDAY2274"),
        pl.lit(rr(r30,3)).alias("A11_PDAY3274"),
        pl.lit(rr(r90,1)).alias("A11_PDAY1273"),
        pl.lit(rr(r90,2)).alias("A11_PDAY2273"),
        pl.lit(rr(r90,3)).alias("A11_PDAY3273"),
    ]).with_columns([pl.col(c).cast(pl.Float64) for c in [
        "A11_PDAY1274","A11_PDAY2274","A11_PDAY3274",
        "A11_PDAY1273","A11_PDAY2273","A11_PDAY3273"
    ]])

# -------- 70–75: CV of intervals (std/mean) --------
def features_70_75_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    30d: A11_PDAY1284 / 2284 / 3284
    90d: A11_PDAY1283 / 2283 / 3283
    Interval overrides first:
      n_txn==0 -> NO_TXN; n_txn==1 -> ONE_TXN
    Then ratio overrides on (num=std(intervals), den=mean(intervals)):
      BOTH0, DEN0; NUM0 not applied (CV=0 valid).
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        return base.with_columns([
            pl.lit(override_value).alias("A11_PDAY1284"),
            pl.lit(override_value).alias("A11_PDAY2284"),
            pl.lit(override_value).alias("A11_PDAY3284"),
            pl.lit(override_value).alias("A11_PDAY1283"),
            pl.lit(override_value).alias("A11_PDAY2283"),
            pl.lit(override_value).alias("A11_PDAY3283"),
        ])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def cv_val(B: pl.DataFrame, rk: int) -> float:
        row = _pick_rank_row(B, key_val, rk)
        if row.height == 0:
            return OVR["NO_TXN"]
        n = int(row["n_txn"][0] or 0)
        if n == 0:
            return OVR["NO_TXN"]
        if n == 1:
            return OVR["ONE_TXN"]

        # --- sanitize intervals list (drop None) ---
        ivals_raw = row["intervals"][0]
        ivals = []
        if isinstance(ivals_raw, list):
            ivals = [int(x) for x in ivals_raw if x is not None]

        # If empty after sanitization, fall back to ratio overrides using avg_interval
        if len(ivals) == 0:
            num = 0.0
            den = float(row["avg_interval"][0] or 0.0)
        else:
            m = sum(ivals) / len(ivals)
            if len(ivals) >= 2:
                var = sum((x - m) ** 2 for x in ivals) / (len(ivals) - 1)  # sample std (ddof=1)
                num = var ** 0.5
            else:
                num = 0.0
            den = m

        # ratio overrides (std/mean): BOTH0, DEN0; NUM0 not applied
        if num == 0.0 and den == 0.0:
            return OVR["BOTH0"]
        if den == 0.0:
            return OVR["DEN0"]
        return num / den

    return base.with_columns([
        pl.lit(cv_val(r30,1)).alias("A11_PDAY1284"),
        pl.lit(cv_val(r30,2)).alias("A11_PDAY2284"),
        pl.lit(cv_val(r30,3)).alias("A11_PDAY3284"),
        pl.lit(cv_val(r90,1)).alias("A11_PDAY1283"),
        pl.lit(cv_val(r90,2)).alias("A11_PDAY2283"),
        pl.lit(cv_val(r90,3)).alias("A11_PDAY3283"),
    ]).with_columns([pl.col(c).cast(pl.Float64) for c in [
        "A11_PDAY1284","A11_PDAY2284","A11_PDAY3284",
        "A11_PDAY1283","A11_PDAY2283","A11_PDAY3283"
    ]])

# -------- Wrapper: combine 46–75 into one small frame --------
def build_features_block_46_75(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    f46_63 = features_46_63_v2(bases, key_val, override_value)
    f64_69 = features_64_69_v2(bases, key_val, override_value)
    f70_75 = features_70_75_v2(bases, key_val, override_value)

    out = ensure_key_i64(f46_63)
    for b in (f64_69, f70_75):
        out = out.join(ensure_key_i64(b), on=KEY, how="left")
    return out

# ==============================
# Copilot_Fast_Pipeline_v2 - PART 4D: Features 76–99
# ==============================

# We reuse _pick_rank_row(KEY, rk) from earlier pieces.

# -------- 76–81: Cross-window mode-interval ratios --------
def features_76_81_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    30d/31–60d:
      Rank1 -> A11_PDAY1293, Rank2 -> A11_PDAY2293, Rank3 -> A11_PDAY3293
    30d/61–90d:
      Rank1 -> A11_PDAY1303, Rank2 -> A11_PDAY2303, Rank3 -> A11_PDAY3303

    Rules (matching your pipeline):
      1) Interval overrides FIRST using min(count_num, count_den):
         - min==0 -> NO_TXN
         - min==1 -> ONE_TXN
      2) Ratio overrides on (mode_interval_num / mode_interval_den):
         - BOTH0 if num==0 and den==0
         - DEN0  if den==0
         - NUM0 not applied (count ratio semantics) -> return 0.0 when num==0 and den>0
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        return base.with_columns([
            pl.lit(override_value).alias("A11_PDAY1293"),
            pl.lit(override_value).alias("A11_PDAY2293"),
            pl.lit(override_value).alias("A11_PDAY3293"),
            pl.lit(override_value).alias("A11_PDAY1303"),
            pl.lit(override_value).alias("A11_PDAY2303"),
            pl.lit(override_value).alias("A11_PDAY3303"),
        ])

    r30    = bases["rank_0_30"]
    r31_60 = bases["rank_31_60"]
    r61_90 = bases["rank_61_90"]

    def cross_ratio(numB: pl.DataFrame, denB: pl.DataFrame, rk: int) -> float:
        nrow = _pick_rank_row(numB, key_val, rk)
        drow = _pick_rank_row(denB, key_val, rk)
        n_cnt = int(nrow["n_txn"][0] or 0) if nrow.height else 0
        d_cnt = int(drow["n_txn"][0] or 0) if drow.height else 0
        mcnt  = min(n_cnt, d_cnt)

        # interval overrides
        if mcnt == 0: return OVR["NO_TXN"]
        if mcnt == 1: return OVR["ONE_TXN"]

        num = float(nrow["mode_interval"][0] or 0.0)
        den = float(drow["mode_interval"][0] or 0.0)
        # ratio overrides (count-ratio semantics)
        if num == 0.0 and den == 0.0: return OVR["BOTH0"]
        if den == 0.0: return OVR["DEN0"]
        if num == 0.0: return 0.0
        return num / den

    return base.with_columns([
        # 30d / 31–60d
        pl.lit(cross_ratio(r30, r31_60, 1)).alias("A11_PDAY1293"),
        pl.lit(cross_ratio(r30, r31_60, 2)).alias("A11_PDAY2293"),
        pl.lit(cross_ratio(r30, r31_60, 3)).alias("A11_PDAY3293"),
        # 30d / 61–90d
        pl.lit(cross_ratio(r30, r61_90, 1)).alias("A11_PDAY1303"),
        pl.lit(cross_ratio(r30, r61_90, 2)).alias("A11_PDAY2303"),
        pl.lit(cross_ratio(r30, r61_90, 3)).alias("A11_PDAY3303"),
    ]).with_columns([pl.col(c).cast(pl.Float64) for c in [
        "A11_PDAY1293","A11_PDAY2293","A11_PDAY3293",
        "A11_PDAY1303","A11_PDAY2303","A11_PDAY3303"
    ]])


# -------- 82–87: Missed paychecks count (> mode interval) --------
def features_82_87_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    30d:
      Rank1 -> A11_PDAY1314, Rank2 -> A11_PDAY2314, Rank3 -> A11_PDAY3314
    90d:
      Rank1 -> A11_PDAY1313, Rank2 -> A11_PDAY2313, Rank3 -> A11_PDAY3313

    If n_txn==0 -> NO_TXN; if n_txn==1 -> ONE_TXN; else -> missed_count
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        return base.with_columns([
            pl.lit(override_value).alias("A11_PDAY1314"),
            pl.lit(override_value).alias("A11_PDAY2314"),
            pl.lit(override_value).alias("A11_PDAY3314"),
            pl.lit(override_value).alias("A11_PDAY1313"),
            pl.lit(override_value).alias("A11_PDAY2313"),
            pl.lit(override_value).alias("A11_PDAY3313"),
        ])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def missed(B: pl.DataFrame, rk: int) -> float:
        row = _pick_rank_row(B, key_val, rk)
        if row.height == 0: return OVR["NO_TXN"]
        n = int(row["n_txn"][0] or 0)
        if n == 0: return OVR["NO_TXN"]
        if n == 1: return OVR["ONE_TXN"]
        return float(row["missed_count"][0] or 0.0)

    return base.with_columns([
        pl.lit(missed(r30,1)).alias("A11_PDAY1314"),
        pl.lit(missed(r30,2)).alias("A11_PDAY2314"),
        pl.lit(missed(r30,3)).alias("A11_PDAY3314"),
        pl.lit(missed(r90,1)).alias("A11_PDAY1313"),
        pl.lit(missed(r90,2)).alias("A11_PDAY2313"),
        pl.lit(missed(r90,3)).alias("A11_PDAY3313"),
    ]).with_columns([pl.col(c).cast(pl.Float64) for c in [
        "A11_PDAY1314","A11_PDAY2314","A11_PDAY3314",
        "A11_PDAY1313","A11_PDAY2313","A11_PDAY3313"
    ]])


# -------- 88–93: Mode Day-of-Week --------
def features_88_93_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    30d (mode DOW):
      Rank1 -> A11_PDAY1324, Rank2 -> A11_PDAY2324, Rank3 -> A11_PDAY3324
    90d (mode DOW):
      Rank1 -> A11_PDAY1323, Rank2 -> A11_PDAY2323, Rank3 -> A11_PDAY3323

    Rules:
      n_txn==0 -> NO_TXN
      n_txn>=1 -> mode_dow (ties resolved to larger weekday in base)
      (When n_txn==1, mode_dow equals the single dow — matches your “single_dow” rule.)
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        return base.with_columns([
            pl.lit(override_value).alias("A11_PDAY1324"),
            pl.lit(override_value).alias("A11_PDAY2324"),
            pl.lit(override_value).alias("A11_PDAY3324"),
            pl.lit(override_value).alias("A11_PDAY1323"),
            pl.lit(override_value).alias("A11_PDAY2323"),
            pl.lit(override_value).alias("A11_PDAY3323"),
        ])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def mode_dow_val(B: pl.DataFrame, rk: int) -> float:
        row = _pick_rank_row(B, key_val, rk)
        if row.height == 0: return OVR["NO_TXN"]
        n = int(row["n_txn"][0] or 0)
        if n == 0: return OVR["NO_TXN"]
        md = row["mode_dow"][0]
        return float(md) if md is not None else OVR["NO_TXN"]

    return base.with_columns([
        pl.lit(mode_dow_val(r30,1)).alias("A11_PDAY1324"),
        pl.lit(mode_dow_val(r30,2)).alias("A11_PDAY2324"),
        pl.lit(mode_dow_val(r30,3)).alias("A11_PDAY3324"),
        pl.lit(mode_dow_val(r90,1)).alias("A11_PDAY1323"),
        pl.lit(mode_dow_val(r90,2)).alias("A11_PDAY2323"),
        pl.lit(mode_dow_val(r90,3)).alias("A11_PDAY3323"),
    ]).with_columns([pl.col(c).cast(pl.Float64) for c in [
        "A11_PDAY1324","A11_PDAY2324","A11_PDAY3324",
        "A11_PDAY1323","A11_PDAY2323","A11_PDAY3323"
    ]])


# -------- 94–99: Mode Day-of-Week dominance --------
def features_94_99_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    30d dominance:
      Rank1 -> A11_PDAY1334, Rank2 -> A11_PDAY2334, Rank3 -> A11_PDAY3334
    90d dominance:
      Rank1 -> A11_PDAY1333, Rank2 -> A11_PDAY2333, Rank3 -> A11_PDAY3333

    Rules:
      n_txn==0 -> NO_TXN
      n_txn==1 -> 1.0
      n_txn>=2 -> mode_dow_dom
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        return base.with_columns([
            pl.lit(override_value).alias("A11_PDAY1334"),
            pl.lit(override_value).alias("A11_PDAY2334"),
            pl.lit(override_value).alias("A11_PDAY3334"),
            pl.lit(override_value).alias("A11_PDAY1333"),
            pl.lit(override_value).alias("A11_PDAY2333"),
            pl.lit(override_value).alias("A11_PDAY3333"),
        ])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def dom(B: pl.DataFrame, rk: int) -> float:
        row = _pick_rank_row(B, key_val, rk)
        if row.height == 0: return OVR["NO_TXN"]
        n = int(row["n_txn"][0] or 0)
        if n == 0: return OVR["NO_TXN"]
        if n == 1: return 1.0
        md = row["mode_dow_dom"][0]
        return float(md or 0.0)

    return base.with_columns([
        pl.lit(dom(r30,1)).alias("A11_PDAY1334"),
        pl.lit(dom(r30,2)).alias("A11_PDAY2334"),
        pl.lit(dom(r30,3)).alias("A11_PDAY3334"),
        pl.lit(dom(r90,1)).alias("A11_PDAY1333"),
        pl.lit(dom(r90,2)).alias("A11_PDAY2333"),
        pl.lit(dom(r90,3)).alias("A11_PDAY3333"),
    ]).with_columns([pl.col(c).cast(pl.Float64) for c in [
        "A11_PDAY1334","A11_PDAY2334","A11_PDAY3334",
        "A11_PDAY1333","A11_PDAY2333","A11_PDAY3333"
    ]])


# -------- Wrapper: combine 76–99 into one small frame --------
def build_features_block_76_99(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    f76_81 = features_76_81_v2(bases, key_val, override_value)
    f82_87 = features_82_87_v2(bases, key_val, override_value)
    f88_93 = features_88_93_v2(bases, key_val, override_value)
    f94_99 = features_94_99_v2(bases, key_val, override_value)

    out = ensure_key_i64(f76_81)
    for b in (f82_87, f88_93, f94_99):
        out = out.join(ensure_key_i64(b), on=KEY, how="left")
    return out

# ==============================
# Copilot_Fast_Pipeline_v2 - PART 4E: Features 100–126
# ==============================

def _pick_rank_row(base: pl.DataFrame, key_val: int, rk: int) -> pl.DataFrame:
    return base.filter((pl.col(KEY) == key_val) & (pl.col("rank") == rk))

def _date_or_ovr(row: pl.DataFrame, col: str) -> date:
    """Return a date from row[col] or the OVR_TS sentinel if missing."""
    if row.height == 0:
        return OVR_TS
    v = row[col][0]
    return v if v is not None else OVR_TS

def _n_or(row: pl.DataFrame, col: str, default: int = 0) -> int:
    if row.height == 0:
        return default
    vv = row[col][0]
    return int(vv or 0)

# 100–108: Date features (latest/earliest paydates)
def features_100_108_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    Date outputs per rank:
      - A11_PDAY1414/2414/3414 : latest paydate in 30d
      - A11_PDAY1413/2413/3413 : latest paydate in 90d
      - A11_PDAY1403/2403/3403 : earliest paydate in 90d
    If n_txn==0 in that window -> OVR_TS.
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})

    # Group A -> dates become the sentinel OVR_TS
    if override_value is not None:
        return base.with_columns([
            pl.lit(OVR_TS).alias("A11_PDAY1414"),
            pl.lit(OVR_TS).alias("A11_PDAY2414"),
            pl.lit(OVR_TS).alias("A11_PDAY3414"),
            pl.lit(OVR_TS).alias("A11_PDAY1413"),
            pl.lit(OVR_TS).alias("A11_PDAY2413"),
            pl.lit(OVR_TS).alias("A11_PDAY3413"),
            pl.lit(OVR_TS).alias("A11_PDAY1403"),
            pl.lit(OVR_TS).alias("A11_PDAY2403"),
            pl.lit(OVR_TS).alias("A11_PDAY3403"),
        ])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def latest30(rk: int) -> date:
        row = _pick_rank_row(r30, key_val, rk)
        return _date_or_ovr(row if _n_or(row, "n_txn") > 0 else pl.DataFrame(), "latest_dt")

    def latest90(rk: int) -> date:
        row = _pick_rank_row(r90, key_val, rk)
        return _date_or_ovr(row if _n_or(row, "n_txn") > 0 else pl.DataFrame(), "latest_dt")

    def earliest90(rk: int) -> date:
        row = _pick_rank_row(r90, key_val, rk)
        return _date_or_ovr(row if _n_or(row, "n_txn") > 0 else pl.DataFrame(), "earliest_dt")

    return base.with_columns([
        pl.lit(latest30(1)).alias("A11_PDAY1414"),
        pl.lit(latest30(2)).alias("A11_PDAY2414"),
        pl.lit(latest30(3)).alias("A11_PDAY3414"),
        pl.lit(latest90(1)).alias("A11_PDAY1413"),
        pl.lit(latest90(2)).alias("A11_PDAY2413"),
        pl.lit(latest90(3)).alias("A11_PDAY3413"),
        pl.lit(earliest90(1)).alias("A11_PDAY1403"),
        pl.lit(earliest90(2)).alias("A11_PDAY2403"),
        pl.lit(earliest90(3)).alias("A11_PDAY3403"),
    ])

# 109–126: Numeric day-diff features (Float64)
def features_109_126_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    ref_date: date,
    override_value: float | None,
) -> pl.DataFrame:
    """
    Day differences per rank:
      ref - latest_30d  -> A11_PDAY1424 / 2424 / 3424
      ref - latest_90d  -> A11_PDAY1423 / 2423 / 3423
      ref - earliest_30d-> A11_PDAY1444 / 2444 / 3444
      ref - earliest_90d-> A11_PDAY1443 / 2443 / 3443
      latest_30d - earliest_30d -> A11_PDAY1454 / 2454 / 3454
      latest_90d - earliest_90d -> A11_PDAY1453 / 2453 / 3453
    If n_txn==0 in the corresponding window: OVR["NO_TXN"].
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})

    if override_value is not None:
        cols = [
            "A11_PDAY1424","A11_PDAY2424","A11_PDAY3424",
            "A11_PDAY1423","A11_PDAY2423","A11_PDAY3423",
            "A11_PDAY1444","A11_PDAY2444","A11_PDAY3444",
            "A11_PDAY1443","A11_PDAY2443","A11_PDAY3443",
            "A11_PDAY1454","A11_PDAY2454","A11_PDAY3454",
            "A11_PDAY1453","A11_PDAY2453","A11_PDAY3453",
        ]
        return base.with_columns([pl.lit(OVR["NO_TXN"]).alias(c) for c in cols])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def ddays(a: date, b: date) -> float:
        return float((a - b).days)

    def diffs_for(B: pl.DataFrame, rk: int, which: str) -> tuple[bool, date]:
        """
        Return (ok, date) where 'ok' means n_txn>0 in this window.
        which in {"latest_dt","earliest_dt"}.
        """
        row = _pick_rank_row(B, key_val, rk)
        n = _n_or(row, "n_txn", 0)
        if n == 0:
            return (False, OVR_TS)
        dt = row[which][0]
        return (True, dt if dt is not None else OVR_TS)

    vals = {}
    for rk in (1,2,3):
        ok_l30, last30 = diffs_for(r30, rk, "latest_dt")
        ok_e30, ear30  = diffs_for(r30, rk, "earliest_dt")
        ok_l90, last90 = diffs_for(r90, rk, "latest_dt")
        ok_e90, ear90  = diffs_for(r90, rk, "earliest_dt")

        # ref - latest
        vals[(rk, "1424")] = ddays(ref_date, last30) if ok_l30 else OVR["NO_TXN"]
        vals[(rk, "1423")] = ddays(ref_date, last90) if ok_l90 else OVR["NO_TXN"]
        # ref - earliest
        vals[(rk, "1444")] = ddays(ref_date, ear30)  if ok_e30 else OVR["NO_TXN"]
        vals[(rk, "1443")] = ddays(ref_date, ear90)  if ok_e90 else OVR["NO_TXN"]
        # latest - earliest (within window)
        vals[(rk, "1454")] = ddays(last30, ear30)    if (ok_l30 and ok_e30) else OVR["NO_TXN"]
        vals[(rk, "1453")] = ddays(last90, ear90)    if (ok_l90 and ok_e90) else OVR["NO_TXN"]

    out = base.with_columns([
        # ref - latest_30d
        pl.lit(vals[(1,"1424")]).alias("A11_PDAY1424"),
        pl.lit(vals[(2,"1424")]).alias("A11_PDAY2424"),
        pl.lit(vals[(3,"1424")]).alias("A11_PDAY3424"),
        # ref - latest_90d
        pl.lit(vals[(1,"1423")]).alias("A11_PDAY1423"),
        pl.lit(vals[(2,"1423")]).alias("A11_PDAY2423"),
        pl.lit(vals[(3,"1423")]).alias("A11_PDAY3423"),
        # ref - earliest_30d
        pl.lit(vals[(1,"1444")]).alias("A11_PDAY1444"),
        pl.lit(vals[(2,"1444")]).alias("A11_PDAY2444"),
        pl.lit(vals[(3,"1444")]).alias("A11_PDAY3444"),
        # ref - earliest_90d
        pl.lit(vals[(1,"1443")]).alias("A11_PDAY1443"),
        pl.lit(vals[(2,"1443")]).alias("A11_PDAY2443"),
        pl.lit(vals[(3,"1443")]).alias("A11_PDAY3443"),
        # latest - earliest (span) 30d
        pl.lit(vals[(1,"1454")]).alias("A11_PDAY1454"),
        pl.lit(vals[(2,"1454")]).alias("A11_PDAY2454"),
        pl.lit(vals[(3,"1454")]).alias("A11_PDAY3454"),
        # latest - earliest (span) 90d
        pl.lit(vals[(1,"1453")]).alias("A11_PDAY1453"),
        pl.lit(vals[(2,"1453")]).alias("A11_PDAY2453"),
        pl.lit(vals[(3,"1453")]).alias("A11_PDAY3453"),
    ])
    # numeric casts
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in out.columns if c != KEY])

# -------- Wrapper: combine 100–126 --------
def build_features_block_100_126(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    ref_date: date,
    override_value: float | None,
) -> pl.DataFrame:
    f100_108 = features_100_108_v2(bases, key_val, override_value)
    f109_126 = features_109_126_v2(bases, key_val, ref_date, override_value)

    out = ensure_key_i64(f100_108)
    out = out.join(ensure_key_i64(f109_126), on=KEY, how="left")
    return out

# ==============================
# Copilot_Fast_Pipeline_v2 - PART 4F: Features 127–174 (Amount families) - CORRECTED
# ==============================

def _pick_rank_row(base: pl.DataFrame, key_val: int, rk: int) -> pl.DataFrame:
    return base.filter((pl.col(KEY) == key_val) & (pl.col("rank") == rk))

def _is_no_txn_row(row: pl.DataFrame) -> bool:
    return (row.height == 0) or (int(row["n_txn"][0] or 0) == 0)

def _amt_or_no_txn(row: pl.DataFrame, col: str) -> float:
    if _is_no_txn_row(row):
        return OVR["NO_TXN"]
    v = row[col][0]
    return float(v) if v is not None else OVR["NO_TXN"]

def ratio_amount_py(num: float, den: float) -> float:
    """
    Amount-ratio override semantics (your pipeline):
      BOTH0 -> if num==0 and den==0
      DEN0  -> if den==0 (and num!=0)
      NUM0  -> if num==0 (and den>0)
      else  -> num/den
    """
    # guard None/NaN
    if num is None: num = 0.0
    if den is None: den = 0.0
    if num == 0.0 and den == 0.0:
        return OVR["BOTH0"]
    if den == 0.0:
        return OVR["DEN0"]
    if num == 0.0:
        return OVR["NUM0"]
    return num / den

# -------- 127–132: Sums by rank (30d & 90d) --------
def features_127_132_v2(
    bases: dict[str, pl.DataFrame], key_val: int, override_value: float | None
) -> pl.DataFrame:
    """
    30d SUM: A11_PDAY1504 / 2504 / 3504
    90d SUM: A11_PDAY1503 / 2503 / 3503
    if n_txn==0 -> OVR["NO_TXN"]
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        cols = ["A11_PDAY1504","A11_PDAY2504","A11_PDAY3504","A11_PDAY1503","A11_PDAY2503","A11_PDAY3503"]
        return base.with_columns([pl.lit(override_value).alias(c) for c in cols])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]
    def s(B: pl.DataFrame, rk: int) -> float: return _amt_or_no_txn(_pick_rank_row(B, key_val, rk), "sum_amt")

    out = base.with_columns([
        pl.lit(s(r30,1)).alias("A11_PDAY1504"),
        pl.lit(s(r30,2)).alias("A11_PDAY2504"),
        pl.lit(s(r30,3)).alias("A11_PDAY3504"),
        pl.lit(s(r90,1)).alias("A11_PDAY1503"),
        pl.lit(s(r90,2)).alias("A11_PDAY2503"),
        pl.lit(s(r90,3)).alias("A11_PDAY3503"),
    ])
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in out.columns if c != KEY])

# -------- 133–138: % of 180d total (rank-agnostic denom) --------
def features_133_138_v2(
    bases: dict[str, pl.DataFrame], key_val: int, override_value: float | None
) -> pl.DataFrame:
    """
    Numerators: SUM_30d (ranks 1–3) and SUM_90d (ranks 1–3)
    Denominator: agn_0_180.sum_amt for this key (rank-agnostic)
    30d %: A11_PDAY1514 / 2514 / 3514
    90d %: A11_PDAY1513 / 2513 / 3513
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        cols = ["A11_PDAY1514","A11_PDAY2514","A11_PDAY3514","A11_PDAY1513","A11_PDAY2513","A11_PDAY3513"]
        return base.with_columns([pl.lit(override_value).alias(c) for c in cols])

    agn180 = bases["agn_0_180"].filter(pl.col(KEY)==key_val)
    denom = float(agn180["sum_amt"][0]) if agn180.height else 0.0

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def pct(B: pl.DataFrame, rk: int) -> float:
        num = _amt_or_no_txn(_pick_rank_row(B, key_val, rk), "sum_amt")
        if num == OVR["NO_TXN"]:
            return OVR["NO_TXN"]
        return ratio_amount_py(num, denom)

    out = base.with_columns([
        pl.lit(pct(r30,1)).alias("A11_PDAY1514"),
        pl.lit(pct(r30,2)).alias("A11_PDAY2514"),
        pl.lit(pct(r30,3)).alias("A11_PDAY3514"),
        pl.lit(pct(r90,1)).alias("A11_PDAY1513"),
        pl.lit(pct(r90,2)).alias("A11_PDAY2513"),
        pl.lit(pct(r90,3)).alias("A11_PDAY3513"),
    ])
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in out.columns if c != KEY])

# -------- 139–144: Means by rank (30d & 90d) --------
def features_139_144_v2(
    bases: dict[str, pl.DataFrame], key_val: int, override_value: float | None
) -> pl.DataFrame:
    """
    30d MEAN: A11_PDAY1524 / 2524 / 3524
    90d MEAN: A11_PDAY1523 / 2523 / 3523
    if n_txn==0 -> OVR["NO_TXN"]
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        cols = ["A11_PDAY1524","A11_PDAY2524","A11_PDAY3524","A11_PDAY1523","A11_PDAY2523","A11_PDAY3523"]
        return base.with_columns([pl.lit(override_value).alias(c) for c in cols])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]
    def m(B: pl.DataFrame, rk: int) -> float: return _amt_or_no_txn(_pick_rank_row(B, key_val, rk), "mean_amt")

    out = base.with_columns([
        pl.lit(m(r30,1)).alias("A11_PDAY1524"),
        pl.lit(m(r30,2)).alias("A11_PDAY2524"),
        pl.lit(m(r30,3)).alias("A11_PDAY3524"),
        pl.lit(m(r90,1)).alias("A11_PDAY1523"),
        pl.lit(m(r90,2)).alias("A11_PDAY2523"),
        pl.lit(m(r90,3)).alias("A11_PDAY3523"),
    ])
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in out.columns if c != KEY])

# -------- 145–150: Latest / Avg amounts --------
def features_145_150_v2(
    bases: dict[str, pl.DataFrame], key_val: int, override_value: float | None
) -> pl.DataFrame:
    """
    (latest_amt / mean_amt)
    30d: A11_PDAY1534 / 2534 / 3534
    90d: A11_PDAY1533 / 2533 / 3533
    ratio_amount_py semantics
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        cols = ["A11_PDAY1534","A11_PDAY2534","A11_PDAY3534","A11_PDAY1533","A11_PDAY2533","A11_PDAY3533"]
        return base.with_columns([pl.lit(override_value).alias(c) for c in cols])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def ratio_latest_mean(B: pl.DataFrame, rk: int) -> float:
        row = _pick_rank_row(B, key_val, rk)
        if _is_no_txn_row(row):
            return OVR["NO_TXN"]
        num = float(row["latest_amt"][0] or 0.0)
        den = float(row["mean_amt"][0]   or 0.0)
        return ratio_amount_py(num, den)

    out = base.with_columns([
        pl.lit(ratio_latest_mean(r30,1)).alias("A11_PDAY1534"),
        pl.lit(ratio_latest_mean(r30,2)).alias("A11_PDAY2534"),
        pl.lit(ratio_latest_mean(r30,3)).alias("A11_PDAY3534"),
        pl.lit(ratio_latest_mean(r90,1)).alias("A11_PDAY1533"),
        pl.lit(ratio_latest_mean(r90,2)).alias("A11_PDAY2533"),
        pl.lit(ratio_latest_mean(r90,3)).alias("A11_PDAY3533"),
    ])
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in out.columns if c != KEY])

# -------- 151–156: (Max - Min) / Avg amounts --------
def features_151_156_v2(
    bases: dict[str, pl.DataFrame], key_val: int, override_value: float | None
) -> pl.DataFrame:
    """
    30d: A11_PDAY1544 / 2544 / 3544
    90d: A11_PDAY1543 / 2543 / 3543
    ratio_amount_py semantics
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        cols = ["A11_PDAY1544","A11_PDAY2544","A11_PDAY3544","A11_PDAY1543","A11_PDAY2543","A11_PDAY3543"]
        return base.with_columns([pl.lit(override_value).alias(c) for c in cols])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def rr(B: pl.DataFrame, rk: int) -> float:
        row = _pick_rank_row(B, key_val, rk)
        if _is_no_txn_row(row):
            return OVR["NO_TXN"]
        maxv = float(row["max_amt"][0]  or 0.0)
        minv = float(row["min_amt"][0]  or 0.0)
        avgv = float(row["mean_amt"][0] or 0.0)
        num = maxv - minv
        den = avgv
        return ratio_amount_py(num, den)

    out = base.with_columns([
        pl.lit(rr(r30,1)).alias("A11_PDAY1544"),
        pl.lit(rr(r30,2)).alias("A11_PDAY2544"),
        pl.lit(rr(r30,3)).alias("A11_PDAY3544"),
        pl.lit(rr(r90,1)).alias("A11_PDAY1543"),
        pl.lit(rr(r90,2)).alias("A11_PDAY2543"),
        pl.lit(rr(r90,3)).alias("A11_PDAY3543"),
    ])
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in out.columns if c != KEY])

# -------- 157–162: Std / Mean (CV) of amounts --------
def features_157_162_v2(
    bases: dict[str, pl.DataFrame], key_val: int, override_value: float | None
) -> pl.DataFrame:
    """
    30d: A11_PDAY1554 / 2554 / 3554
    90d: A11_PDAY1553 / 2553 / 3553
    ratio_amount_py semantics
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        cols = ["A11_PDAY1554","A11_PDAY2554","A11_PDAY3554","A11_PDAY1553","A11_PDAY2553","A11_PDAY3553"]
        return base.with_columns([pl.lit(override_value).alias(c) for c in cols])

    r30 = bases["rank_0_30"]; r90 = bases["rank_0_90"]

    def cv(B: pl.DataFrame, rk: int) -> float:
        row = _pick_rank_row(B, key_val, rk)
        if _is_no_txn_row(row):
            return OVR["NO_TXN"]
        num = float(row["std_amt"][0]  or 0.0)
        den = float(row["mean_amt"][0] or 0.0)
        return ratio_amount_py(num, den)

    out = base.with_columns([
        pl.lit(cv(r30,1)).alias("A11_PDAY1554"),
        pl.lit(cv(r30,2)).alias("A11_PDAY2554"),
        pl.lit(cv(r30,3)).alias("A11_PDAY3554"),
        pl.lit(cv(r90,1)).alias("A11_PDAY1553"),
        pl.lit(cv(r90,2)).alias("A11_PDAY2553"),
        pl.lit(cv(r90,3)).alias("A11_PDAY3553"),
    ])
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in out.columns if c != KEY])

# -------- 163–168: Cross-window SUM ratios (30d / 31–60d, 30d / 61–90d) --------
def features_163_168_v2(
    bases: dict[str, pl.DataFrame], key_val: int, override_value: float | None
) -> pl.DataFrame:
    """
    30d / 31–60d SUM: A11_PDAY1564 / 2564 / 3564
    30d / 61–90d SUM: A11_PDAY1574 / 2574 / 3574
    ratio_amount_py semantics
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        cols = ["A11_PDAY1564","A11_PDAY2564","A11_PDAY3564","A11_PDAY1574","A11_PDAY2574","A11_PDAY3574"]
        return base.with_columns([pl.lit(override_value).alias(c) for c in cols])

    r30, r31_60, r61_90 = bases["rank_0_30"], bases["rank_31_60"], bases["rank_61_90"]

    def ratio_sum(nB: pl.DataFrame, dB: pl.DataFrame, rk: int) -> float:
        nrow = _pick_rank_row(nB, key_val, rk)
        drow = _pick_rank_row(dB, key_val, rk)
        if _is_no_txn_row(nrow) or _is_no_txn_row(drow):
            return OVR["NO_TXN"]
        n = float(nrow["sum_amt"][0] or 0.0)
        d = float(drow["sum_amt"][0] or 0.0)
        return ratio_amount_py(n, d)

    out = base.with_columns([
        pl.lit(ratio_sum(r30, r31_60, 1)).alias("A11_PDAY1564"),
        pl.lit(ratio_sum(r30, r31_60, 2)).alias("A11_PDAY2564"),
        pl.lit(ratio_sum(r30, r31_60, 3)).alias("A11_PDAY3564"),
        pl.lit(ratio_sum(r30, r61_90, 1)).alias("A11_PDAY1574"),
        pl.lit(ratio_sum(r30, r61_90, 2)).alias("A11_PDAY2574"),
        pl.lit(ratio_sum(r30, r61_90, 3)).alias("A11_PDAY3574"),
    ])
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in out.columns if c != KEY])

# -------- 169–174: Cross-window MEAN ratios (30d / 31–60d, 30d / 61–90d) --------
def features_169_174_v2(
    bases: dict[str, pl.DataFrame], key_val: int, override_value: float | None
) -> pl.DataFrame:
    """
    30d / 31–60d MEAN: A11_PDAY1584 / 2584 / 3584
    30d / 61–90d MEAN: A11_PDAY1594 / 2594 / 3594
    ratio_amount_py semantics
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        cols = ["A11_PDAY1584","A11_PDAY2584","A11_PDAY3584","A11_PDAY1594","A11_PDAY2594","A11_PDAY3594"]
        return base.with_columns([pl.lit(override_value).alias(c) for c in cols])

    r30, r31_60, r61_90 = bases["rank_0_30"], bases["rank_31_60"], bases["rank_61_90"]

    def ratio_mean(nB: pl.DataFrame, dB: pl.DataFrame, rk: int) -> float:
        nrow = _pick_rank_row(nB, key_val, rk)
        drow = _pick_rank_row(dB, key_val, rk)
        if _is_no_txn_row(nrow) or _is_no_txn_row(drow):
            return OVR["NO_TXN"]
        n = float(nrow["mean_amt"][0] or 0.0)
        d = float(drow["mean_amt"][0] or 0.0)
        return ratio_amount_py(n, d)

    out = base.with_columns([
        pl.lit(ratio_mean(r30, r31_60, 1)).alias("A11_PDAY1584"),
        pl.lit(ratio_mean(r30, r31_60, 2)).alias("A11_PDAY2584"),
        pl.lit(ratio_mean(r30, r31_60, 3)).alias("A11_PDAY3584"),
        pl.lit(ratio_mean(r30, r61_90, 1)).alias("A11_PDAY1594"),
        pl.lit(ratio_mean(r30, r61_90, 2)).alias("A11_PDAY2594"),
        pl.lit(ratio_mean(r30, r61_90, 3)).alias("A11_PDAY3594"),
    ])
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in out.columns if c != KEY])

# -------- Wrapper: combine 127–174 --------
def build_features_block_127_174(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    f127_132 = features_127_132_v2(bases, key_val, override_value)
    f133_138 = features_133_138_v2(bases, key_val, override_value)
    f139_144 = features_139_144_v2(bases, key_val, override_value)
    f145_150 = features_145_150_v2(bases, key_val, override_value)
    f151_156 = features_151_156_v2(bases, key_val, override_value)
    f157_162 = features_157_162_v2(bases, key_val, override_value)
    f163_168 = features_163_168_v2(bases, key_val, override_value)
    f169_174 = features_169_174_v2(bases, key_val, override_value)

    out = ensure_key_i64(f127_132)
    for b in (f133_138, f139_144, f145_150, f151_156, f157_162, f163_168, f169_174):
        out = out.join(ensure_key_i64(b), on=KEY, how="left")
    return out

# ==============================
# Copilot_Fast_Pipeline_v2 - PART 5A: Features 181–198 (Estimated Next Paydates)
# ==============================

def _pick_rank_row(base: pl.DataFrame, key_val: int, rk: int) -> pl.DataFrame:
    return base.filter((pl.col(KEY) == key_val) & (pl.col("rank") == rk))

def _safe_dt(v) -> date | None:
    return v if isinstance(v, date) else None

def _est_dates_for_rank(row: pl.DataFrame, ref_date: date) -> list[date]:
    """
    Return 6 estimated next paydates for this (key, rank) row using:
        est_k = latest_dt + k * mode_interval (k = 1..6)
    Overrides:
      - n_txn==0 -> [OVR_TS]*6
      - n_txn==1 -> [OVR_EST_ONE]*6
      - any est < ref_date -> OVR_EST_BEFORE for that est
      - missing/zero mode_interval -> [OVR_EST_BEFORE]*6
    """
    if row.height == 0:
        return [OVR_TS]*6
    n = int(row["n_txn"][0] or 0)
    if n == 0:
        return [OVR_TS]*6
    if n == 1:
        return [OVR_EST_ONE]*6

    last_dt = _safe_dt(row["latest_dt"][0])
    mode_iv = row["mode_interval"][0]
    if last_dt is None or mode_iv is None or int(mode_iv) <= 0:
        return [OVR_EST_BEFORE]*6

    iv = int(mode_iv)
    ests = [last_dt + timedelta(days=iv*k) for k in range(1,7)]
    # mask: any estimate BEFORE ref_date -> OVR_EST_BEFORE
    return [e if e >= ref_date else OVR_EST_BEFORE for e in ests]

def features_181_198_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    ref_date: date,
    override_value: float | None,
) -> pl.DataFrame:
    """
    Output (dates):
      Rank 1: A11_PDAY1710, 1720, 1730, 1740, 1750, 1760
      Rank 2: A11_PDAY2710, 2720, 2730, 2740, 2750, 2760
      Rank 3: A11_PDAY3710, 3720, 3730, 3740, 3750, 3760
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})

    # Group A: set all to OVR_TS
    if override_value is not None:
        cols = [
            # rank 1
            "A11_PDAY1710","A11_PDAY1720","A11_PDAY1730","A11_PDAY1740","A11_PDAY1750","A11_PDAY1760",
            # rank 2
            "A11_PDAY2710","A11_PDAY2720","A11_PDAY2730","A11_PDAY2740","A11_PDAY2750","A11_PDAY2760",
            # rank 3
            "A11_PDAY3710","A11_PDAY3720","A11_PDAY3730","A11_PDAY3740","A11_PDAY3750","A11_PDAY3760",
        ]
        return base.with_columns([pl.lit(OVR_TS).alias(c) for c in cols])

    r90 = bases["rank_0_90"]

    # build 6 dates per rank
    ests = {}
    for rk, cols in [
        (1, ["A11_PDAY1710","A11_PDAY1720","A11_PDAY1730","A11_PDAY1740","A11_PDAY1750","A11_PDAY1760"]),
        (2, ["A11_PDAY2710","A11_PDAY2720","A11_PDAY2730","A11_PDAY2740","A11_PDAY2750","A11_PDAY2760"]),
        (3, ["A11_PDAY3710","A11_PDAY3720","A11_PDAY3730","A11_PDAY3740","A11_PDAY3750","A11_PDAY3760"]),
    ]:
        row = _pick_rank_row(r90, key_val, rk)
        six = _est_dates_for_rank(row, ref_date)
        for c, v in zip(cols, six):
            ests[c] = v

    return base.with_columns([pl.lit(ests[c]).alias(c) for c in ests.keys()])

def build_features_block_181_198(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    ref_date: date,
    override_value: float | None,
) -> pl.DataFrame:
    return features_181_198_v2(bases, key_val, ref_date, override_value)

# ==============================
# Copilot_Fast_Pipeline_v2 - PART 5B: Features 175–180 (Stability)
# ==============================

def _pick_rank_row(base: pl.DataFrame, key_val: int, rk: int) -> pl.DataFrame:
    return base.filter((pl.col(KEY) == key_val) & (pl.col("rank") == rk))

def _stability_components_from_row(row: pl.DataFrame) -> tuple[int, float, float, float]:
    """
    Returns (n_txn, mode_dom, cv_intervals, missed_ratio) for a single (key, rank) row
    from a ranked base. Fully vectorized except for the final tiny Python math over a
    (short) list (the intervals list).
    """
    if row.height == 0:
        return (0, 0.0, 0.0, 0.0)

    n = int(row["n_txn"][0] or 0)
    md = float(row["mode_dom"][0] or 0.0)

    # sanitize intervals list (drop None)
    iv_raw = row["intervals"][0]
    iv = [int(x) for x in iv_raw if x is not None] if isinstance(iv_raw, list) else []

    # CV (sample std / mean); for 0/1-length lists, std = 0
    if len(iv) == 0:
        cv = 0.0
        iv_len = 0
    else:
        m = sum(iv) / len(iv)
        if len(iv) >= 2:
            var = sum((x - m) ** 2 for x in iv) / (len(iv) - 1)
            cv = (var ** 0.5) / m if m != 0 else 0.0
        else:
            cv = 0.0
        iv_len = len(iv)

    # missed_ratio: > mode_interval occurrences divided by number of intervals
    mc = int(row["missed_count"][0] or 0)
    missed_ratio = (mc / iv_len) if iv_len > 0 else 0.0

    return (n, md, cv, missed_ratio)

def _stability_score(md: float, cv: float, miss: float,
                     w_dom: float = 0.50, w_cv: float = 0.30, w_miss: float = 0.20,
                     cv_cap: float = 1.5) -> float:
    """
    Combine components into a 0..100 stability score.
      - md in [0,1] (higher is better)
      - cv penalized: use 1 - min(cv, cv_cap)/cv_cap
      - miss penalized: use 1 - miss (share of 'missed' intervals)
    Weights (w_dom, w_cv, w_miss) sum to 1.0 by default.
    """
    cv_term = 1.0 - min(max(cv, 0.0), cv_cap) / cv_cap
    miss_term = 1.0 - min(max(miss, 0.0), 1.0)
    score = (w_dom * md + w_cv * cv_term + w_miss * miss_term)
    # clamp
    return float(max(0.0, min(1, score)))

def _stability_for_rank(base: pl.DataFrame, key_val: int, rk: int,
                        override_value: float | None) -> float:
    """
    Apply the same override semantics used elsewhere:
      - Group A (override_value is not None) handled in the caller by broadcasting override.
      - If n_txn==0  -> OVR["NO_TXN"]
      - If n_txn==1  -> OVR["ONE_TXN"]
      - Else compute score from components.
    """
    row = _pick_rank_row(base, key_val, rk)
    n, md, cv, miss = _stability_components_from_row(row)

    if n == 0:
        return OVR["NO_TXN"]
    if n == 1:
        return OVR["ONE_TXN"]
    return _stability_score(md, cv, miss)

def features_175_180_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    Outputs (Float64):
      30d (rank 1..3): A11_PDAY1604 / 2604 / 3604
      90d (rank 1..3): A11_PDAY1603 / 2603 / 3603
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})

    # Group A: everything becomes the same numeric override (as in your pipeline)
    if override_value is not None:
        return base.with_columns([
            pl.lit(override_value).alias("A11_PDAY1604"),
            pl.lit(override_value).alias("A11_PDAY2604"),
            pl.lit(override_value).alias("A11_PDAY3604"),
            pl.lit(override_value).alias("A11_PDAY1603"),
            pl.lit(override_value).alias("A11_PDAY2603"),
            pl.lit(override_value).alias("A11_PDAY3603"),
        ])

    r30 = bases["rank_0_30"]
    r90 = bases["rank_0_90"]

    out = base.with_columns([
        # 30d
        pl.lit(_stability_for_rank(r30, key_val, 1, override_value)).alias("A11_PDAY1604"),
        pl.lit(_stability_for_rank(r30, key_val, 2, override_value)).alias("A11_PDAY2604"),
        pl.lit(_stability_for_rank(r30, key_val, 3, override_value)).alias("A11_PDAY3604"),
        # 90d
        pl.lit(_stability_for_rank(r90, key_val, 1, override_value)).alias("A11_PDAY1603"),
        pl.lit(_stability_for_rank(r90, key_val, 2, override_value)).alias("A11_PDAY2603"),
        pl.lit(_stability_for_rank(r90, key_val, 3, override_value)).alias("A11_PDAY3603"),
    ])

    return out.with_columns([pl.col(c).cast(pl.Float64) for c in out.columns if c != KEY])

def build_features_block_175_180(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    return features_175_180_v2(bases, key_val, override_value)

# ==============================
# Copilot_Fast_Pipeline_v2 - PART 6: Rank-agnostic 199–235 + Final Assembler
# ==============================

# ---------- small helpers ----------
def _agn_row(bases: dict[str, pl.DataFrame], win_key: str, key_val: int) -> pl.DataFrame:
    return bases[win_key].filter(pl.col(KEY) == key_val)

def _interval_cv_from_list(ivals: list[int] | None) -> tuple[float, float, int]:
    """Return (cv, mean, n) from a sanitized interval list (drop None)."""
    if not isinstance(ivals, list):
        return (0.0, 0.0, 0)
    xs = [int(x) for x in ivals if x is not None]
    if len(xs) == 0:
        return (0.0, 0.0, 0)
    m = sum(xs) / len(xs)
    if len(xs) >= 2:
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)  # sample
        sd = var ** 0.5
    else:
        sd = 0.0
    cv = (sd / m) if m != 0 else 0.0
    return (cv, m, len(xs))

def ratio_amount_py(num: float, den: float) -> float:
    """Amount-ratio semantics used in PIECE 4F."""
    if num is None: num = 0.0
    if den is None: den = 0.0
    if num == 0.0 and den == 0.0: return OVR["BOTH0"]
    if den == 0.0: return OVR["DEN0"]
    if num == 0.0: return OVR["NUM0"]
    return num / den

def ratio_interval_py(num: float, den: float) -> float:
    """Interval-ratio semantics used in PIECE 4B/4C (BOTH0/DEN0/NUM0)."""
    if num is None: num = 0.0
    if den is None: den = 0.0
    if num == 0.0 and den == 0.0: return OVR["BOTH0"]
    if den == 0.0: return OVR["DEN0"]
    if num == 0.0: return OVR["NUM0"]
    return num / den

# ---------- 199–205: agnostic date features (outputs are DATE columns) ----------
def features_agn_dates_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    Date outputs:
      A11_PDAY4414 : latest paydate in 30d (agn)
      A11_PDAY4413 : latest paydate in 90d (agn)
      A11_PDAY4403 : earliest paydate in 90d (agn)
    (We keep to the '44xx' code family for rank-agnostic dates as used in your A11_DATE list.)
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})

    if override_value is not None:
        return base.with_columns([
            pl.lit(OVR_TS).alias("A11_PDAY4414"),
            pl.lit(OVR_TS).alias("A11_PDAY4413"),
            pl.lit(OVR_TS).alias("A11_PDAY4403"),
        ])

    r30 = _agn_row(bases, "agn_0_30", key_val)
    r90 = _agn_row(bases, "agn_0_90", key_val)

    def pick_date(row: pl.DataFrame, col: str) -> date:
        if row.height == 0 or int(row["n_txn"][0] or 0) == 0:
            return OVR_TS
        v = row[col][0]
        return v if v is not None else OVR_TS

    return base.with_columns([
        pl.lit(pick_date(r30, "latest_dt")).alias("A11_PDAY4414"),
        pl.lit(pick_date(r90, "latest_dt")).alias("A11_PDAY4413"),
        pl.lit(pick_date(r90, "earliest_dt")).alias("A11_PDAY4403"),
    ])

# ---------- 206–216: agnostic day-diffs (Float)  (#216 included: ref - latest_90d) ----------
def features_agn_daydiffs_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    ref_date: date,
    override_value: float | None,
) -> pl.DataFrame:
    """
    Numeric diffs:
      A11_PDAY4424 : ref - latest_30d
      A11_PDAY4423 : ref - latest_90d   <-- (#216 corrected)
      A11_PDAY4444 : ref - earliest_30d
      A11_PDAY4443 : ref - earliest_90d
      A11_PDAY4454 : latest_30d - earliest_30d (span)
      A11_PDAY4453 : latest_90d - earliest_90d (span)
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        cols = ["A11_PDAY4424","A11_PDAY4423","A11_PDAY4444","A11_PDAY4443","A11_PDAY4454","A11_PDAY4453"]
        return base.with_columns([pl.lit(OVR["NO_TXN"]).alias(c) for c in cols])

    r30 = _agn_row(bases, "agn_0_30", key_val)
    r90 = _agn_row(bases, "agn_0_90", key_val)

    def dd(a: date, b: date) -> float:
        return float((a - b).days)

    def pick_dd(row: pl.DataFrame, col: str, *others: str) -> tuple[bool, date]:
        if row.height == 0 or int(row["n_txn"][0] or 0) == 0:
            return (False, OVR_TS)
        v = row[col][0]
        return (True, v if v is not None else OVR_TS)

    ok_l30, last30 = pick_dd(r30, "latest_dt")
    ok_e30, ear30  = pick_dd(r30, "earliest_dt")
    ok_l90, last90 = pick_dd(r90, "latest_dt")
    ok_e90, ear90  = pick_dd(r90, "earliest_dt")

    out = base.with_columns([
        pl.lit(dd(ref_date, last30) if ok_l30 else OVR["NO_TXN"]).alias("A11_PDAY4424"),
        pl.lit(dd(ref_date, last90) if ok_l90 else OVR["NO_TXN"]).alias("A11_PDAY4423"),  # #216
        pl.lit(dd(ref_date, ear30)  if ok_e30 else OVR["NO_TXN"]).alias("A11_PDAY4444"),
        pl.lit(dd(ref_date, ear90)  if ok_e90 else OVR["NO_TXN"]).alias("A11_PDAY4443"),
        pl.lit(dd(last30, ear30)    if (ok_l30 and ok_e30) else OVR["NO_TXN"]).alias("A11_PDAY4454"),
        pl.lit(dd(last90, ear90)    if (ok_l90 and ok_e90) else OVR["NO_TXN"]).alias("A11_PDAY4453"),
    ])
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in out.columns if c != KEY])

# ---------- 217–228: agnostic interval stats (Float) ----------
def features_agn_intervals_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    Intervals (rank-agnostic):
      Mode interval:  A11_PDAY4204 (30d), A11_PDAY4203 (90d)
      Mode dominance%:A11_PDAY4214 (30d), A11_PDAY4213 (90d)
      Large count:    A11_PDAY4224 (30d), A11_PDAY4223 (90d)
      Recent/Mode:    A11_PDAY4234 (30d), A11_PDAY4233 (90d)
      Max/Min/Avg:    A11_PDAY4244/4254/4264 (30d), 4243/4253/4263 (90d)
      Range ratio:    A11_PDAY4274 (30d), A11_PDAY4273 (90d)
      CV intervals:   A11_PDAY4284 (30d), A11_PDAY4283 (90d)
    Interval overrides:
      n_txn==0 -> NO_TXN; n_txn==1 -> ONE_TXN (except ratios which follow ratio semantics)
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        cols = [
            "A11_PDAY4204","A11_PDAY4203","A11_PDAY4214","A11_PDAY4213","A11_PDAY4224","A11_PDAY4223",
            "A11_PDAY4234","A11_PDAY4233","A11_PDAY4244","A11_PDAY4254","A11_PDAY4264",
            "A11_PDAY4243","A11_PDAY4253","A11_PDAY4263","A11_PDAY4274","A11_PDAY4273",
            "A11_PDAY4284","A11_PDAY4283",
        ]
        return base.with_columns([pl.lit(override_value).alias(c) for c in cols])

    a30 = _agn_row(bases, "agn_0_30", key_val)
    a90 = _agn_row(bases, "agn_0_90", key_val)

    def ints_row_val(A: pl.DataFrame, col: str, *, as_pct: bool = False) -> float:
        if A.height == 0:
            return OVR["NO_TXN"]
        n = int(A["n_txn"][0] or 0)
        if n == 0: return OVR["NO_TXN"]
        if n == 1:
            # For scalars (mode, stats) we flag ONE_TXN; for dominance% and ratios we emit valid numbers.
            if col in ("mode_dom", "recent_interval", "mode_interval", "max_interval", "min_interval", "avg_interval"):
                return OVR["ONE_TXN"]
        v = A[col][0]
        if v is None:
            return OVR["NO_TXN"]
        return float(v * 100.0) if as_pct else float(v)

    # Scalars
    out = base.with_columns([
        pl.lit(ints_row_val(a30, "mode_interval")).alias("A11_PDAY4204"),
        pl.lit(ints_row_val(a90, "mode_interval")).alias("A11_PDAY4203"),
        pl.lit(ints_row_val(a30, "mode_dom", as_pct=True)).alias("A11_PDAY4214"),
        pl.lit(ints_row_val(a90, "mode_dom", as_pct=True)).alias("A11_PDAY4213"),
        pl.lit(ints_row_val(a30, "large_count")).alias("A11_PDAY4224"),
        pl.lit(ints_row_val(a90, "large_count")).alias("A11_PDAY4223"),
        pl.lit(ints_row_val(a30, "max_interval")).alias("A11_PDAY4244"),
        pl.lit(ints_row_val(a30, "min_interval")).alias("A11_PDAY4254"),
        pl.lit(ints_row_val(a30, "avg_interval")).alias("A11_PDAY4264"),
        pl.lit(ints_row_val(a90, "max_interval")).alias("A11_PDAY4243"),
        pl.lit(ints_row_val(a90, "min_interval")).alias("A11_PDAY4253"),
        pl.lit(ints_row_val(a90, "avg_interval")).alias("A11_PDAY4263"),
    ])

    # Ratios: recent/mode, range_ratio, CV
    def recent_over_mode(A: pl.DataFrame) -> float:
        if A.height == 0: return OVR["NO_TXN"]
        n = int(A["n_txn"][0] or 0)
        if n == 0: return OVR["NO_TXN"]
        if n == 1: return OVR["ONE_TXN"]
        num = A["recent_interval"][0]
        den = A["mode_interval"][0]
        return ratio_interval_py(num, den)

    def range_ratio(A: pl.DataFrame) -> float:
        if A.height == 0: return OVR["NO_TXN"]
        n = int(A["n_txn"][0] or 0)
        if n == 0: return OVR["NO_TXN"]
        if n == 1: return OVR["ONE_TXN"]
        num = float(A["max_interval"][0] or 0.0) - float(A["min_interval"][0] or 0.0)
        den = float(A["avg_interval"][0] or 0.0)
        return ratio_interval_py(num, den)

    def cv_from_list(A: pl.DataFrame) -> float:
        if A.height == 0: return OVR["NO_TXN"]
        n = int(A["n_txn"][0] or 0)
        if n == 0: return OVR["NO_TXN"]
        if n == 1: return OVR["ONE_TXN"]
        cv, mean_iv, _ = _interval_cv_from_list(A["intervals"][0] if A.height else None)
        # ratio overrides for std/mean were handled in 70–75; here we follow the same outcome:
        if mean_iv == 0.0 and cv == 0.0: return OVR["BOTH0"]
        if mean_iv == 0.0: return OVR["DEN0"]
        return float(cv)

    out = out.with_columns([
        pl.lit(recent_over_mode(a30)).alias("A11_PDAY4234"),
        pl.lit(recent_over_mode(a90)).alias("A11_PDAY4233"),
        pl.lit(range_ratio(a30)).alias("A11_PDAY4274"),
        pl.lit(range_ratio(a90)).alias("A11_PDAY4273"),
        pl.lit(cv_from_list(a30)).alias("A11_PDAY4284"),
        pl.lit(cv_from_list(a90)).alias("A11_PDAY4283"),
    ])

    return out.with_columns([pl.col(c).cast(pl.Float64) for c in out.columns if c != KEY])

# ---------- 229–235: agnostic amount stats (Float) ----------
def features_agn_amounts_v2(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    override_value: float | None,
) -> pl.DataFrame:
    """
    Amounts (rank-agnostic):
      SUM:        A11_PDAY4504 (30d), A11_PDAY4503 (90d)
      MEAN:       A11_PDAY4524 (30d), A11_PDAY4523 (90d)
      LATEST/AVG: A11_PDAY4534 (30d), A11_PDAY4533 (90d)
      (MAX-MIN)/AVG: A11_PDAY4544 (30d), A11_PDAY4543 (90d)
      STD/MEAN:   A11_PDAY4554 (30d), A11_PDAY4553 (90d)
    """
    base = pl.DataFrame({KEY: pl.Series(KEY, [key_val], dtype=pl.Int64)})
    if override_value is not None:
        cols = [
            "A11_PDAY4504","A11_PDAY4503","A11_PDAY4524","A11_PDAY4523",
            "A11_PDAY4534","A11_PDAY4533","A11_PDAY4544","A11_PDAY4543",
            "A11_PDAY4554","A11_PDAY4553",
        ]
        return base.with_columns([pl.lit(override_value).alias(c) for c in cols])

    a30 = _agn_row(bases, "agn_0_30", key_val)
    a90 = _agn_row(bases, "agn_0_90", key_val)

    def val_or(row: pl.DataFrame, col: str) -> float:
        if row.height == 0 or int(row["n_txn"][0] or 0) == 0:
            return OVR["NO_TXN"]
        v = row[col][0]
        return float(v) if v is not None else OVR["NO_TXN"]

    def latest_over_avg(row: pl.DataFrame) -> float:
        if row.height == 0 or int(row["n_txn"][0] or 0) == 0:
            return OVR["NO_TXN"]
        num = float(row["latest_amt"][0] or 0.0)
        den = float(row["mean_amt"][0]   or 0.0)
        return ratio_amount_py(num, den)

    def range_over_avg(row: pl.DataFrame) -> float:
        if row.height == 0 or int(row["n_txn"][0] or 0) == 0:
            return OVR["NO_TXN"]
        maxv = float(row["max_amt"][0]  or 0.0)
        minv = float(row["min_amt"][0]  or 0.0)
        avgv = float(row["mean_amt"][0] or 0.0)
        num = maxv - minv
        den = avgv
        return ratio_amount_py(num, den)

    def std_over_mean(row: pl.DataFrame) -> float:
        if row.height == 0 or int(row["n_txn"][0] or 0) == 0:
            return OVR["NO_TXN"]
        num = float(row["std_amt"][0]  or 0.0)
        den = float(row["mean_amt"][0] or 0.0)
        return ratio_amount_py(num, den)

    out = base.with_columns([
        # SUM
        pl.lit(val_or(a30, "sum_amt")).alias("A11_PDAY4504"),
        pl.lit(val_or(a90, "sum_amt")).alias("A11_PDAY4503"),
        # MEAN
        pl.lit(val_or(a30, "mean_amt")).alias("A11_PDAY4524"),
        pl.lit(val_or(a90, "mean_amt")).alias("A11_PDAY4523"),
        # LATEST/AVG
        pl.lit(latest_over_avg(a30)).alias("A11_PDAY4534"),
        pl.lit(latest_over_avg(a90)).alias("A11_PDAY4533"),
        # (MAX-MIN)/AVG
        pl.lit(range_over_avg(a30)).alias("A11_PDAY4544"),
        pl.lit(range_over_avg(a90)).alias("A11_PDAY4543"),
        # STD/MEAN
        pl.lit(std_over_mean(a30)).alias("A11_PDAY4554"),
        pl.lit(std_over_mean(a90)).alias("A11_PDAY4553"),
    ])
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in out.columns if c != KEY])

# ---------- Wrapper for 199–235 ----------
def build_features_block_199_235(
    bases: dict[str, pl.DataFrame],
    key_val: int,
    ref_date: date,
    override_value: float | None,
) -> pl.DataFrame:
    b_dates   = features_agn_dates_v2(bases, key_val, override_value)
    b_ddiff   = features_agn_daydiffs_v2(bases, key_val, ref_date, override_value)
    b_ivals   = features_agn_intervals_v2(bases, key_val, override_value)
    b_amounts = features_agn_amounts_v2(bases, key_val, override_value)

    out = ensure_key_i64(b_dates)
    for b in (b_ddiff, b_ivals, b_amounts):
        out = out.join(ensure_key_i64(b), on=KEY, how="left")
    return out

# ---------- FINAL ASSEMBLER ----------
def compute_payday_attributes_for_customer_v2(
    df_eck: pl.DataFrame,
    hashing: HashingVectorizer | None = None,
    clustering: DBSCAN | None = None,
) -> pl.DataFrame:
    """
    Returns a single-row DataFrame: [KEY + 235 A11_* columns].
    """
    # refdate, windows, clustering+rank
    ref_date = refdate_from_customer(df_eck)
    df_txn_with_ranks = cluster_and_rank_top3(df_eck, hashing=hashing, clustering=clustering, ref_date=ref_date)
    windows = build_windows(ref_date)
    group, override_value = group_type_and_override(df_txn_with_ranks, ref_date)
    bases = build_bases_v2(df_txn_with_ranks, windows)
    key_val = int(df_txn_with_ranks[0, KEY])

    # ---- assemble rank-specific blocks
    f_1_21   = build_features_block_1_21(df_txn_with_ranks, bases, windows, override_value)
    f_22_45  = build_features_block_22_45(bases, key_val, override_value)
    f_46_75  = build_features_block_46_75(bases, key_val, override_value)
    f_76_99  = build_features_block_76_99(bases, key_val, override_value)
    f_100_126= build_features_block_100_126(bases, key_val, ref_date, override_value)
    f_127_174= build_features_block_127_174(bases, key_val, override_value)
    f_175_180= build_features_block_175_180(bases, key_val, override_value)
    f_181_198= build_features_block_181_198(bases, key_val, ref_date, override_value)

    # ---- rank-agnostic 199–235
    f_199_235= build_features_block_199_235(bases, key_val, ref_date, override_value)

    # ---- stitch
    blocks = [f_1_21, f_22_45, f_46_75, f_76_99, f_100_126, f_127_174, f_175_180, f_181_198, f_199_235]
    row = ensure_key_i64(blocks[0])
    for b in blocks[1:]:
        row = row.join(ensure_key_i64(b), on=KEY, how="left")

    # ---- final cast (string/date/float)
    row = cast_all_A11_features(row)
    return row

# Keep old entry-point name for your runner:
def compute_payday_attributes_for_customer(
    df_eck: pl.DataFrame,
    hashing: HashingVectorizer | None = None,
    clustering: DBSCAN | None = None,
) -> pl.DataFrame:
    return compute_payday_attributes_for_customer_v2(df_eck, hashing=hashing, clustering=clustering)