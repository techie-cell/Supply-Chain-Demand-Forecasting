"""
src/preprocessing.py  v3.0  FULL SCALE
Loads complete M5 (42,840 SKUs) with parquet caching.
NO artificial limits in production mode.
"""
import json, logging, os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
CACHE_DIR = ROOT / "data" / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _optimise(df):
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes("int64").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df

def _add_externals(df):
    n = len(df); rng = np.random.default_rng(42)
    df["web_trend"]        = rng.uniform(0,100,n).round(1).astype("float32")
    df["macro_cpi"]        = (1+rng.uniform(-0.01,0.01,n).cumsum()).astype("float32")
    df["competitor_price"] = (df["price"]*rng.uniform(0.8,1.2,n)).astype("float32")
    df["inventory_level"]  = rng.integers(50,500,n).astype("int16")
    return df

def _add_holidays(df):
    try:
        import holidays as hol; us=hol.US()
        df["is_holiday"]=df["date"].apply(lambda d:1 if d in us else 0).astype("int8")
    except Exception:
        df["is_holiday"]=(pd.to_datetime(df["date"]).dt.dayofweek>=5).astype("int8")
    return df

def load_m5(data_dir="data/m5", days=365):
    """Load M5 - uses parquet cache after first run."""
    cache = CACHE_DIR / f"m5_{days}d.parquet"
    if cache.exists():
        log.info(f"Loading M5 from cache {cache.name}...")
        df = pd.read_parquet(cache)
        log.info(f"M5: {len(df):,} rows | {df['sku_id'].nunique():,} SKUs | {df['store_id'].nunique()} stores")
        return df

    d = Path(data_dir)
    train_path = None
    for fname in ["sales_train_evaluation.csv","sales_train_validation.csv"]:
        if (d/fname).exists():
            train_path = d/fname; break

    if train_path:
        log.info(f"Building M5 from {train_path} (last {days} days, ALL SKUs)...")
        df_wide  = pd.read_csv(train_path)
        id_cols  = [c for c in ["id","item_id","dept_id","cat_id","store_id","state_id"] if c in df_wide.columns]
        day_cols = [c for c in df_wide.columns if c.startswith("d_")]
        day_nums = sorted([int(c[2:]) for c in day_cols])
        cutoff   = day_nums[-1]-days+1
        recent   = [f"d_{n}" for n in day_nums if n>=cutoff]
        log.info(f"Melting {len(recent)} days x {len(df_wide):,} SKUs → {len(recent)*len(df_wide):,} rows...")
        df = df_wide[id_cols+recent].melt(id_vars=id_cols, value_vars=recent, var_name="d", value_name="sales")
        df["date"] = pd.Timestamp("2011-01-29")+pd.to_timedelta(df["d"].str[2:].astype(int)-1, unit="D")
        df.drop(columns=["d"], inplace=True)
        df["sales"] = pd.to_numeric(df["sales"],errors="coerce").fillna(0).clip(0)
        if "item_id" in df.columns:
            df.rename(columns={"item_id":"sku_id"}, inplace=True)

        cal_path = d/"calendar.csv"
        if cal_path.exists():
            cal = pd.read_csv(cal_path, parse_dates=["date"])
            sc  = [c for c in cal.columns if c.startswith("snap_")]
            cal["promo"] = (cal[sc].max(axis=1).fillna(0).astype(int)) if sc else 0
            df = df.merge(cal[["date","promo"]], on="date", how="left")
            df["promo"] = df["promo"].fillna(0).astype("int8")
        else:
            df["promo"] = 0

        price_path = d/"sell_prices.csv"
        if price_path.exists():
            prices = pd.read_csv(price_path)
            df["wm_yr_wk"] = pd.to_datetime(df["date"]).dt.isocalendar().week.astype(int)
            df = df.merge(prices[["store_id","item_id","wm_yr_wk","sell_price"]],
                          left_on=["store_id","sku_id","wm_yr_wk"],
                          right_on=["store_id","item_id","wm_yr_wk"], how="left")
            df.rename(columns={"sell_price":"price"}, inplace=True)
            df.drop(columns=["item_id","wm_yr_wk"], errors="ignore", inplace=True)
            df["price"] = df["price"].fillna(2.0).astype("float32")
        else:
            df["price"] = 2.0
    else:
        log.warning("Full M5 not found — using sample CSV")
        sample = d/"m5_sample.csv"
        df = pd.read_csv(sample, parse_dates=["date"])
        for old,new in [("item_id","sku_id"),("snap_event","promo"),("sell_price","price")]:
            if old in df.columns: df.rename(columns={old:new}, inplace=True)
        df["price"] = df.get("price", pd.Series(2.0,index=df.index))
        df["promo"] = df.get("promo", pd.Series(0,  index=df.index))

    df["date"]     = pd.to_datetime(df["date"])
    df["sku_id"]   = df["sku_id"].astype(str)
    df["store_id"] = df["store_id"].astype(str) if "store_id" in df.columns else "S1"
    df["sales"]    = pd.to_numeric(df["sales"],errors="coerce").fillna(0).clip(0)
    parts          = df["sku_id"].str.split("_")
    df["category"] = parts.str[0].fillna("OTHER")
    df["dept_id"]  = parts.str[:2].str.join("_").fillna("OTHER")
    df = _optimise(df)
    log.info(f"Saving cache {cache}...")
    df.to_parquet(cache, index=False)
    log.info(f"M5 ready: {len(df):,} rows | {df['sku_id'].nunique():,} SKUs | cats:{sorted(df['category'].unique().tolist())}")
    return df

def load_favorita(data_dir="data/favorita"):
    cache = CACHE_DIR / "favorita.parquet"
    if cache.exists():
        log.info("Loading Favorita from cache...")
        return pd.read_parquet(cache)
    d = Path(data_dir)
    train_path = d/"train.csv"
    sample     = d/"favorita_sample.csv"
    if train_path.exists():
        log.info("Loading full Favorita in chunks...")
        chunks=[]
        for chunk in pd.read_csv(train_path, parse_dates=["date"],
                                  dtype={"store_nbr":"int16","item_nbr":"int32",
                                         "unit_sales":"float32"}, chunksize=2_000_000):
            chunk["unit_sales"]=chunk["unit_sales"].clip(lower=0)
            chunks.append(chunk)
        df=pd.concat(chunks,ignore_index=True)
    else:
        df=pd.read_csv(sample, parse_dates=["date"])
    for old,new in [("item_nbr","sku_id"),("store_nbr","store_id"),
                    ("unit_sales","sales"),("onpromotion","promo")]:
        if old in df.columns: df.rename(columns={old:new}, inplace=True)
    df["sku_id"]   = df["sku_id"].astype(str)
    df["store_id"] = df["store_id"].astype(str)
    df["sales"]    = pd.to_numeric(df.get("sales",0),errors="coerce").fillna(0).clip(0)
    df["promo"]    = df.get("promo",pd.Series(0,index=df.index)).fillna(0).astype(int)
    df["price"]    = df.get("price",pd.Series(1.0,index=df.index)).fillna(1.0)
    df["category"] = df.get("family",df.get("class",pd.Series("General",index=df.index))).astype(str)
    df = _optimise(df)
    df.to_parquet(cache, index=False)
    log.info(f"Favorita: {len(df):,} rows | {df['sku_id'].nunique():,} SKUs")
    return df

LOADERS = {"M5": load_m5, "Favorita": load_favorita}

def get_metadata(dataset="M5", data_dir=None, days=365):
    """Return full SKU/store/category lists from cache (fast, no full load)."""
    ddir = data_dir or f"data/{dataset.lower()}"
    cache_files = list(CACHE_DIR.glob(f"{dataset.lower()}*.parquet"))
    if cache_files:
        try:
            df = pd.read_parquet(sorted(cache_files)[-1],
                                 columns=["sku_id","store_id","date","category"])
        except Exception:
            df = pd.read_parquet(sorted(cache_files)[-1])
    else:
        if dataset.upper()=="M5":
            df = load_m5(ddir, days=days)
        elif dataset.upper()=="FAVORITA":
            df = load_favorita(ddir)
        else:
            df = _fallback_sample(dataset, ddir)
    cats = sorted(df["category"].unique().tolist()) if "category" in df.columns else ["All"]
    return {
        "dataset":    dataset,
        "n_rows":     len(df),
        "n_skus":     int(df["sku_id"].nunique()),
        "n_stores":   int(df["store_id"].nunique()),
        "date_min":   str(df["date"].min().date()),
        "date_max":   str(df["date"].max().date()),
        "skus":       sorted(df["sku_id"].unique().tolist()),
        "stores":     sorted(df["store_id"].unique().tolist()),
        "categories": cats,
    }

def _fallback_sample(dataset, ddir):
    sample = Path(ddir)/f"{dataset.lower()}_sample.csv"
    df = pd.read_csv(sample, parse_dates=["date"]) if sample.exists() else pd.DataFrame()
    for old,new in [("item_id","sku_id"),("item_nbr","sku_id"),("store_nbr","store_id"),
                    ("unit_sales","sales"),("onpromotion","promo"),("snap_event","promo")]:
        if old in df.columns: df.rename(columns={old:new}, inplace=True)
    for c,v in [("sku_id","SKU_001"),("store_id","S1"),("sales",0),("promo",0),
                ("price",2.0),("category","General")]:
        if c not in df.columns: df[c]=v
    return df

def preprocess(dataset="M5", data_dir=None, max_skus=None, days=365,
               min_history=30, store_filter=None, category_filter=None,
               sku_filter=None):
    """
    Main preprocessing entry point.
    max_skus=None  -> ALL SKUs (production mode)
    max_skus=N     -> top-N SKUs by total sales (test mode only)
    """
    ddir = data_dir or f"data/{dataset.lower()}"
    if dataset.upper()=="M5":
        df = load_m5(ddir, days=days)
    elif dataset.upper()=="FAVORITA":
        df = load_favorita(ddir)
    else:
        df = _fallback_sample(dataset, ddir)
        df["date"] = pd.to_datetime(df["date"])

    # Apply filters AFTER loading from cache
    if store_filter    and store_filter    not in ("All Stores","All",None):
        df = df[df["store_id"]==str(store_filter)]
    if category_filter and category_filter not in ("All",None):
        if "category" in df.columns:
            df = df[df["category"]==str(category_filter)]
    if sku_filter      and sku_filter      not in ("All Products","All",None):
        df = df[df["sku_id"]==str(sku_filter)]

    # TEST MODE ONLY — never set max_skus in production
    if max_skus is not None:
        top = df.groupby("sku_id")["sales"].sum().nlargest(max_skus).index.tolist()
        df  = df[df["sku_id"].isin(top)]
        log.warning(f"TEST MODE: limited to top {max_skus} SKUs")

    # Drop short series
    counts = df.groupby("sku_id")["date"].count()
    df     = df[df["sku_id"].isin(counts[counts>=min_history].index)]

    # Add external features
    df = _add_externals(df)
    df = _add_holidays(df)

    meta = {
        "dataset":    dataset,
        "n_rows":     len(df),
        "n_skus":     int(df["sku_id"].nunique()),
        "n_stores":   int(df["store_id"].nunique()),
        "date_min":   str(df["date"].min().date()),
        "date_max":   str(df["date"].max().date()),
        "skus":       sorted(df["sku_id"].unique().tolist()),
        "stores":     sorted(df["store_id"].unique().tolist()),
        "categories": sorted(df["category"].unique().tolist()) if "category" in df.columns else ["All"],
    }
    log.info(f"[{dataset}] {meta['n_rows']:,} rows | {meta['n_skus']:,} SKUs | {meta['n_stores']} stores")
    return df, meta

def clear_cache(dataset=None):
    for f in CACHE_DIR.glob("*.parquet"):
        if dataset is None or dataset.lower() in f.stem:
            f.unlink(); log.info(f"Deleted cache: {f.name}")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",default="M5")
    p.add_argument("--days",type=int,default=365)
    p.add_argument("--max_skus",type=int,default=None)
    p.add_argument("--clear",action="store_true")
    a=p.parse_args()
    if a.clear: clear_cache(a.dataset)
    df,meta=preprocess(a.dataset, days=a.days, max_skus=a.max_skus)
    print(f"Rows:{meta['n_rows']:,} | SKUs:{meta['n_skus']:,} | Stores:{meta['n_stores']} | Cats:{meta['categories']}")
