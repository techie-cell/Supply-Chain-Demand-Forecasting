"""
download_m5.py
Downloads M5 dataset from Kaggle and builds memory-efficient cache.
Run: python download_m5.py --days 365
RAM guide: 8GB→365days, 6GB→180days, 4GB→90days
"""
import os, sys, subprocess, zipfile
from pathlib import Path

G="\033[92m";Y="\033[93m";R="\033[91m";C="\033[96m";B="\033[1m";E="\033[0m"
def ok(m):   print(f"{G}✓ {m}{E}")
def info(m): print(f"{C}  {m}{E}")
def err(m):  print(f"{R}✗ {m}{E}")
def hdr(m):  print(f"\n{B}{'='*55}\n  {m}\n{'='*55}{E}")

def check_kaggle():
    try: import kaggle; ok("kaggle installed")
    except ImportError:
        subprocess.run([sys.executable,"-m","pip","install","kaggle","-q"])
    cred=Path.home()/".kaggle"/"kaggle.json"
    if not cred.exists():
        err(f"Kaggle token not found: {cred}")
        print(f"""
{Y}Steps:{E}
  1. Go to https://www.kaggle.com/settings
  2. Click "Create New Token" → downloads kaggle.json
  3. Place at: C:\\Users\\{os.getenv('USERNAME','user')}\\.kaggle\\kaggle.json
  4. Run this script again
  
{Y}Manual download (alternative):{E}
  1. https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
  2. Download: sales_train_evaluation.csv, calendar.csv, sell_prices.csv
  3. Place in: data\\m5\\""")
        return False
    ok(f"Kaggle token found")
    return True

def download():
    dest=Path("data/m5"); dest.mkdir(parents=True,exist_ok=True)
    needed=["sales_train_evaluation.csv","calendar.csv","sell_prices.csv"]
    if all((dest/f).exists() for f in needed):
        ok("All M5 files already present"); return True
    info("Downloading M5 from Kaggle (may take 5-10 mins)...")
    
    # FIXED LINE
    r=subprocess.run(["kaggle","competitions","download",
                      "-c","m5-forecasting-accuracy","-p",str(dest)])
    
    if r.returncode!=0:
        err("Download failed. Use manual download instructions above.")
        return False
    for zf in dest.glob("*.zip"):
        info(f"Extracting {zf.name}...")
        with zipfile.ZipFile(zf,"r") as z: z.extractall(dest)
        zf.unlink()
    ok("Downloaded and extracted!")
    for f in needed:
        p=dest/f
        if p.exists(): ok(f"  {f} ({p.stat().st_size/1e6:.0f} MB)")
    return True

def build_cache(days=365):
    hdr(f"Building cache — ALL SKUs, last {days} days")
    from src.preprocessing import load_m5, CACHE_DIR
    cache=CACHE_DIR/f"m5_{days}d.parquet"
    if cache.exists():
        ok(f"Cache already exists: {cache}"); 
        import pandas as pd
        df=pd.read_parquet(cache)
        ok(f"SKUs:{df['sku_id'].nunique():,} | Stores:{df['store_id'].nunique()} | Rows:{len(df):,}")
        return
    df=load_m5("data/m5", days=days)
    ok(f"Cache built: {cache} ({cache.stat().st_size/1e6:.0f} MB)")
    ok(f"SKUs:{df['sku_id'].nunique():,} | Stores:{df['store_id'].nunique()} | Rows:{len(df):,}")
    ok(f"Categories: {sorted(df['category'].unique().tolist())}")

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--days",type=int,default=365,
                   help="Days per SKU (365=8GB, 180=6GB, 90=4GB)")
    p.add_argument("--skip-download",action="store_true")
    a=p.parse_args()

    hdr(f"M5 Walmart Dataset Setup (--days {a.days})")
    print(f"\nRAM guide: 8GB→365 | 6GB→180 | 4GB→90\n")

    if not a.skip_download:
        if not check_kaggle(): sys.exit(1)
        if not download():     sys.exit(1)

    build_cache(days=a.days)

    hdr("DONE!")
    print(f"""{G}
M5 dataset ready with ALL 42,840 SKUs!

Next steps:
  .\\START_ALL.bat           <- Launch API + Dashboard

Dashboard will show:
  - All 42,840 SKUs
  - 3 categories: FOODS, HOBBIES, HOUSEHOLD
  - 10 stores: CA_1..CA_4, TX_1..TX_3, WI_1..WI_3
{E}""")

if __name__=="__main__":
    main()