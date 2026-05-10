"""
run.py
Master entrypoint — run any part of the project from one place.
Usage:
  python run.py setup
  python run.py train --dataset M5 --model arima
  python run.py api
  python run.py dashboard
  python run.py retrain --dataset M5
  python run.py demo
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ── colour helpers ────────────────────────────────────────────────────────────
G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; C = "\033[96m"; B = "\033[1m"; E = "\033[0m"
def ok(m):  print(f"{G}✓ {m}{E}")
def info(m): print(f"{C}ℹ {m}{E}")
def warn(m): print(f"{Y}⚠ {m}{E}")
def err(m):  print(f"{R}✗ {m}{E}")
def hdr(m):  print(f"\n{B}{C}{'='*60}\n  {m}\n{'='*60}{E}")


def run_cmd(cmd: list, env_extra: dict = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent)
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(cmd, env=env)
    return result.returncode


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_setup(args):
    hdr("Project Setup")
    import create_project
    create_project.main()


def cmd_train(args):
    hdr(f"Training {args.model.upper()} on {args.dataset}")
    scripts = {
        "arima":   "src/train_arima.py",
        "xgboost": "src/train_xgboost.py",
        "lstm":    "src/train_lstm.py",
        "tft":     "src/train_tft.py",
    }
    if args.model not in scripts:
        err(f"Unknown model: {args.model}. Choose: {list(scripts.keys())}")
        return 1

    cmd = [sys.executable, scripts[args.model],
           "--dataset", args.dataset,
           "--max_skus", str(args.max_skus)]
    if args.model in ("arima",):
        cmd += ["--horizon", str(args.horizon)]
    if args.model == "lstm":
        cmd += ["--epochs", str(args.epochs)]

    return run_cmd(cmd)


def cmd_train_all(args):
    hdr(f"Training ALL models on {args.dataset}")
    models = ["arima", "xgboost", "lstm"]
    for m in models:
        info(f"\n→ Training {m.upper()} …")
        args.model = m
        rc = cmd_train(args)
        if rc != 0:
            warn(f"{m} training failed (continuing …)")
    ok("All models trained!")


def cmd_api(args):
    hdr("Starting FastAPI Backend")
    info(f"  URL: http://localhost:{args.port}")
    info("  Docs: http://localhost:{}/docs".format(args.port))
    info("  Press Ctrl+C to stop\n")
    return run_cmd([sys.executable, "-m", "uvicorn", "api.app:app",
                    "--reload", "--host", "0.0.0.0", "--port", str(args.port)])


def cmd_dashboard(args):
    hdr("Starting Streamlit Dashboard")
    info(f"  URL: http://localhost:{args.port}")
    info("  Press Ctrl+C to stop\n")
    info("  ⚠  Start the API first: python run.py api\n")
    return run_cmd([sys.executable, "-m", "streamlit", "run",
                    "dashboard/app_streamlit.py",
                    "--server.port", str(args.port),
                    "--server.address", "localhost"])


def cmd_retrain(args):
    hdr(f"MLOps Retraining Pipeline — {args.dataset}")
    sys.path.insert(0, str(Path(__file__).parent))
    from src.retrain import run_pipeline
    result = run_pipeline(
        dataset       = args.dataset,
        check_drift   = not args.no_drift,
        force_retrain = args.force,
        models        = args.models,
    )
    print(json.dumps({k: v for k, v in result.items()
                      if not isinstance(v, dict)}, indent=2))


def cmd_evaluate(args):
    hdr(f"Evaluation Report — {args.dataset}")
    import glob
    reports = glob.glob(f"reports/{args.dataset}_*.json")
    if not reports:
        warn("No evaluation reports found. Train models first.")
        return

    rows = []
    for rp in sorted(reports):
        with open(rp) as f:
            d = json.load(f)
        rows.append({
            "Model":   d.get("model", "?"),
            "WMAPE":   d.get("avg_WMAPE", d.get("WMAPE", "?")),
            "MAE":     d.get("MAE", "?"),
            "n_SKUs":  d.get("n_skus", "?"),
        })

    print(f"\n{'Model':<15} {'WMAPE%':<12} {'MAE':<12} {'SKUs':<8}")
    print("─" * 50)
    for r in rows:
        print(f"{r['Model']:<15} {str(r['WMAPE']):<12} {str(r['MAE']):<12} {str(r['n_SKUs']):<8}")
    print()


def cmd_demo(args):
    hdr("Full Demo: Setup → Train → Forecast")
    info("Step 1/4: Setup project …")
    cmd_setup(args)

    info("\nStep 2/4: Train ARIMA on M5 (fast) …")
    a = argparse.Namespace(dataset="M5", model="arima", max_skus=3,
                            horizon=14, epochs=10)
    cmd_train(a)

    info("\nStep 3/4: Train XGBoost on M5 …")
    a.model = "xgboost"; a.max_skus = 5
    cmd_train(a)

    info("\nStep 4/4: Quick forecast via Python API …")
    sys.path.insert(0, ".")
    try:
        from src.preprocessing import preprocess
        df, meta = preprocess("M5", max_skus=3)
        sku = df["sku_id"].iloc[0]
        series = df[df["sku_id"] == sku].sort_values("date")["sales"]
        from src.evaluation import moving_average_forecast, evaluate
        fc = moving_average_forecast(series, horizon=7)
        m  = evaluate(series.tail(7).values, fc, model_name="MA_Demo")
        print(f"\n  SKU: {sku}")
        print(f"  7-day forecast: {[round(v,1) for v in fc]}")
        print(f"  WMAPE: {m['WMAPE']}%")
    except Exception as e:
        warn(f"Demo forecast failed: {e}")

    print(f"\n{B}{G}✓ Demo complete!{E}")
    print(f"\n{C}Next steps:")
    print(f"  Terminal 1: {B}python run.py api{E}")
    print(f"  Terminal 2: {B}python run.py dashboard{E}")


def cmd_sota(args):
    hdr(f"Running SOTA Module: {args.module}")
    sys.path.insert(0, ".")
    from src.preprocessing import preprocess
    df, _ = preprocess("M5", max_skus=5)

    if args.module == "anomaly":
        from sota.anomaly_detection import AnomalyDetectionPipeline, detect_supply_disruptions
        pipe = AnomalyDetectionPipeline()
        result_df = pipe.run(df)
        report = pipe.get_anomaly_report(result_df)
        print(json.dumps(pipe.results, indent=2))
        disruptions = detect_supply_disruptions(df)
        print(f"\nSupply disruptions detected in {len(disruptions)} SKUs")

    elif args.module == "gnn":
        from sota.gnn_model import GNNDemandPipeline
        pipe = GNNDemandPipeline()
        pipe.build_graph(df)
        pipe.build_model()
        result = pipe.train(df, epochs=5)
        print(json.dumps(result, indent=2))

    elif args.module == "llm":
        from sota.llm_integration import SentimentSignalExtractor, ForecastCommentator
        ext   = SentimentSignalExtractor()
        ext.load_model()
        sents = ext.analyze(ext.SAMPLE_NEWS)
        sig   = ext.to_demand_signal(sents)
        print(f"\nDemand signal from news: {sig:.3f}")
        comm = ForecastCommentator()
        text = comm.generate_commentary({"sku": "FOODS_1_001", "horizon": 14,
                                          "forecast": [30]*14, "trend": "stable",
                                          "model": "TFT"})
        print(text)

    elif args.module == "multitask":
        from sota.multitask_tft import MultiTaskTFTPipeline
        pipe = MultiTaskTFTPipeline()
        result = pipe.run(df, max_epochs=3)
        print(json.dumps({k: v for k, v in result.items()
                           if isinstance(v, (str, int, float, bool))}, indent=2))
    else:
        err(f"Unknown module: {args.module}. Choose: anomaly | gnn | llm | multitask")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="AI Supply Chain Demand Forecasting — Master Entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py setup
  python run.py train --dataset M5 --model arima
  python run.py train --dataset Favorita --model xgboost --max_skus 10
  python run.py train-all --dataset UCI
  python run.py api --port 8000
  python run.py dashboard --port 8501
  python run.py retrain --dataset M5 --models arima xgboost --force
  python run.py evaluate --dataset M5
  python run.py sota --module anomaly
  python run.py demo
""")

    sub = parser.add_subparsers(dest="command")

    # setup
    sub.add_parser("setup", help="Create directories and sample data")

    # train
    tr = sub.add_parser("train", help="Train a single model")
    tr.add_argument("--dataset",   default="M5",      choices=["M5","Favorita","UCI"])
    tr.add_argument("--model",     default="arima",   choices=["arima","xgboost","lstm","tft"])
    tr.add_argument("--max_skus",  type=int, default=5)
    tr.add_argument("--horizon",   type=int, default=14)
    tr.add_argument("--epochs",    type=int, default=10)

    # train-all
    ta = sub.add_parser("train-all", help="Train all models for a dataset")
    ta.add_argument("--dataset",  default="M5", choices=["M5","Favorita","UCI"])
    ta.add_argument("--max_skus", type=int, default=5)
    ta.add_argument("--horizon",  type=int, default=14)
    ta.add_argument("--epochs",   type=int, default=10)

    # api
    ap = sub.add_parser("api", help="Start FastAPI backend")
    ap.add_argument("--port", type=int, default=8000)

    # dashboard
    db = sub.add_parser("dashboard", help="Start Streamlit dashboard")
    db.add_argument("--port", type=int, default=8501)

    # retrain
    rt = sub.add_parser("retrain", help="MLOps retraining pipeline")
    rt.add_argument("--dataset",   default="M5",  choices=["M5","Favorita","UCI"])
    rt.add_argument("--models",    nargs="+", default=["arima","xgboost"],
                    choices=["arima","xgboost","lstm","tft"])
    rt.add_argument("--force",     action="store_true")
    rt.add_argument("--no-drift",  action="store_true")

    # evaluate
    ev = sub.add_parser("evaluate", help="Print evaluation report")
    ev.add_argument("--dataset", default="M5", choices=["M5","Favorita","UCI"])

    # sota
    so = sub.add_parser("sota", help="Run a SOTA module")
    so.add_argument("--module", default="anomaly",
                    choices=["anomaly","gnn","llm","multitask"])

    # demo
    sub.add_parser("demo", help="Full end-to-end demo")

    args = parser.parse_args()

    dispatch = {
        "setup":     cmd_setup,
        "train":     cmd_train,
        "train-all": cmd_train_all,
        "api":       cmd_api,
        "dashboard": cmd_dashboard,
        "retrain":   cmd_retrain,
        "evaluate":  cmd_evaluate,
        "sota":      cmd_sota,
        "demo":      cmd_demo,
    }

    if not args.command:
        parser.print_help()
        return

    fn = dispatch.get(args.command)
    if fn:
        sys.exit(fn(args) or 0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
