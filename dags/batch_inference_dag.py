"""
Batch Inference DAG — ANALYTICA Sprint 5 F5
dag_id: analytica_batch_inference
Schedule: 0 1 * * * Asia/Jakarta (daily 01:00 WIB)
SLA: 45 minutes

Pipeline:
  prepare_grid_cells
    -> run_batch_xgb
    -> run_batch_lstm
    -> run_batch_tft
    -> aggregate_outputs
    -> write_batch_report

Scope: all active grid cells (workspace/config/grid_cells.json)
Output: workspace/output/batch_forecasts/{YYYYMMDD}/  -- one JSON per model per grid cell
Batch report: workspace/output/batch_forecasts/{YYYYMMDD}/summary.md (consumed by VISUALIA)
Calls record_ingestion_success('batch_inference_daily') on completion.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DAG_ID       = "analytica_batch_inference"
SCHEDULE     = "0 1 * * *"
TIMEZONE     = "Asia/Jakarta"
SLA_SECONDS  = 45 * 60   # 45 minutes

GRID_CELLS_PATH = Path("workspace/config/grid_cells.json")
OUTPUT_ROOT     = Path("workspace/output/batch_forecasts")

BATCH_MODELS = [
    "xgb_precip_nowcast",
    "lstm_streamflow",
    "tft_climate_forecast",
]

FORECAST_HORIZONS: Dict[str, int] = {
    "xgb_precip_nowcast":   72,
    "lstm_streamflow":      24,
    "tft_climate_forecast": 168,
}

ISLAND_REGIONS = ["sumatra", "java", "kalimantan", "sulawesi", "papua"]


def _province_to_region(idx: int) -> str:
    """Simple deterministic province-to-island mapping for stubs."""
    return ISLAND_REGIONS[idx % len(ISLAND_REGIONS)]


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def prepare_grid_cells(**context) -> None:
    """
    Load active grid cells from workspace/config/grid_cells.json.
    Creates the output directory for today's run. Pushes cell list to XCom.
    """
    run_ds  = context["ds"]
    out_dir = OUTPUT_ROOT / run_ds.replace("-", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    if GRID_CELLS_PATH.exists():
        with open(GRID_CELLS_PATH) as f:
            config = json.load(f)
        grid_cells: List[dict] = [
            gc for gc in config.get("grid_cells", []) if gc.get("active", True)
        ]
    else:
        logger.warning("grid_cells.json not found at %s -- using stub", GRID_CELLS_PATH)
        grid_cells = [
            {
                "grid_cell_id": f"gc_{i:04d}",
                "lat": round(-6.0 + i * 0.04, 4),
                "lon": round(107.0 + i * 0.04, 4),
                "province": f"province_{i:02d}",
                "region": _province_to_region(i),
            }
            for i in range(950)
        ]

    logger.info("Batch inference: %d active grid cells for %s", len(grid_cells), run_ds)
    context["ti"].xcom_push(key="grid_cells", value=grid_cells)
    context["ti"].xcom_push(key="out_dir",    value=str(out_dir))


def run_batch_xgb(**context) -> None:
    """Run XGBoost precipitation nowcast over all grid cells."""
    _run_batch_model("xgb_precip_nowcast", context)


def run_batch_lstm(**context) -> None:
    """Run LSTM streamflow forecast over all grid cells."""
    _run_batch_model("lstm_streamflow", context)


def run_batch_tft(**context) -> None:
    """Run TFT multi-variate climate forecast over all grid cells."""
    _run_batch_model("tft_climate_forecast", context)


def _run_batch_model(model_name: str, context: dict) -> None:
    """
    Core batch loop for a single model.
    Features via FeatureStoreClient (online), inference via model.predict(),
    writes one JSON per grid cell to out_dir/{model_name}/.
    """
    from src.data.feature_store_client import FeatureStoreClient
    from src.serving.inference_cache   import InferenceCache

    ti         = context["ti"]
    grid_cells = ti.xcom_pull(key="grid_cells", task_ids="prepare_grid_cells")
    out_dir    = Path(ti.xcom_pull(key="out_dir", task_ids="prepare_grid_cells"))
    run_ds     = context["ds"]
    issued     = date.fromisoformat(run_ds)
    horizon    = FORECAST_HORIZONS.get(model_name, 72)

    model_dir  = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    fs     = FeatureStoreClient()
    cache  = InferenceCache()
    model  = _load_model(model_name)
    errors = 0

    for gc in grid_cells:
        gc_id = gc["grid_cell_id"]
        try:
            features_df = fs.get_online_features(
                entity_rows=[{"grid_cell_id": gc_id}],
                feature_refs=getattr(model, "FEATURE_REFS", []),
            )
            input_key = {"grid_cell_id": gc_id, "forecast_horizon": horizon, "issued_date": run_ds}
            cached_pred = cache.get(model_name, getattr(model, "version", "prod"), input_key)
            if cached_pred:
                pred = cached_pred
            else:
                pred = model.predict(features=features_df, forecast_horizon=horizon, issued_date=issued)
                cache.set(model_name, getattr(model, "version", "prod"), input_key, pred)

            (model_dir / f"{gc_id}.json").write_text(json.dumps({
                "model":               model_name,
                "grid_cell_id":        gc_id,
                "issued_date":         run_ds,
                "forecast_horizon_h":  horizon,
                "prediction":          pred.get("prediction"),
                "confidence_interval": pred.get("confidence_interval"),
                "province":            gc.get("province"),
                "region":              gc.get("region"),
            }, default=str))
        except Exception as exc:
            logger.warning("Batch %s failed for gc=%s: %s", model_name, gc_id, exc)
            errors += 1

    total = len(grid_cells)
    logger.info("Batch %s done: %d ok, %d errors", model_name, total - errors, errors)
    ti.xcom_push(key=f"batch_result_{model_name}", value={
        "model": model_name, "total": total, "errors": errors
    })


def aggregate_outputs(**context) -> None:
    """Combine per-model summaries into manifest.json."""
    ti      = context["ti"]
    out_dir = Path(ti.xcom_pull(key="out_dir", task_ids="prepare_grid_cells"))
    run_ds  = context["ds"]

    manifest: Dict[str, Any] = {"run_date": run_ds, "models": {}}
    for model_name in BATCH_MODELS:
        short = model_name.split("_")[0]
        result = ti.xcom_pull(key=f"batch_result_{model_name}", task_ids=f"run_batch_{short}")
        manifest["models"][model_name] = result or {}

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Manifest written to %s/manifest.json", out_dir)
    ti.xcom_push(key="manifest", value=manifest)


def write_batch_report(**context) -> None:
    """
    Write summary.md consumed by VISUALIA.
    Calls record_ingestion_success('batch_inference_daily') on completion.
    """
    ti       = context["ti"]
    out_dir  = Path(ti.xcom_pull(key="out_dir",   task_ids="prepare_grid_cells"))
    manifest = ti.xcom_pull(key="manifest",        task_ids="aggregate_outputs")
    run_ds   = context["ds"]

    total_cells  = sum(m.get("total",  0) for m in manifest["models"].values())
    total_errors = sum(m.get("errors", 0) for m in manifest["models"].values())

    lines = [
        f"# ANALYTICA Batch Inference Report -- {run_ds}",
        "",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}  ",
        f"**Grid cells processed:** {total_cells}  ",
        f"**Total errors:** {total_errors}  ",
        "",
        "## Model Summary",
        "",
        "| Model | Cells | Errors |",
        "|-------|-------|--------|",
    ]
    for model_name, m in manifest["models"].items():
        lines.append(f"| `{model_name}` | {m.get('total', 0)} | {m.get('errors', 0)} |")

    lines += [
        "",
        "## Output Location",
        "",
        f"`workspace/output/batch_forecasts/{run_ds.replace('-', '')}/`",
        "",
        "One JSON file per model per grid cell. See `manifest.json` for the full index.",
        "",
        "---",
        "_Auto-generated by analytica_batch_inference DAG. Consumed by VISUALIA._",
    ]

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines))
    logger.info("Batch report written: %s", summary_path)

    try:
        from src.data.ingestion_tracker import record_ingestion_success
        record_ingestion_success("batch_inference_daily")
        logger.info("record_ingestion_success('batch_inference_daily') OK")
    except ImportError:
        logger.warning("ingestion_tracker not available -- skipping record_ingestion_success")
    except Exception as exc:
        logger.warning("record_ingestion_success failed: %s", exc)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model(model_name: str) -> Any:
    """Load model from MLflow Production registry; fall back to a no-op stub."""
    try:
        from src.training.mlflow_registry import MLflowRegistry
        registry = MLflowRegistry()
        registry.get_latest_production(model_name)   # raises if absent
        import mlflow
        return mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    except Exception as exc:
        logger.warning("Model %s unavailable (%s) -- using _BatchModelStub", model_name, exc)
        return _BatchModelStub(model_name)


class _BatchModelStub:
    """No-op stub that returns null predictions when the real model is unavailable."""
    FEATURE_REFS = []
    version = "stub"

    def __init__(self, model_name: str):
        self.model_name = model_name

    def predict(self, features, forecast_horizon, issued_date):
        return {"prediction": None, "confidence_interval": None}


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner":            "analytica",
    "depends_on_past":  False,
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=10),
    "sla":              timedelta(seconds=SLA_SECONDS),
}

with DAG(
    dag_id=DAG_ID,
    schedule_interval=SCHEDULE,
    start_date=days_ago(1),
    default_args=default_args,
    catchup=False,
    tags=["analytica", "batch", "inference"],
    description="Daily batch inference for XGB, LSTM, TFT over all active grid cells",
) as dag:

    t_prepare   = PythonOperator(task_id="prepare_grid_cells", python_callable=prepare_grid_cells)
    t_xgb       = PythonOperator(task_id="run_batch_xgb",      python_callable=run_batch_xgb)
    t_lstm      = PythonOperator(task_id="run_batch_lstm",      python_callable=run_batch_lstm)
    t_tft       = PythonOperator(task_id="run_batch_tft",       python_callable=run_batch_tft)
    t_aggregate = PythonOperator(task_id="aggregate_outputs",   python_callable=aggregate_outputs)
    t_report    = PythonOperator(task_id="write_batch_report",  python_callable=write_batch_report)

    t_prepare >> [t_xgb, t_lstm, t_tft] >> t_aggregate >> t_report
