"""
Batch Inference DAG — ANALYTICA Sprint 6 H5
dag_id: analytica_batch_inference
Schedule: 0 1 * * * Asia/Jakarta (daily 01:00 WIB)
SLA: 45 minutes

Pipeline:
  prepare_grid_cells
    -> run_batch_xgb  ─┐
    -> run_batch_lstm  ├─ parallel
    -> run_batch_tft  ─┘
    -> aggregate_outputs
    -> write_batch_report

Grid cell source: workspace/config/grid_cells.json (20 DAS Strategis Nasional stub)
Per-model output:  workspace/output/batch_forecasts/{YYYYMMDD}/{model_id}_{grid_cell_id}.json
Summary report:    workspace/output/batch_forecasts/{YYYYMMDD}/summary.md  (consumed by VISUALIA)
Completion signal: record_ingestion_success('batch_inference_daily')
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DAG_ID      = "analytica_batch_inference"
SCHEDULE    = "0 1 * * *"
TIMEZONE    = "Asia/Jakarta"
SLA_SECONDS = 45 * 60

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

# 20 DAS Strategis Nasional fallback — mirrors workspace/config/grid_cells.json
_DAS_STUB: List[dict] = [
    {"grid_cell_id": "das_ciliwung",      "region": "java",         "province": "DKI Jakarta / Jawa Barat"},
    {"grid_cell_id": "das_citarum",       "region": "java",         "province": "Jawa Barat"},
    {"grid_cell_id": "das_brantas",       "region": "java",         "province": "Jawa Timur"},
    {"grid_cell_id": "das_bengawan_solo", "region": "java",         "province": "Jawa Tengah / Jawa Timur"},
    {"grid_cell_id": "das_serayu",        "region": "java",         "province": "Jawa Tengah"},
    {"grid_cell_id": "das_musi",          "region": "sumatra",      "province": "Sumatera Selatan"},
    {"grid_cell_id": "das_batanghari",    "region": "sumatra",      "province": "Jambi / Sumatera Barat"},
    {"grid_cell_id": "das_kampar",        "region": "sumatra",      "province": "Riau"},
    {"grid_cell_id": "das_rokan",         "region": "sumatra",      "province": "Riau / Sumatera Utara"},
    {"grid_cell_id": "das_asahan",        "region": "sumatra",      "province": "Sumatera Utara"},
    {"grid_cell_id": "das_kapuas",        "region": "kalimantan",   "province": "Kalimantan Barat"},
    {"grid_cell_id": "das_mahakam",       "region": "kalimantan",   "province": "Kalimantan Timur"},
    {"grid_cell_id": "das_barito",        "region": "kalimantan",   "province": "Kalimantan Tengah/Selatan"},
    {"grid_cell_id": "das_kahayan",       "region": "kalimantan",   "province": "Kalimantan Tengah"},
    {"grid_cell_id": "das_tondano",       "region": "sulawesi",     "province": "Sulawesi Utara"},
    {"grid_cell_id": "das_saddang",       "region": "sulawesi",     "province": "Sulawesi Selatan"},
    {"grid_cell_id": "das_lariang",       "region": "sulawesi",     "province": "Sulawesi Tengah"},
    {"grid_cell_id": "das_memberamo",     "region": "maluku_papua", "province": "Papua"},
    {"grid_cell_id": "das_digul",         "region": "maluku_papua", "province": "Papua Selatan"},
    {"grid_cell_id": "das_baliem",        "region": "maluku_papua", "province": "Papua Pegunungan"},
]


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def prepare_grid_cells(**context) -> None:
    """
    Load active grid cells from workspace/config/grid_cells.json.
    Falls back to the 20 DAS Strategis Nasional in-memory stub if file absent.
    Creates output directory for today's run. Pushes cell list to XCom.
    """
    run_ds  = context["ds"]
    out_dir = OUTPUT_ROOT / run_ds.replace("-", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    if GRID_CELLS_PATH.exists():
        with open(GRID_CELLS_PATH) as f:
            config = json.load(f)
        grid_cells: List[dict] = [gc for gc in config.get("grid_cells", []) if gc.get("active", True)]
        logger.info("Loaded %d grid cells from %s", len(grid_cells), GRID_CELLS_PATH)
    else:
        logger.warning("grid_cells.json absent — using 20 DAS Strategis Nasional stub")
        grid_cells = _DAS_STUB

    context["ti"].xcom_push(key="grid_cells", value=grid_cells)
    context["ti"].xcom_push(key="out_dir",    value=str(out_dir))
    context["ti"].xcom_push(key="run_ds",     value=run_ds)


def run_batch_xgb(**context) -> None:
    """XGBoost precipitation nowcast — all grid cells."""
    _run_batch_model("xgb_precip_nowcast", context)


def run_batch_lstm(**context) -> None:
    """LSTM streamflow forecast — all grid cells."""
    _run_batch_model("lstm_streamflow", context)


def run_batch_tft(**context) -> None:
    """TFT multi-variate climate forecast — all grid cells."""
    _run_batch_model("tft_climate_forecast", context)


def _run_batch_model(model_id: str, context: dict) -> None:
    """
    Core batch loop for a single model.
    Output: workspace/output/batch_forecasts/{YYYYMMDD}/{model_id}_{grid_cell_id}.json
    """
    from src.data.feature_store_client import FeatureStoreClient
    from src.serving.inference_cache   import InferenceCache

    ti         = context["ti"]
    grid_cells = ti.xcom_pull(key="grid_cells", task_ids="prepare_grid_cells")
    out_dir    = Path(ti.xcom_pull(key="out_dir", task_ids="prepare_grid_cells"))
    run_ds     = ti.xcom_pull(key="run_ds",     task_ids="prepare_grid_cells")
    issued     = date.fromisoformat(run_ds)
    horizon    = FORECAST_HORIZONS.get(model_id, 72)

    fs     = FeatureStoreClient()
    cache  = InferenceCache()
    model  = _load_model(model_id)
    errors = 0

    for gc in grid_cells:
        gc_id = gc["grid_cell_id"]
        try:
            features_df = fs.get_online_features(
                entity_rows=[{"grid_cell_id": gc_id}],
                feature_refs=getattr(model, "FEATURE_REFS", []),
            )
            input_key = {"grid_cell_id": gc_id, "forecast_horizon": horizon, "issued_date": run_ds}
            pred = cache.get(model_id, getattr(model, "version", "prod"), input_key)
            if not pred:
                pred = model.predict(features=features_df, forecast_horizon=horizon, issued_date=issued)
                cache.set(model_id, getattr(model, "version", "prod"), input_key, pred)

            # Output path: {model_id}_{grid_cell_id}.json (Sprint 6 spec)
            out_file = out_dir / f"{model_id}_{gc_id}.json"
            out_file.write_text(json.dumps({
                "model_id":            model_id,
                "grid_cell_id":        gc_id,
                "issued_date":         run_ds,
                "forecast_horizon_h":  horizon,
                "prediction":          pred.get("prediction"),
                "confidence_interval": pred.get("confidence_interval"),
                "province":            gc.get("province"),
                "region":              gc.get("region"),
            }, default=str))
        except Exception as exc:
            logger.warning("Batch %s failed for gc=%s: %s", model_id, gc_id, exc)
            errors += 1

    total = len(grid_cells)
    logger.info("Batch %s: %d ok, %d errors", model_id, total - errors, errors)
    ti.xcom_push(key=f"result_{model_id}", value={"model_id": model_id, "total": total, "errors": errors})


def aggregate_outputs(**context) -> None:
    """Merge per-model XCom summaries into manifest.json."""
    ti      = context["ti"]
    out_dir = Path(ti.xcom_pull(key="out_dir", task_ids="prepare_grid_cells"))
    run_ds  = ti.xcom_pull(key="run_ds",    task_ids="prepare_grid_cells")

    manifest: Dict[str, Any] = {"run_date": run_ds, "models": {}}
    for model_id in BATCH_MODELS:
        result = ti.xcom_pull(key=f"result_{model_id}", task_ids=f"run_batch_{model_id.split('_')[0]}")
        manifest["models"][model_id] = result or {}

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    ti.xcom_push(key="manifest", value=manifest)
    logger.info("Manifest written to %s/manifest.json", out_dir)


def write_batch_report(**context) -> None:
    """
    Write summary.md consumed by VISUALIA.
    Calls record_ingestion_success('batch_inference_daily') on success.
    """
    ti       = context["ti"]
    out_dir  = Path(ti.xcom_pull(key="out_dir",  task_ids="prepare_grid_cells"))
    manifest = ti.xcom_pull(key="manifest",       task_ids="aggregate_outputs")
    run_ds   = ti.xcom_pull(key="run_ds",         task_ids="prepare_grid_cells")

    total_cells  = sum(m.get("total",  0) for m in manifest["models"].values())
    total_errors = sum(m.get("errors", 0) for m in manifest["models"].values())
    yyyymmdd     = run_ds.replace("-", "")

    lines = [
        f"# ANALYTICA Batch Inference Report — {run_ds}",
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
    for model_id, m in manifest["models"].items():
        lines.append(f"| `{model_id}` | {m.get('total', 0)} | {m.get('errors', 0)} |")

    lines += [
        "",
        "## Output Files",
        "",
        f"Path: `workspace/output/batch_forecasts/{yyyymmdd}/`  ",
        f"Pattern: `{{model_id}}_{{grid_cell_id}}.json`  ",
        "Index: `manifest.json`",
        "",
        "---",
        "_Auto-generated by analytica_batch_inference DAG (Sprint 6 H5). Consumed by VISUALIA._",
    ]

    (out_dir / "summary.md").write_text("\n".join(lines))
    logger.info("Batch report written: %s/summary.md", out_dir)

    try:
        from src.data.ingestion_tracker import record_ingestion_success
        record_ingestion_success("batch_inference_daily")
        logger.info("record_ingestion_success('batch_inference_daily') OK")
    except ImportError:
        logger.warning("ingestion_tracker unavailable — skipping record_ingestion_success")
    except Exception as exc:
        logger.warning("record_ingestion_success failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_id: str) -> Any:
    """Load from MLflow Production; fall back to stub."""
    try:
        from src.training.mlflow_registry import MLflowRegistry
        import mlflow.pyfunc
        MLflowRegistry().get_latest_production(model_id)
        return mlflow.pyfunc.load_model(f"models:/{model_id}/Production")
    except Exception as exc:
        logger.warning("Model %s unavailable (%s) — using stub", model_id, exc)
        return _BatchStub(model_id)


class _BatchStub:
    FEATURE_REFS = []
    version = "stub"

    def __init__(self, model_id: str):
        self.model_id = model_id

    def predict(self, **kwargs) -> dict:
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
    description="Daily batch inference: XGB + LSTM + TFT over 20 DAS Strategis Nasional",
) as dag:

    t_prepare   = PythonOperator(task_id="prepare_grid_cells", python_callable=prepare_grid_cells)
    t_xgb       = PythonOperator(task_id="run_batch_xgb",      python_callable=run_batch_xgb)
    t_lstm      = PythonOperator(task_id="run_batch_lstm",      python_callable=run_batch_lstm)
    t_tft       = PythonOperator(task_id="run_batch_tft",       python_callable=run_batch_tft)
    t_aggregate = PythonOperator(task_id="aggregate_outputs",   python_callable=aggregate_outputs)
    t_report    = PythonOperator(task_id="write_batch_report",  python_callable=write_batch_report)

    t_prepare >> [t_xgb, t_lstm, t_tft] >> t_aggregate >> t_report
