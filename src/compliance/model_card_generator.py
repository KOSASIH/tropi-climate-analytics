"""
Model Card Generator — ANALYTICA Sprint 7 L5
Module: src/compliance/model_card_generator.py

Class: ModelCardGenerator
  generate(model_id, mlflow_run_id=None) → ModelCard

Sections:
  model_details    — name, version, type, training_date, MLflow run_id, commit SHA
  intended_use     — primary use, out-of-scope, deployment geography (Indonesia)
  training_data    — feature store entities, date range, n_samples, DQ result summary
  evaluation       — metrics table (MAE/RMSE/MAPE/NSE/KGE/F1), per-region breakdown (5 islands)
  explainability   — top-10 SHAP features (from workspace/output/explainability/)
  limitations      — known failure modes, drift thresholds, retraining schedule
  regulatory_notes — PP 71/2019 article refs, data residency (Jakarta region), auditability

Output:
  workspace/output/model_cards/{model_id}_v{version}_card.md
  workspace/output/model_cards/{model_id}_v{version}_card.json

Auto-triggered: on MLflowRegistry.promote_to_production() via post-promote hook
Prometheus: MODEL_CARD_GENERATED{model_id, version} Counter
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

OUTPUT_DIR    = Path("workspace/output/model_cards")
EXPLAINER_DIR = Path("workspace/output/explainability")

# Model metadata registry
MODEL_REGISTRY_META: Dict[str, Dict[str, Any]] = {
    "xgb_precipitation": {
        "display_name":   "XGBoost Precipitation Nowcaster",
        "type":           "Gradient Boosted Trees (XGBoost)",
        "use_case":       "24–72 hour precipitation nowcasting for Indonesian weather stations",
        "entity_types":   ["station_id", "grid_cell_id"],
        "out_of_scope":   ["Long-range (>72h) forecasting", "Non-tropical climate regimes",
                           "Regions outside Indonesia"],
        "primary_metric": "val_mae",
        "retraining_schedule": "Weekly (Monday 02:00 WIB) — RETRAIN_XGB variable gate",
        "failure_modes":  ["Extreme convective events (typhoon-grade)", "Missing upstream station data",
                           "Rapid ENSO phase transitions"],
    },
    "prophet_seasonal": {
        "display_name":   "Prophet Seasonal Climate Forecaster",
        "type":           "Additive Decomposition Model (Prophet)",
        "use_case":       "Monthly seasonal streamflow and precipitation forecasting at watershed scale",
        "entity_types":   ["watershed_id"],
        "out_of_scope":   ["Sub-daily forecasting", "Point-level station nowcasting"],
        "primary_metric": "val_mape",
        "retraining_schedule": "Monthly (1st 03:00 WIB) — RETRAIN_PROPHET variable gate",
        "failure_modes":  ["Anomalous ENSO seasons (El Niño / La Niña extremes)",
                           "Structural breaks in land use patterns"],
    },
    "cnn_landcover": {
        "display_name":   "CNN Land Cover Classifier (ResNet-18)",
        "type":           "Convolutional Neural Network (ResNet-18, 4-channel input)",
        "use_case":       "LULC classification from multispectral satellite imagery (8 classes)",
        "entity_types":   ["grid_cell_id"],
        "out_of_scope":   ["Sub-10m resolution classification", "Radar-based precipitation estimation"],
        "primary_metric": "val_macro_f1",
        "retraining_schedule": "Quarterly (1st 04:00 WIB) — RETRAIN_CNN variable gate",
        "failure_modes":  ["Cloud-contaminated scenes (>30% cloud cover)",
                           "Seasonal greenness changes misclassified as deforestation",
                           "Rare class underrepresentation (e.g. mangrove, wetland)"],
    },
    "lstm_streamflow": {
        "display_name":   "LSTM Streamflow Predictor",
        "type":           "2-Layer LSTM (hidden=256, dropout=0.2, seq_len=72h)",
        "use_case":       "6–72 hour streamflow prediction for Ciliwung, Brantas, Solo river systems",
        "entity_types":   ["watershed_id"],
        "out_of_scope":   ["Unmonitored ungauged watersheds without SMAP data",
                           "Flash flood events shorter than 3-hour response time"],
        "primary_metric": "val_nse",
        "retraining_schedule": "Monthly (15th 03:00 WIB) — RETRAIN_LSTM variable gate",
        "failure_modes":  ["Upstream dam operations not captured in training data",
                           "Extreme rainfall exceeding 500mm/6hr (outside training distribution)"],
    },
}

# PP 71/2019 — Indonesian Government Regulation on Electronic Systems
PP71_ARTICLES: Dict[str, str] = {
    "Article 3":   "Electronic system classification — Public system requiring registration with BSSN",
    "Article 17":  "Data processing in Indonesian territory — Jakarta region (ap-southeast-3) compliant",
    "Article 22":  "Electronic data protection — encrypted at rest (AES-256) and in transit (TLS 1.3)",
    "Article 29":  "Auditability — MLflow experiment tracking provides complete training audit trail",
    "Article 30":  "Incident response — RETRAIN variable gate + Prometheus alerting pipeline",
    "Article 40":  "Data retention — model artifacts retained 5 years per BIG archival standard",
}

INDONESIA_REGIONS = ["Jawa", "Sumatera", "Kalimantan", "Sulawesi", "Papua"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    model_id:        str
    version:         str
    model_details:   Dict[str, Any]        = field(default_factory=dict)
    intended_use:    Dict[str, Any]        = field(default_factory=dict)
    training_data:   Dict[str, Any]        = field(default_factory=dict)
    evaluation:      Dict[str, Any]        = field(default_factory=dict)
    explainability:  Dict[str, Any]        = field(default_factory=dict)
    limitations:     Dict[str, Any]        = field(default_factory=dict)
    regulatory_notes: Dict[str, Any]       = field(default_factory=dict)
    generated_at:    str                   = ""
    card_md_path:    Optional[str]         = None
    card_json_path:  Optional[str]         = None

    def to_dict(self) -> dict:
        return {
            "model_id":        self.model_id,
            "version":         self.version,
            "generated_at":    self.generated_at,
            "model_details":   self.model_details,
            "intended_use":    self.intended_use,
            "training_data":   self.training_data,
            "evaluation":      self.evaluation,
            "explainability":  self.explainability,
            "limitations":     self.limitations,
            "regulatory_notes": self.regulatory_notes,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ModelCardGenerator:
    """
    Auto-generates model documentation for PP 71/2019 and Indonesian AI governance compliance.
    Called post-promote in MLflowRegistry.promote_to_production() and by retraining DAGs.
    """

    def generate(
        self,
        model_id:     str,
        mlflow_run_id: Optional[str] = None,
        version:      Optional[str] = None,
    ) -> ModelCard:
        """
        Generate a model card for the specified model.

        Args:
            model_id:      Registered MLflow model name.
            mlflow_run_id: Specific training run ID (if None, fetches latest Production run).
            version:       Model version string (if None, fetches from MLflow).

        Returns:
            ModelCard dataclass with all sections populated.
        """
        now_str  = datetime.now(timezone.utc).isoformat()
        meta     = MODEL_REGISTRY_META.get(model_id, {})

        # Resolve run and version from MLflow
        run_data, version = self._resolve_mlflow(model_id, mlflow_run_id, version)

        card = ModelCard(
            model_id=model_id,
            version=version,
            generated_at=now_str,
        )

        card.model_details   = self._build_model_details(model_id, version, meta, run_data)
        card.intended_use    = self._build_intended_use(meta)
        card.training_data   = self._build_training_data(model_id, run_data)
        card.evaluation      = self._build_evaluation(model_id, run_data)
        card.explainability  = self._build_explainability(model_id)
        card.limitations     = self._build_limitations(meta)
        card.regulatory_notes = self._build_regulatory_notes(model_id)

        # Write outputs
        card.card_md_path   = self._write_markdown(card)
        card.card_json_path = self._write_json(card)

        # Prometheus
        try:
            from src.data.metrics import MODEL_CARD_GENERATED
            MODEL_CARD_GENERATED.labels(model_id=model_id, version=str(version)).inc()
        except ImportError:
            pass

        logger.info("Model card generated: %s v%s → %s", model_id, version, card.card_md_path)
        return card

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _resolve_mlflow(
        self, model_id: str, run_id: Optional[str], version: Optional[str]
    ):
        run_data = {}
        resolved_version = version or "unknown"
        try:
            import mlflow
            client = mlflow.MlflowClient()
            if run_id:
                run = client.get_run(run_id)
                run_data = {"params": run.data.params, "metrics": run.data.metrics, "tags": run.data.tags}
                run_data["run_id"] = run_id
            else:
                mvs = client.get_latest_versions(model_id, stages=["Production"])
                if mvs:
                    resolved_version = mvs[0].version
                    run = client.get_run(mvs[0].run_id)
                    run_data = {
                        "params":  run.data.params,
                        "metrics": run.data.metrics,
                        "tags":    run.data.tags,
                        "run_id":  mvs[0].run_id,
                    }
        except Exception as exc:
            logger.warning("MLflow unavailable for model card (%s): %s", model_id, exc)
        return run_data, resolved_version

    @staticmethod
    def _build_model_details(
        model_id: str, version: str, meta: dict, run_data: dict
    ) -> dict:
        params  = run_data.get("params", {})
        tags    = run_data.get("tags", {})
        return {
            "name":          meta.get("display_name", model_id),
            "model_id":      model_id,
            "version":       version,
            "type":          meta.get("type", "Unknown"),
            "framework":     _infer_framework(model_id),
            "training_date": tags.get("training_date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
            "mlflow_run_id": run_data.get("run_id", "N/A"),
            "commit_sha":    tags.get("commit_sha", _get_latest_sha()),
            "mlflow_experiment": f"{model_id.replace('_', '-')}_retraining",
            "hyperparameters": {k: v for k, v in params.items() if k in (
                "n_estimators", "learning_rate", "max_depth", "hidden_size", "n_layers",
                "epochs", "batch_size", "lr", "seq_len", "dropout",
            )},
        }

    @staticmethod
    def _build_intended_use(meta: dict) -> dict:
        return {
            "primary_use_case":     meta.get("use_case", "Climate prediction"),
            "deployment_geography": "Indonesia — all 34 provinces, 5 major island groups",
            "target_users":         ["BMKG (meteorology agency)", "KLHK (forestry & environment)",
                                     "BNPB (disaster mitigation)", "LAPAN (space agency)",
                                     "Regional government flood management units"],
            "out_of_scope_uses":    meta.get("out_of_scope", []),
            "deployment_context":   "Operational production service — Tropi Climate Analytics API Gateway",
        }

    @staticmethod
    def _build_training_data(model_id: str, run_data: dict) -> dict:
        params  = run_data.get("params", {})
        metrics = run_data.get("metrics", {})
        meta    = MODEL_REGISTRY_META.get(model_id, {})
        return {
            "feature_store_entities":  meta.get("entity_types", []),
            "data_sources":            ["NASA GPM (precipitation)", "BMKG ground stations",
                                        "NASA MODIS (NDVI/land cover)", "NASA SMAP (soil moisture)",
                                        "LAPAN Landsat (satellite imagery)"],
            "training_window":         params.get("training_window", "See retraining DAG config"),
            "n_train_samples":         metrics.get("n_train", "See MLflow run"),
            "n_validation_samples":    metrics.get("n_eval", metrics.get("n_val", "See MLflow run")),
            "data_quality_gate":       "FeatureStoreDataQuality.run_suite() — DataQualityError blocks training",
            "data_quality_summary":    "Violations logged to workspace/output/data_quality/",
            "preprocessing":           ["Z-score normalization (LSTM)", "Quantile-binning (XGB PSI)",
                                        "Temporal lag engineering (FeaturePipeline)",
                                        "Cyclical encoding (hour-of-day, day-of-week)"],
        }

    @staticmethod
    def _build_evaluation(model_id: str, run_data: dict) -> dict:
        metrics = run_data.get("metrics", {})
        # Build metrics table from whatever is available
        metric_table: Dict[str, Any] = {}
        for k, v in metrics.items():
            if any(m in k for m in ("mae", "rmse", "mape", "nse", "kge", "f1")):
                metric_table[k] = round(float(v), 4) if v else "N/A"

        # Per-region breakdown structure (populated from regional eval if available)
        regional = {island: {} for island in INDONESIA_REGIONS}

        return {
            "holdout_split":      "Last 15% chronological (temporal integrity preserved)",
            "cv_strategy":        "5-fold time-series CV (XGBoost) / Last 90d holdout (Prophet, LSTM)",
            "metrics":            metric_table if metric_table else {"note": "Fetch from MLflow run"},
            "per_region_metrics": regional,
            "promotion_gate":     _promotion_gate_description(model_id),
            "evaluation_cadence": "Post-retraining (automatic) + weekly performance monitoring",
        }

    @staticmethod
    def _build_explainability(model_id: str) -> dict:
        """Load top-10 SHAP features from workspace/output/explainability/ if available."""
        top_shap: List[Dict[str, Any]] = []
        pattern = f"shap_{model_id}_*.json"
        shap_files = sorted(EXPLAINER_DIR.glob(pattern)) if EXPLAINER_DIR.exists() else []
        if shap_files:
            try:
                data = json.loads(shap_files[-1].read_text())
                scores = data.get("mean_abs_shap", data.get("importance_scores", {}))
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
                top_shap = [{"feature": k, "mean_abs_shap": round(v, 4)} for k, v in ranked]
            except Exception as exc:
                logger.warning("Could not load SHAP scores for %s: %s", model_id, exc)

        return {
            "method":            "SHAP (SHapley Additive exPlanations)",
            "explainer_type":    "TreeExplainer (XGB/LSTM) / KernelExplainer (Prophet)",
            "top_10_features":   top_shap if top_shap else [{"note": "Run SHAPExplainer to populate"}],
            "shap_output_dir":   str(EXPLAINER_DIR),
            "global_vs_local":   "Both available — global (mean |SHAP|) and local (per-prediction waterfall)",
            "feature_pipeline":  "FeaturePipeline.select_features(method='shap') selects top-30 for retraining",
        }

    @staticmethod
    def _build_limitations(meta: dict) -> dict:
        return {
            "known_failure_modes":    meta.get("failure_modes", []),
            "drift_thresholds":       {
                "psi_warning":         0.2,
                "psi_critical":        0.25,
                "ks_pvalue_threshold": 0.05,
                "mae_degraded_factor": 1.15,
                "mae_critical_factor": 1.30,
            },
            "retraining_schedule":    meta.get("retraining_schedule", "See DAG configuration"),
            "retraining_trigger":     "Automatic via ModelMonitor CRITICAL → Airflow Variable → DAG ShortCircuit",
            "data_coverage_gaps":     ["Eastern Indonesia (Papua/Maluku) — sparse ground station density",
                                       "Offshore maritime zones — no in-situ precipitation gauge coverage"],
            "temporal_limitations":   "Models trained on 2018–2026 data; pre-2018 climate regime may differ",
            "uncertainty_note":       "EnsembleForecaster 90% CI via conformal prediction; degrades at distribution shift",
        }

    @staticmethod
    def _build_regulatory_notes(model_id: str) -> dict:
        return {
            "regulation":          "PP Number 71 Tahun 2019 — Penyelenggaraan Sistem dan Transaksi Elektronik",
            "articles":            PP71_ARTICLES,
            "data_residency":      "AWS ap-southeast-3 (Jakarta) — all training data and model artifacts",
            "classification":      "Strategis — climate forecasting system for national disaster mitigation",
            "bssn_registration":   "Required — Electronic system operator registration under Article 3",
            "data_processor":      "Tropi Climate Analytics Platform — KOSASIH (operator)",
            "audit_trail":         f"MLflow experiment: {model_id.replace('_', '-')}_retraining; "
                                   f"all runs versioned and immutable",
            "retention_policy":    "Model artifacts: 5 years | Training logs: 7 years (BIG archival standard)",
            "pii_handling":        "No PII processed — geospatial environmental data only",
            "last_compliance_review": datetime.now(timezone.utc).strftime("%Y-%m"),
        }

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def _write_markdown(self, card: ModelCard) -> str:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"{card.model_id}_v{card.version}_card.md"
        md    = self._render_markdown(card)
        path  = OUTPUT_DIR / fname
        path.write_text(md)
        return str(path)

    def _write_json(self, card: ModelCard) -> str:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"{card.model_id}_v{card.version}_card.json"
        path  = OUTPUT_DIR / fname
        payload = {**card.to_dict(), "written_at": card.generated_at}
        path.write_text(json.dumps(payload, indent=2, default=str))
        return str(path)

    @staticmethod
    def _render_markdown(card: ModelCard) -> str:
        d  = card.model_details
        u  = card.intended_use
        tr = card.training_data
        ev = card.evaluation
        ex = card.explainability
        li = card.limitations
        rn = card.regulatory_notes

        def _table(rows: List[tuple]) -> str:
            return "| Metric | Value |\n|--------|-------|\n" + \
                   "\n".join(f"| {k} | {v} |" for k, v in rows)

        shap_rows = "\n".join(
            f"| {i+1} | {f.get('feature','—')} | {f.get('mean_abs_shap','—')} |"
            for i, f in enumerate(ex.get("top_10_features", []))
        )
        metrics_rows = "\n".join(
            f"| {k} | {v} |"
            for k, v in ev.get("metrics", {}).items()
        ) or "| — | Run MLflow to populate |"

        article_rows = "\n".join(
            f"| {art} | {desc} |"
            for art, desc in rn.get("articles", {}).items()
        )

        return f"""# Model Card: {d.get('name', card.model_id)}

> Auto-generated by ModelCardGenerator — ANALYTICA Sprint 7 L5
> Generated: {card.generated_at}

---

## 1. Model Details

| Field | Value |
|-------|-------|
| Model ID | `{card.model_id}` |
| Version | `{card.version}` |
| Type | {d.get('type', '—')} |
| Framework | {d.get('framework', '—')} |
| Training Date | {d.get('training_date', '—')} |
| MLflow Run ID | `{d.get('mlflow_run_id', '—')}` |
| Commit SHA | `{d.get('commit_sha', '—')}` |
| MLflow Experiment | `{d.get('mlflow_experiment', '—')}` |

### Hyperparameters
```json
{json.dumps(d.get('hyperparameters', {}), indent=2)}
```

---

## 2. Intended Use

- **Primary Use Case:** {u.get('primary_use_case', '—')}
- **Deployment Geography:** {u.get('deployment_geography', '—')}
- **Target Users:** {', '.join(u.get('target_users', []))}
- **Deployment Context:** {u.get('deployment_context', '—')}

### Out-of-Scope Uses
{chr(10).join(f"- {x}" for x in u.get('out_of_scope_uses', ['None documented']))}

---

## 3. Training Data

| Field | Value |
|-------|-------|
| Feature Store Entities | {', '.join(tr.get('feature_store_entities', []))} |
| Training Window | {tr.get('training_window', '—')} |
| N Train Samples | {tr.get('n_train_samples', '—')} |
| N Validation Samples | {tr.get('n_validation_samples', '—')} |
| Data Quality Gate | {tr.get('data_quality_gate', '—')} |

**Data Sources:** {', '.join(tr.get('data_sources', []))}

**Preprocessing:** {', '.join(tr.get('preprocessing', []))}

---

## 4. Evaluation

- **Holdout Split:** {ev.get('holdout_split', '—')}
- **CV Strategy:** {ev.get('cv_strategy', '—')}
- **Promotion Gate:** {ev.get('promotion_gate', '—')}
- **Evaluation Cadence:** {ev.get('evaluation_cadence', '—')}

### Metrics (Holdout)
| Metric | Value |
|--------|-------|
{metrics_rows}

---

## 5. Explainability

- **Method:** {ex.get('method', '—')}
- **Explainer Type:** {ex.get('explainer_type', '—')}
- **Scope:** {ex.get('global_vs_local', '—')}

### Top-10 SHAP Features
| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
{shap_rows if shap_rows else "| — | Run SHAPExplainer to populate | — |"}

---

## 6. Limitations

- **Retraining Schedule:** {li.get('retraining_schedule', '—')}
- **Retraining Trigger:** {li.get('retraining_trigger', '—')}
- **Temporal Limitation:** {li.get('temporal_limitations', '—')}
- **Uncertainty Note:** {li.get('uncertainty_note', '—')}

### Drift Thresholds
```json
{json.dumps(li.get('drift_thresholds', {}), indent=2)}
```

### Known Failure Modes
{chr(10).join(f"- {x}" for x in li.get('known_failure_modes', ['None documented']))}

### Data Coverage Gaps
{chr(10).join(f"- {x}" for x in li.get('data_coverage_gaps', []))}

---

## 7. Regulatory Notes (PP 71/2019)

| Field | Value |
|-------|-------|
| Regulation | {rn.get('regulation', '—')} |
| Classification | {rn.get('classification', '—')} |
| Data Residency | {rn.get('data_residency', '—')} |
| Data Processor | {rn.get('data_processor', '—')} |
| PII Handling | {rn.get('pii_handling', '—')} |
| Audit Trail | {rn.get('audit_trail', '—')} |
| Retention Policy | {rn.get('retention_policy', '—')} |
| Last Compliance Review | {rn.get('last_compliance_review', '—')} |

### PP 71/2019 Article References
| Article | Requirement |
|---------|-------------|
{article_rows}

---

*This model card was auto-generated by ANALYTICA ModelCardGenerator and should be reviewed by a qualified ML engineer before external distribution.*
"""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _infer_framework(model_id: str) -> str:
    if "xgb" in model_id:
        return "XGBoost 2.x + MLflow"
    if "prophet" in model_id:
        return "Prophet (Meta) + MLflow pyfunc"
    if "cnn" in model_id:
        return "PyTorch (torchvision ResNet-18) + MLflow"
    if "lstm" in model_id:
        return "PyTorch + MLflow"
    return "Python MLflow"


def _promotion_gate_description(model_id: str) -> str:
    gates = {
        "xgb_precipitation": "Challenger MAE < Champion MAE × 0.97",
        "xgb_precip_nowcast": "Challenger MAE < Champion MAE × 0.97",
        "prophet_seasonal":  "Challenger MAPE < Champion MAPE × 0.97",
        "cnn_landcover":     "Challenger macro-F1 > Champion macro-F1 × 0.98 AND no per-class F1 drop > 5%",
        "lstm_streamflow":   "Challenger NSE > Champion NSE + 0.02 AND no river KGE degradation (Ciliwung/Brantas/Solo)",
    }
    return gates.get(model_id, "See MLflow promotion policy")


def _get_latest_sha() -> str:
    """Attempt to read latest commit SHA from git."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "N/A"
    except Exception:
        return "N/A"
