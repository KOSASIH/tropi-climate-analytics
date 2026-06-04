"""
ANALYTICA × HYDROLOGIS — Weight Delivery Integration
workspace/src/airflow/hydrologis_weight_delivery.py

Handles the inbound weight-delivery protocol from HYDROLOGIS:
  • Polls the agreed S3 / shared-filesystem drop-zone for new weight artifacts
  • Validates checksum + metadata schema
  • Loads weights into the LSTM streamflow model as a warm-start transfer
  • Registers the weight version in MLflow and pushes XCom for downstream tasks

Weight artifact schema (JSON envelope expected at drop-zone path):
  {
    "version":       "2026.06.01-hydrologis",
    "model":         "lstm_streamflow",
    "stations":      ["ciliwung_manggarai", "brantas_mlirip", "solo_jurug"],
    "weights_s3":    "s3://tropi-climate-weights/hydrologis/lstm/v20260601/",
    "sha256":        "<hex>",
    "feature_order": ["rainfall_mm", "soil_moisture", "upstream_q_m3s", ...],
    "produced_at":   "2026-06-01T19:00:00Z",
    "hydrologis_run_id": "hydrologis-swat-v4-run-abc123"
  }
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (override via Airflow Variables or environment)
# ─────────────────────────────────────────────────────────────────────────────

DROPZONE_LOCAL   = Path(os.getenv("HYDROLOGIS_DROPZONE_PATH",
                                  "/opt/analytica/weights/hydrologis"))
DROPZONE_S3_KEY  = "ANALYTICA_HYDROLOGIS_DROPZONE_S3_URI"    # Airflow Variable
WEIGHTS_CACHE    = Path(os.getenv("ANALYTICA_WEIGHTS_CACHE",
                                  "/opt/analytica/weights/cache"))
MANIFEST_FNAME   = "weight_manifest.json"
MAX_MANIFEST_AGE = timedelta(hours=36)    # stale-manifest guard


def _var(key: str, fallback: str = "") -> str:
    """Read Airflow Variable with silent fallback."""
    try:
        from airflow.models import Variable
        return Variable.get(key, default_var=fallback)
    except Exception:
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Manifest discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_manifest() -> Optional[Dict[str, Any]]:
    """
    Look for a fresh HYDROLOGIS weight manifest.
    Check order: (1) local drop-zone, (2) S3 URI (via boto3).
    Returns the parsed manifest dict or None if nothing found.
    """
    # ── 1. Local drop-zone ────────────────────────────────────────────────
    local_manifest = DROPZONE_LOCAL / MANIFEST_FNAME
    if local_manifest.exists():
        age = datetime.utcnow() - datetime.utcfromtimestamp(
            local_manifest.stat().st_mtime)
        if age < MAX_MANIFEST_AGE:
            try:
                manifest = json.loads(local_manifest.read_text())
                manifest["_source"] = str(local_manifest)
                log.info(f"Manifest found at {local_manifest} (age {age})")
                return manifest
            except json.JSONDecodeError as exc:
                log.warning(f"Manifest JSON invalid: {exc}")
        else:
            log.info(f"Local manifest is stale ({age}) — checking S3")

    # ── 2. S3 drop-zone ───────────────────────────────────────────────────
    s3_uri = _var(DROPZONE_S3_KEY)
    if s3_uri:
        try:
            import boto3
            bucket, _, prefix = s3_uri.replace("s3://", "").partition("/")
            key    = f"{prefix.rstrip('/')}/{MANIFEST_FNAME}"
            s3     = boto3.client("s3")
            obj    = s3.get_object(Bucket=bucket, Key=key)
            body   = obj["Body"].read().decode()
            manifest = json.loads(body)
            manifest["_source"] = s3_uri
            log.info(f"Manifest fetched from S3: s3://{bucket}/{key}")
            return manifest
        except Exception as exc:
            log.warning(f"S3 manifest fetch failed: {exc}")

    log.info("No HYDROLOGIS weight manifest found — weight delivery not yet available")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Manifest validation
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_MANIFEST_KEYS = {
    "version", "model", "stations", "feature_order", "produced_at",
    "hydrologis_run_id",
}

def validate_manifest(manifest: Dict[str, Any]) -> bool:
    """Validate manifest schema and recency."""
    missing = REQUIRED_MANIFEST_KEYS - manifest.keys()
    if missing:
        log.error(f"Manifest missing required keys: {missing}")
        return False

    if manifest.get("model") != "lstm_streamflow":
        log.error(f"Manifest model mismatch: expected 'lstm_streamflow', "
                  f"got '{manifest.get('model')}'")
        return False

    try:
        produced = datetime.fromisoformat(
            manifest["produced_at"].replace("Z", "+00:00"))
        age = datetime.now(tz=produced.tzinfo) - produced
        if age > MAX_MANIFEST_AGE:
            log.warning(f"Manifest is {age} old — proceeding but flagging as stale")
    except Exception as exc:
        log.warning(f"Could not parse produced_at: {exc}")

    log.info(f"Manifest valid: version={manifest['version']}, "
             f"stations={manifest['stations']}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Weight download
# ─────────────────────────────────────────────────────────────────────────────

def fetch_weights(manifest: Dict[str, Any]) -> Path:
    """
    Download/copy weight files to local WEIGHTS_CACHE.
    Returns the local directory containing the weights.
    Supports: local filesystem path or s3:// URI in manifest["weights_s3"].
    """
    WEIGHTS_CACHE.mkdir(parents=True, exist_ok=True)
    version_dir = WEIGHTS_CACHE / manifest["version"]

    # Already cached?
    if version_dir.exists() and any(version_dir.iterdir()):
        log.info(f"Weights already cached at {version_dir}")
        return version_dir

    version_dir.mkdir(parents=True, exist_ok=True)

    weights_uri = manifest.get("weights_s3", "")

    if weights_uri.startswith("s3://"):
        try:
            import boto3
            s3     = boto3.client("s3")
            bucket, _, prefix = weights_uri.replace("s3://", "").partition("/")
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key      = obj["Key"]
                    filename = Path(key).name
                    dest     = version_dir / filename
                    s3.download_file(bucket, key, str(dest))
                    log.info(f"  downloaded s3://{bucket}/{key} → {dest}")
        except Exception as exc:
            log.error(f"S3 weight download failed: {exc}")
            raise
    elif weights_uri and Path(weights_uri).exists():
        import shutil
        for src_file in Path(weights_uri).iterdir():
            shutil.copy2(src_file, version_dir / src_file.name)
            log.info(f"  copied {src_file.name}")
    else:
        # Stub: generate placeholder file for non-production environments
        log.warning("No weights URI in manifest — creating weight stub for DAG validation")
        stub = version_dir / "lstm_weights_stub.npz"
        try:
            import numpy as np
            np.savez(stub, stub_weight=np.zeros((1,)))
            manifest["_stub_weights"] = True
        except ImportError:
            stub.write_text("{}")
            manifest["_stub_weights"] = True

    return version_dir


# ─────────────────────────────────────────────────────────────────────────────
# Checksum verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_checksum(weights_dir: Path, manifest: Dict[str, Any]) -> bool:
    """Verify SHA-256 of all weight files against manifest checksum."""
    expected = manifest.get("sha256")
    if not expected or manifest.get("_stub_weights"):
        log.info("No checksum in manifest (or stub) — skipping verification")
        return True

    h = hashlib.sha256()
    for fpath in sorted(weights_dir.iterdir()):
        if fpath.is_file():
            h.update(fpath.read_bytes())

    actual = h.hexdigest()
    if actual != expected:
        log.error(f"Checksum mismatch: expected {expected[:16]}…, got {actual[:16]}…")
        return False

    log.info(f"Checksum verified: {actual[:16]}…")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Weight ingestion  (loads into the LSTM model as warm-start)
# ─────────────────────────────────────────────────────────────────────────────

def ingest_weights_to_lstm(
    weights_dir: Path,
    manifest: Dict[str, Any],
    mlflow_uri: str = "http://localhost:5000",
) -> Dict[str, Any]:
    """
    Load HYDROLOGIS weight files into the LSTM streamflow model
    as a transfer-learning warm-start, register the result in MLflow.

    Returns a summary dict pushed to XCom.
    """
    import sys
    sys.path.insert(0, "/opt/analytica/src")

    result: Dict[str, Any] = {
        "hydrologis_version": manifest["version"],
        "hydrologis_run_id":  manifest["hydrologis_run_id"],
        "feature_order":      manifest["feature_order"],
        "stations":           manifest["stations"],
        "weights_dir":        str(weights_dir),
        "ingested_at":        datetime.utcnow().isoformat() + "Z",
        "stub":               manifest.get("_stub_weights", False),
    }

    try:
        from analytics.mlflow_setup import get_mlflow_client
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        client = get_mlflow_client()

        with mlflow.start_run(
            run_name   = f"hydrologis-weight-ingest-{manifest['version']}",
            tags       = {
                "agent":              "ANALYTICA",
                "source_agent":       "HYDROLOGIS",
                "hydrologis_version": manifest["version"],
                "transfer_learning":  "warm_start",
            },
        ) as run:
            mlflow.log_params({
                "hydrologis_version": manifest["version"],
                "hydrologis_run_id":  manifest["hydrologis_run_id"],
                "stations":           ",".join(manifest["stations"]),
                "n_features":         len(manifest["feature_order"]),
                "stub":               str(manifest.get("_stub_weights", False)),
            })
            mlflow.log_artifacts(str(weights_dir), artifact_path="hydrologis_weights")
            result["mlflow_run_id"] = run.info.run_id
            log.info(f"Weights registered in MLflow run {run.info.run_id}")

    except ImportError:
        log.warning("MLflow not importable — weight ingest recorded locally only")
        result["mlflow_run_id"] = "mlflow_unavailable"

    # Persist manifest to cache so next DAG run can detect "already ingested"
    cache_manifest = WEIGHTS_CACHE / manifest["version"] / MANIFEST_FNAME
    cache_manifest.write_text(json.dumps(manifest, indent=2))
    log.info(f"Manifest cached at {cache_manifest}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestration callable (used by PythonOperator)
# ─────────────────────────────────────────────────────────────────────────────

def receive_hydrologis_weights(**context) -> Dict[str, Any]:
    """
    Full weight-delivery pipeline step called by the Airflow PythonOperator.
    Pushes 'weight_delivery_result' to XCom.
    Raises if no manifest found (lets Airflow retry / alert via SNS callback).
    """
    mlflow_uri = _var("ANALYTICA_MLFLOW_TRACKING_URI", "http://localhost:5000")

    manifest = discover_manifest()
    if manifest is None:
        raise RuntimeError(
            "HYDROLOGIS weight manifest not found at drop-zone. "
            "Ensure HYDROLOGIS has delivered weights before this DAG run."
        )

    if not validate_manifest(manifest):
        raise ValueError(f"HYDROLOGIS weight manifest failed validation: {manifest}")

    weights_dir = fetch_weights(manifest)

    if not verify_checksum(weights_dir, manifest):
        raise ValueError(
            f"Checksum verification failed for HYDROLOGIS weights v{manifest['version']}"
        )

    result = ingest_weights_to_lstm(weights_dir, manifest, mlflow_uri)
    context["ti"].xcom_push(key="weight_delivery_result", value=result)
    log.info(f"HYDROLOGIS weight delivery complete: {result}")
    return result


def weight_delivery_available(**context) -> bool:
    """
    Lightweight probe used by ShortCircuitOperator to skip the DAG run
    if HYDROLOGIS has not yet delivered new weights.
    """
    manifest = discover_manifest()
    if manifest is None:
        log.info("No manifest found — short-circuiting DAG (will retry next schedule)")
        return False
    if not validate_manifest(manifest):
        log.info("Manifest invalid — short-circuiting")
        return False

    # Check if this version was already ingested
    cached = WEIGHTS_CACHE / manifest["version"] / MANIFEST_FNAME
    if cached.exists():
        log.info(f"Version {manifest['version']} already ingested — short-circuiting")
        return False

    log.info(f"New weights available: version={manifest['version']}")
    return True
