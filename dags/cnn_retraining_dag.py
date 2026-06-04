"""
CNN Land Cover Retraining DAG — ANALYTICA Sprint 6 I3
dag_id: analytica_cnn_retraining
Schedule: 0 4 1 */3 * Asia/Jakarta (quarterly 1st 04:00 WIB)
  Also triggerable: set Airflow Variable RETRAIN_CNN=true

SLA: 180 minutes

Pipeline:
  check_retrain_flag
    -> load_image_data      (workspace/data/satellite/landcover_{YYYYMM}/ stub path)
    -> train_cnn            (ResNet-18 backbone, 50 epochs, early stopping patience=7)
    -> evaluate_model       (per-class F1, macro-F1)
    -> register_if_improved (promote if macro-F1 > champion * 0.98 AND no per-class drop > 5%)
    -> clear_retrain_flag

Prometheus: RETRAINING_DURATION{model_id='cnn_landcover', status=success|failed}
MLflow experiment: cnn_landcover_retraining
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

DAG_ID      = "analytica_cnn_retraining"
SCHEDULE    = "0 4 1 */3 *"
TIMEZONE    = "Asia/Jakarta"
SLA_SECONDS = 180 * 60
MODEL_ID    = "cnn_landcover"
EXPERIMENT  = "cnn_landcover_retraining"
VARIABLE    = "RETRAIN_CNN"

EPOCHS         = 50
PATIENCE       = 7
BATCH_SIZE     = 32
LR             = 1e-4
IMG_SIZE       = (224, 224)

PROMO_F1_THRESHOLD          = 0.98   # macro-F1 > champion * 0.98
PROMO_PER_CLASS_MAX_DROP    = 0.05   # no per-class drop > 5%

LAND_COVER_CLASSES = [
    "forest", "shrubland", "grassland", "cropland",
    "wetland", "urban", "bareland", "water",
]


def check_retrain_flag(**context) -> bool:
    flag      = Variable.get(VARIABLE, default_var="false").strip().lower()
    is_manual = context.get("dag_run") and context["dag_run"].external_trigger
    should_run = (flag == "true") or bool(is_manual)
    if not should_run:
        logger.info("RETRAIN_CNN=%s and not manually triggered — short-circuiting", flag)
    return should_run


def load_image_data(**context) -> None:
    """
    Load satellite land cover imagery from stub path:
      workspace/data/satellite/landcover_{YYYYMM}/
    Falls back to synthetic stub if path absent (CI/staging environments).
    """
    run_ds   = context["ds"]
    run_dt   = datetime.strptime(run_ds, "%Y-%m-%d")
    yyyymm   = run_dt.strftime("%Y%m")
    img_path = Path(f"workspace/data/satellite/landcover_{yyyymm}")

    if img_path.exists():
        image_files = sorted(img_path.glob("*.npy"))
        label_file  = img_path / "labels.npy"
        if image_files and label_file.exists():
            X = np.stack([np.load(str(f)) for f in image_files])
            y = np.load(str(label_file))
            logger.info("Loaded %d images from %s", len(X), img_path)
        else:
            logger.warning("No .npy images in %s — using synthetic stub", img_path)
            X, y = _make_synthetic_stub()
    else:
        logger.warning("Image path %s absent — using synthetic stub", img_path)
        X, y = _make_synthetic_stub()

    # Save to workspace to avoid XCom size limits
    data_dir = Path("workspace/data/training")
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(data_dir / f"cnn_X_{run_ds.replace('-','')}.npy"), X)
    np.save(str(data_dir / f"cnn_y_{run_ds.replace('-','')}.npy"), y)

    context["ti"].xcom_push(key="data_prefix", value=str(data_dir / f"cnn_{run_ds.replace('-','')}"))
    context["ti"].xcom_push(key="n_samples",   value=int(len(X)))
    context["ti"].xcom_push(key="n_classes",   value=int(len(LAND_COVER_CLASSES)))


def _make_synthetic_stub() -> Tuple[np.ndarray, np.ndarray]:
    """Minimal synthetic stub: 200 samples, 4-channel 64×64 patches."""
    n, h, w, c = 200, 64, 64, 4
    X = np.random.rand(n, h, w, c).astype(np.float32)
    y = np.random.randint(0, len(LAND_COVER_CLASSES), size=n)
    return X, y


def train_cnn(**context) -> None:
    """
    Fine-tune ResNet-18 backbone for land cover classification.
    Uses PyTorch when available; falls back to sklearn dummy for CI environments.
    """
    import mlflow

    ti          = context["ti"]
    data_prefix = ti.xcom_pull(key="data_prefix", task_ids="load_image_data")
    n_classes   = ti.xcom_pull(key="n_classes",   task_ids="load_image_data")
    run_ds      = context["ds"]

    X = np.load(f"{data_prefix}_X.npy")
    y = np.load(f"{data_prefix}_y.npy")

    split_idx = int(len(X) * 0.85)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"cnn_train_{run_ds}") as run:
        mlflow.log_params({
            "epochs": EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
            "lr": LR, "backbone": "resnet18", "n_classes": n_classes,
            "n_train": len(X_train), "n_val": len(X_val),
        })

        try:
            macro_f1, per_class_f1 = _train_pytorch_resnet(X_train, y_train, X_val, y_val, n_classes)
        except ImportError:
            logger.warning("PyTorch unavailable — using sklearn DummyClassifier stub")
            macro_f1, per_class_f1 = _train_sklearn_stub(X_train, y_train, X_val, y_val, n_classes)

        mlflow.log_metric("val_macro_f1", macro_f1)
        for cls_idx, f1 in enumerate(per_class_f1):
            cls_name = LAND_COVER_CLASSES[cls_idx] if cls_idx < len(LAND_COVER_CLASSES) else f"class_{cls_idx}"
            mlflow.log_metric(f"val_f1_{cls_name}", f1)

        logger.info("CNN val macro-F1=%.4f per-class=%s", macro_f1, per_class_f1)
        run_id = run.info.run_id

    ti.xcom_push(key="run_id",         value=run_id)
    ti.xcom_push(key="val_macro_f1",   value=float(macro_f1))
    ti.xcom_push(key="per_class_f1",   value=[float(v) for v in per_class_f1])


def _train_pytorch_resnet(X_train, y_train, X_val, y_val, n_classes):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.models import resnet18
    from sklearn.metrics import f1_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Adapt 4-channel input: replace first conv to accept arbitrary channels
    model = resnet18(pretrained=False)
    in_ch = X_train.shape[-1] if X_train.ndim == 4 else 3
    model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc    = nn.Linear(model.fc.in_features, n_classes)
    model.to(device)

    def _to_tensor(X, y):
        # (N, H, W, C) -> (N, C, H, W)
        X_t = torch.from_numpy(X.transpose(0, 3, 1, 2)).float()
        y_t = torch.from_numpy(y).long()
        return X_t, y_t

    X_tr_t, y_tr_t = _to_tensor(X_train, y_train)
    X_va_t, y_va_t = _to_tensor(X_val,   y_val)
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=BATCH_SIZE)

    optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion  = nn.CrossEntropyLoss()
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_f1    = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                preds = model(Xb.to(device)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        if macro_f1 > best_f1:
            best_f1    = macro_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, PATIENCE)
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            all_preds.extend(model(Xb.to(device)).argmax(dim=1).cpu().numpy())
            all_labels.extend(yb.numpy())

    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(n_classes)))
    return float(best_f1), per_class.tolist()


def _train_sklearn_stub(X_train, y_train, X_val, y_val, n_classes):
    from sklearn.dummy    import DummyClassifier
    from sklearn.metrics  import f1_score
    X_flat = X_train.reshape(len(X_train), -1)
    clf = DummyClassifier(strategy="stratified", random_state=42)
    clf.fit(X_flat, y_train)
    preds = clf.predict(X_val.reshape(len(X_val), -1))
    per_class = f1_score(y_val, preds, average=None, zero_division=0, labels=list(range(n_classes)))
    macro_f1  = float(per_class.mean())
    return macro_f1, per_class.tolist()


def evaluate_model(**context) -> None:
    ti = context["ti"]
    logger.info("CNN val macro-F1=%.4f", ti.xcom_pull(key="val_macro_f1", task_ids="train_cnn"))


def register_if_improved(**context) -> None:
    import mlflow
    from src.training.mlflow_registry import MLflowRegistry

    try:
        from src.data.metrics import RETRAINING_DURATION
        _metrics = True
    except ImportError:
        _metrics = False

    ti           = context["ti"]
    run_id       = ti.xcom_pull(key="run_id",       task_ids="train_cnn")
    val_f1       = ti.xcom_pull(key="val_macro_f1", task_ids="train_cnn")
    per_class_f1 = ti.xcom_pull(key="per_class_f1", task_ids="train_cnn")
    t_start      = time.time()
    status       = "failed"
    registry     = MLflowRegistry()

    try:
        champion_f1: Optional[float]          = None
        champion_per_class: List[float]        = []
        try:
            mv = registry.get_latest_production(MODEL_ID)
            if mv:
                m_data = mlflow.MlflowClient().get_run(mv.run_id).data.metrics
                champion_f1 = m_data.get("val_macro_f1")
                champion_per_class = [
                    m_data.get(f"val_f1_{cls}", 0.0) for cls in LAND_COVER_CLASSES
                ]
        except Exception:
            pass

        new_mv = registry.register_model(run_id=run_id, model_name=MODEL_ID, metrics={"val_macro_f1": val_f1})

        # Promotion gate: macro-F1 > champion * 0.98 AND no per-class drop > 5%
        promote = False
        if champion_f1 is None:
            promote = True  # first model
            logger.info("No champion — promoting %s v%s as first Production", MODEL_ID, new_mv.version)
        elif val_f1 > champion_f1 * PROMO_F1_THRESHOLD:
            per_class_ok = True
            for i, (new_f1, champ_f1) in enumerate(zip(per_class_f1, champion_per_class)):
                if champ_f1 > 0 and (champ_f1 - new_f1) / champ_f1 > PROMO_PER_CLASS_MAX_DROP:
                    cls_name = LAND_COVER_CLASSES[i] if i < len(LAND_COVER_CLASSES) else f"class_{i}"
                    logger.warning("Per-class F1 drop > 5%% on class '%s': %.4f -> %.4f", cls_name, champ_f1, new_f1)
                    per_class_ok = False
            if per_class_ok:
                promote = True
            else:
                logger.info("NOT promoted: per-class F1 degradation blocked promotion")
        else:
            logger.info("NOT promoted: macro-F1 %.4f < champion %.4f * 0.98", val_f1, champion_f1)

        if promote:
            registry.promote_to_production(MODEL_ID, new_mv.version)
            logger.info("PROMOTED %s v%s (macro-F1=%.4f)", MODEL_ID, new_mv.version, val_f1)

        status = "success"
    except Exception as exc:
        logger.exception("register_if_improved(cnn) failed: %s", exc)
        raise
    finally:
        if _metrics:
            RETRAINING_DURATION.labels(model_id="cnn_landcover", status=status).observe(time.time() - t_start)


def clear_retrain_flag(**context) -> None:
    Variable.set(VARIABLE, "false")
    logger.info("Cleared %s", VARIABLE)


default_args = {
    "owner": "analytica", "depends_on_past": False, "email_on_failure": True,
    "retries": 1, "retry_delay": timedelta(minutes=15),
    "sla": timedelta(seconds=SLA_SECONDS),
}

with DAG(
    dag_id=DAG_ID, schedule_interval=SCHEDULE, start_date=days_ago(1),
    default_args=default_args, catchup=False,
    tags=["analytica", "retraining", "cnn"],
    description="Quarterly CNN land cover retraining — ResNet-18, F1 gate (RETRAIN_CNN-gated)",
) as dag:
    t_flag  = ShortCircuitOperator(task_id="check_retrain_flag",  python_callable=check_retrain_flag)
    t_load  = PythonOperator(task_id="load_image_data",           python_callable=load_image_data)
    t_train = PythonOperator(task_id="train_cnn",                 python_callable=train_cnn)
    t_eval  = PythonOperator(task_id="evaluate_model",            python_callable=evaluate_model)
    t_reg   = PythonOperator(task_id="register_if_improved",      python_callable=register_if_improved)
    t_clear = PythonOperator(task_id="clear_retrain_flag",        python_callable=clear_retrain_flag)

    t_flag >> t_load >> t_train >> t_eval >> t_reg >> t_clear
