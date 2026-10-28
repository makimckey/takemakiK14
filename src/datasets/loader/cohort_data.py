"""Sleep-Data-Hubのコホート/CV APIを授業のtrain/val/testに接続する。"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from .sleep_loader import (
    CVConfig,
    CrossValidationSetting,
    load_cohort,
    build_cv_setting,
    load_training_testing_data,
    load_npz,
    make_torch_xy_from_loaded,
)


def validate_config(config):
    legacy = {"paths", "total_folds", "config", "transforms"}.intersection(config)
    if legacy:
        raise ValueError(
            f"Legacy dataset settings {sorted(legacy)}: migrate using "
            "docs/07_Loader_APIへの移行.md"
        )
    if int(config["cv"]["n_folds"]) < 3:
        raise ValueError("train/validation/test requires cv.n_folds >= 3")
    if config.get("limit") is not None and int(config["limit"]) < 1:
        raise ValueError("dataset.limit must be positive or null")
    if config.get("normalization", "none") not in (
        "none",
        "epoch-wise",
        "record-wise",
        "train-wise",
    ):
        raise ValueError("Unknown dataset.normalization")
    if config.get("y_spec", "stages(latest)") not in (
        "stages(latest)",
        "stages(center)",
        "stages(vote)",
    ):
        raise ValueError("This classifier requires stages(latest/center/vote)")
    if config.get("window_sec", 30) <= 0 or config.get("stride_sec", 30) <= 0:
        raise ValueError("window_sec and stride_sec must be positive")
    if not config.get("channel_set"):
        raise ValueError("dataset.channel_set must name the input channels")


def load_experiment_cohort(config, *, saved_setting=None):
    """SQLで教師あり記録を選び、Loaderでロード・被験者単位CVを作る。"""
    validate_config(config)
    where = "r.usage LIKE ?"
    params = ['%"supervised_stage"%']
    limit = config.get("limit")
    saved = None
    if saved_setting is not None:
        saved = json.loads(Path(saved_setting).read_text())
        if saved.get("version") != 1 or saved["cohort_id"] != config["cohort_id"]:
            raise ValueError("Saved CV setting does not match this cohort/version")
        if saved["cv"] != asdict(CVConfig(**config["cv"])):
            raise ValueError("Saved CV parameters differ from experiment config")
        ids = [row["record_id"] for row in saved["records"]]
        if not ids:
            raise ValueError("Saved CV setting contains no records")
        where += " AND r.record_id IN (" + ",".join("?" for _ in ids) + ")"
        params.extend(ids)
        limit = None
    elif config.get("where"):
        where += f" AND ({config['where']})"
        params.extend(config.get("params", []))

    cohort = load_cohort(
        config["cohort_dir"],
        cohort_id=config["cohort_id"],
        purpose="supervised_stage"
        + "".join(f"+signal({name})" for name in config["channel_set"]),
        where=where,
        params=params,
        limit=limit,
        allow_pickle=False,
    )
    try:
        selected = set(cohort.hits_df["record_id"].astype(str))
        actual = {obj.path.stem for obj in cohort.loaded.values()}
        expected = {r["record_id"] for r in saved["records"]} if saved else selected
        if not actual or actual != expected:
            raise ValueError(
                f"Some selected records could not be loaded: expected={len(expected)}, "
                f"loaded={len(actual)}. Check Loader messages and data placement."
            )
        if saved:
            cv = CrossValidationSetting(
                CVConfig(**saved["cv"]), saved["participant_to_fold"]
            )
            current = {(pid, obj.path.stem) for (pid, _), obj in cohort.loaded.items()}
            previous = {(r["participant_id"], r["record_id"]) for r in saved["records"]}
            if current != previous or {pid for pid, _ in current} != set(
                cv.participant_to_fold
            ):
                raise ValueError(
                    "Saved records/participants changed; cannot reproduce the split"
                )
            if any(
                not 0 <= int(f) < cv.n_folds for f in cv.participant_to_fold.values()
            ):
                raise ValueError("Saved fold assignment is invalid")
        else:
            cv = build_cv_setting(cohort, CVConfig(**config["cv"]))
        return cohort, cv
    except Exception:
        cohort.release()
        raise


def save_cv_setting(path, cohort, cv):
    content = {
        "version": 1,
        "cohort_id": cohort.cohort_id,
        "cv": {
            "n_folds": cv.n_folds,
            "seed": cv.config.seed,
            "shuffle_subjects": cv.config.shuffle_subjects,
        },
        "participant_to_fold": cv.participant_to_fold,
        "records": [
            {"participant_id": pid, "record_id": obj.path.stem}
            for (pid, _), obj in cohort.loaded.items()
        ],
    }
    Path(path).write_text(json.dumps(content, ensure_ascii=False, indent=2))


def _window_args(config):
    return dict(
        channel_set=config["channel_set"],
        window_sec=config.get("window_sec", 30.0),
        stride_sec=config.get("stride_sec", 30.0),
        y_spec=config.get("y_spec", "stages(latest)"),
    )


def _normalize(x, mode, *, stats=None):
    """単一記録の推論と、validationを除外したtrain-wise正規化用。"""
    if mode == "none":
        return x, None
    if mode == "train-wise" and stats is not None:
        mean, std = stats["mean"], stats["std"]
    else:
        axes = 2 if mode == "epoch-wise" else (0, 2)
        mean = x.mean(dim=axes, keepdim=True)
        std = x.std(dim=axes, keepdim=True, unbiased=False)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (x - mean) / std, {"mean": mean, "std": std}


def _keep_labels(y, excluded):
    return ~torch.isin(y, torch.as_tensor(excluded, dtype=y.dtype))


def _to_model_dataset(x, y):
    if len(x) == 0:
        raise ValueError(
            "A split contains no usable epochs; increase dataset.limit or check labels"
        )
    if torch.any((y < 0) | (y > 4)):
        raise ValueError(
            "Expected five-class stages 0..4; configure exclude_labels for other labels"
        )
    # Loader [N,C,L] -> モデル [N,1,L,C]。窓の時間幅はLoaderに任せる。
    return TensorDataset(x.transpose(1, 2).unsqueeze(1), y.long())


@dataclass
class PreparedFold:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    participants: dict
    normalization_stats: dict | None


def prepare_fold(cohort, cv, config, fold_idx, *, normalization_stats=None):
    mode = config.get("normalization", "none")
    # APIのtrainはvalidationも含む。train-wiseだけは3分割後に統計量を作る。
    bundle = load_training_testing_data(
        cohort,
        cv,
        fold_idx=fold_idx,
        **_window_args(config),
        normalization="none" if mode == "train-wise" else mode,
    )
    val_fold = (fold_idx + 1) % cv.n_folds
    is_validation_participant = torch.tensor(
        [
            cv.participant_to_fold[pid] == val_fold
            for pid in bundle.train.participant_ids
        ]
    )
    val_mask = is_validation_participant[bundle.train.participant_index]
    excluded = config.get("exclude_labels", [-1])
    pool_keep = _keep_labels(bundle.train.y, excluded)
    test_keep = _keep_labels(bundle.test.y, excluded)
    train_x, train_y = (
        bundle.train.x[~val_mask & pool_keep],
        bundle.train.y[~val_mask & pool_keep],
    )
    val_x, val_y = (
        bundle.train.x[val_mask & pool_keep],
        bundle.train.y[val_mask & pool_keep],
    )
    test_x, test_y = bundle.test.x[test_keep], bundle.test.y[test_keep]
    stats = None
    if mode == "train-wise":
        if len(train_x) == 0:
            raise ValueError("No training samples for normalization")
        train_x, stats = _normalize(train_x, mode, stats=normalization_stats)
        val_x, _ = _normalize(val_x, mode, stats=stats)
        test_x, _ = _normalize(test_x, mode, stats=stats)
    participants = {
        "train": sorted(
            pid
            for pid, f in cv.participant_to_fold.items()
            if f not in (fold_idx, val_fold)
        ),
        "val": sorted(
            pid for pid, f in cv.participant_to_fold.items() if f == val_fold
        ),
        "test": sorted(
            pid for pid, f in cv.participant_to_fold.items() if f == fold_idx
        ),
    }
    return PreparedFold(
        _to_model_dataset(train_x, train_y),
        _to_model_dataset(val_x, val_y),
        _to_model_dataset(test_x, test_y),
        participants,
        stats,
    )


def load_prediction_record(path, config, *, normalization_stats=None):
    """学習と同じ窓・正規化で単一記録を描画用に読む。返却索引は元の窓位置。"""
    mode = config.get("normalization", "none")
    if mode == "train-wise" and normalization_stats is None:
        raise ValueError(
            "train-wise prediction requires the saved training normalization statistics"
        )
    purpose = "supervised_stage" + "".join(
        f"+signal({name})" for name in config["channel_set"]
    )
    loaded = load_npz(path, purpose=purpose, allow_pickle=False)
    try:
        xs, ys, _, _ = make_torch_xy_from_loaded(
            loaded,
            x_channel_sets=[config["channel_set"]],
            window_sec=config.get("window_sec", 30),
            stride_sec=config.get("stride_sec", 30),
            y_specs=[config.get("y_spec", "stages(latest)")],
        )
        x, y = xs[0], ys[0]
        if mode != "train-wise":
            x, _ = _normalize(x, mode)
        keep = _keep_labels(y, config.get("exclude_labels", [-1]))
        indices = torch.where(keep)[0]
        selected = x[keep]
        if mode == "train-wise":
            selected, _ = _normalize(selected, mode, stats=normalization_stats)
        return _to_model_dataset(selected, y[keep]), y.numpy(), indices.numpy()
    finally:
        loaded.release()
