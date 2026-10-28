"""実際のLoader APIで、SQLite選択と被験者CV・評価の再現性を検証する。"""

import json
import sqlite3

import pytest
import torch

from test_standard_sleep_data import write_record
from src.datasets.loader.cohort_data import (
    load_experiment_cohort,
    save_cv_setting,
    prepare_fold,
    load_prediction_record,
    validate_config,
)
from src.datasets.loader.standard_sleep_data import DEFAULT_CHANNELS


def make_cohort(root):
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root / "physionet2018.sqlite")
    conn.executescript("""
        CREATE TABLE Cohort (cohort_id TEXT, scoring_rule TEXT);
        INSERT INTO Cohort VALUES ('physionet2018','AASM');
        CREATE TABLE Participant (participant_id TEXT PRIMARY KEY, record_num INTEGER);
        CREATE TABLE Record (
            record_id TEXT PRIMARY KEY, participant_id TEXT, signal_file_relpath TEXT,
            stage_file_relpath TEXT, scorer TEXT, trim_policy TEXT, trim_start_clock TEXT,
            trim_duration_sec REAL, usage TEXT
        );
    """)
    records = [
        ("a-unlabeled", "unlabeled", False),
        ("r0a", "p0", True),
        ("r0b", "p0", True),
        *[(f"r{i}", f"p{i}", True) for i in range(1, 5)],
    ]
    for name, pid, labeled in records:
        add_record(root, conn, name, pid, labeled=labeled)
    conn.close()
    return {
        "cohort_dir": str(root),
        "cohort_id": "physionet2018",
        "limit": None,
        "cv": {"n_folds": 3, "seed": 42, "shuffle_subjects": True},
        "channel_set": list(DEFAULT_CHANNELS),
        "window_sec": 30.0,
        "stride_sec": 30.0,
        "y_spec": "stages(latest)",
        "normalization": "record-wise",
        "exclude_labels": [-1],
    }


def add_record(root, conn, name, pid, *, labeled=True):
    write_record(root / f"{name}.npz", labeled=labeled, participant_id=pid)
    conn.execute("INSERT OR IGNORE INTO Participant VALUES (?,0)", (pid,))
    conn.execute(
        "UPDATE Participant SET record_num=record_num+1 WHERE participant_id=?", (pid,)
    )
    usage = (
        '["self_supervised", "supervised_stage"]' if labeled else '["self_supervised"]'
    )
    conn.execute(
        "INSERT INTO Record VALUES (?,?,?,?,?,?,?,?,?)",
        (
            name,
            pid,
            name + ".edf",
            name + ".xml" if labeled else None,
            "expert",
            "all_interval",
            "00:00:00",
            150.0,
            usage,
        ),
    )
    conn.commit()


@pytest.fixture
def cohort_config(tmp_path):
    return make_cohort(tmp_path / "cohort")


def test_sql_filters_unlabeled_before_limit_and_keeps_subjects_together(cohort_config):
    cohort_config["limit"] = 4
    cohort, cv = load_experiment_cohort(cohort_config)
    try:
        assert len(cohort.loaded) == 4
        assert "unlabeled" not in cv.participant_to_fold
        assert len(cv.participant_to_fold) == 3
        assert sum(pid == "p0" for pid, _ in cohort.loaded) == 2
        for fold_idx in range(cv.n_folds):
            fold = prepare_fold(cohort, cv, cohort_config, fold_idx)
            train, val, test = [
                set(fold.participants[k]) for k in ("train", "val", "test")
            ]
            assert train | val | test == set(cv.participant_to_fold)
            assert not train & val and not train & test and not val & test
            assert sum(len(x) for x in (fold.train, fold.val, fold.test)) == 4 * 4
            x, y = fold.train[0]
            assert x.shape == (1, 6000, 4) and 0 <= int(y) <= 4
    finally:
        cohort.release()


def test_saved_setting_reuses_exact_records_even_when_database_grows(
    cohort_config, tmp_path
):
    cohort_config["limit"] = 4
    cohort, cv = load_experiment_cohort(cohort_config)
    setting = tmp_path / "cv_setting.json"
    save_cv_setting(setting, cohort, cv)
    original_ids = {obj.path.stem for obj in cohort.loaded.values()}
    original_mapping = dict(cv.participant_to_fold)
    cohort.release()
    root = tmp_path / "cohort"
    conn = sqlite3.connect(root / "physionet2018.sqlite")
    add_record(root, conn, "a-new-labeled", "new")
    conn.close()
    restored, restored_cv = load_experiment_cohort(cohort_config, saved_setting=setting)
    try:
        assert {obj.path.stem for obj in restored.loaded.values()} == original_ids
        assert restored_cv.participant_to_fold == original_mapping
    finally:
        restored.release()


def test_missing_saved_record_fails_instead_of_resplitting(cohort_config, tmp_path):
    cohort, cv = load_experiment_cohort(cohort_config)
    setting = tmp_path / "cv_setting.json"
    save_cv_setting(setting, cohort, cv)
    cohort.release()
    rows = json.loads(setting.read_text())["records"]
    (tmp_path / "cohort" / (rows[0]["record_id"] + ".npz")).unlink()
    with pytest.raises(ValueError, match="could not be loaded"):
        load_experiment_cohort(cohort_config, saved_setting=setting)


def test_trainwise_statistics_exclude_validation_and_test(cohort_config):
    cohort_config["normalization"] = "train-wise"
    cohort, cv = load_experiment_cohort(cohort_config)
    try:
        before = prepare_fold(cohort, cv, cohort_config, 0)
        for (pid, _), record in cohort.loaded.items():
            if cv.participant_to_fold[pid] in (0, 1):
                for frame in record.dfs_by_dt_ms.values():
                    frame += 1000000
        after = prepare_fold(cohort, cv, cohort_config, 0)
        torch.testing.assert_close(
            before.normalization_stats["mean"], after.normalization_stats["mean"]
        )
        torch.testing.assert_close(
            before.normalization_stats["std"], after.normalization_stats["std"]
        )
        torch.testing.assert_close(before.train.tensors[0], after.train.tensors[0])
        assert not torch.allclose(before.val.tensors[0], after.val.tensors[0])
        replay = prepare_fold(
            cohort, cv, cohort_config, 0, normalization_stats=before.normalization_stats
        )
        torch.testing.assert_close(after.test.tensors[0], replay.test.tensors[0])
    finally:
        cohort.release()


@pytest.mark.parametrize("mode", ["none", "record-wise", "epoch-wise", "train-wise"])
def test_single_record_inference_matches_training_preprocessing(cohort_config, mode):
    cohort_config["normalization"] = mode
    cohort, cv = load_experiment_cohort(cohort_config)
    try:
        fold = prepare_fold(cohort, cv, cohort_config, 0)
        # testのうち最初の記録の順序はLoaderの記録順。
        obj = next(
            obj
            for (pid, _), obj in cohort.loaded.items()
            if cv.participant_to_fold[pid] == 0
        )
        dataset, timeline, indices = load_prediction_record(
            obj.path, cohort_config, normalization_stats=fold.normalization_stats
        )
        assert timeline.tolist() == [0, -1, 2, 1, 4]
        assert indices.tolist() == [0, 2, 3, 4]
        torch.testing.assert_close(dataset.tensors[0], fold.test.tensors[0][:4])
        torch.testing.assert_close(dataset.tensors[1], fold.test.tensors[1][:4])
    finally:
        cohort.release()


def test_legacy_config_and_invalid_cv_fail_with_actionable_message(cohort_config):
    with pytest.raises(ValueError, match="移行"):
        validate_config({**cohort_config, "paths": []})
    with pytest.raises(ValueError, match="n_folds >= 3"):
        validate_config({**cohort_config, "cv": {"n_folds": 2}})


def test_loader_cv_defaults_can_be_saved_and_restored(cohort_config, tmp_path):
    cohort_config["cv"] = {"n_folds": 3}
    cohort, cv = load_experiment_cohort(cohort_config)
    path = tmp_path / "cv.json"
    save_cv_setting(path, cohort, cv)
    expected = dict(cv.participant_to_fold)
    cohort.release()
    restored, restored_cv = load_experiment_cohort(cohort_config, saved_setting=path)
    try:
        assert restored_cv.participant_to_fold == expected
    finally:
        restored.release()


def test_explicit_auxiliary_channel_is_requested_from_loader(cohort_config):
    cohort_config["channel_set"] = ["Flow"]
    cohort, cv = load_experiment_cohort(cohort_config)
    try:
        fold = prepare_fold(cohort, cv, cohort_config, 0)
        assert fold.test[0][0].shape == (1, 750, 1)
        obj = next(
            obj
            for (pid, _), obj in cohort.loaded.items()
            if cv.participant_to_fold[pid] == 0
        )
        prediction, _, _ = load_prediction_record(obj.path, cohort_config)
        torch.testing.assert_close(prediction.tensors[0], fold.test.tensors[0][:4])
    finally:
        cohort.release()
