"""標準形式の信号名・時刻・ラベルの対応を検証する。実データは使わない。"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.datasets.loader import sleep_stage_dataset as dataset_module
from src.datasets.loader.sleep_stage_dataset import (
    RecordShuffleSampler,
    SleepStageDataset,
)
from src.datasets.loader.standard_sleep_data import (
    DEFAULT_CHANNELS,
    find_example_file,
    load_sleep_record,
    read_record_metadata,
)
from src.datasets.loader.transforms import Compose, ExcludeLabels, ZScoreNormalize
from src.models.sleep_cam import SleepCAM


def write_record(
    path: Path, stages=(0, -1, 2, 1, 4), *, labeled=True, tail=0, participant_id=None
):
    # 保存順を授業の選択順と変え、異なる周波数のFlowも混ぜる。
    names = ["EMG_L-EMG_R", "EEG_C4-R", "Flow", "EOG_LOC-R", "EEG_C3-R"]
    intervals = [5, 5, 40, 5, 5]
    arrays = [
        np.arange(len(stages) * (30000 // dt) + tail, dtype=np.float32) + i * 100000
        for i, dt in enumerate(intervals)
    ]
    content = dict(
        cohort_id="physionet2018",
        participant_id=participant_id or path.stem,
        record_id=path.stem,
        usage_mask=3 if labeled else 1,
        signal_concat=np.concatenate(arrays),
        signal_offsets=np.r_[0, np.cumsum([len(x) for x in arrays])],
        signal_info=np.array([[name, "test"] for name in names]),
        signal_sampling_interval_ms=np.asarray(intervals, dtype=np.int32),
    )
    if labeled:
        content["stages"] = np.asarray(stages, dtype=np.int32)
    np.savez_compressed(path, **content)
    return arrays


def test_named_channels_epoch_alignment_and_tail(tmp_path):
    path = tmp_path / "record.npz"
    arrays = write_record(path, tail=17)
    record = load_sleep_record(path)
    assert record.channel_names == list(DEFAULT_CHANNELS)
    assert record.signals.shape == (5, 6000, 4)
    assert record.sampling_rate == 200
    np.testing.assert_array_equal(record.stages, [0, -1, 2, 1, 4])
    for selected, original in enumerate([4, 1, 3, 0]):
        np.testing.assert_array_equal(
            record.signals[2, :, selected], arrays[original][12000:18000]
        )


def test_select_single_channel_and_reject_missing_or_mixed_rate(tmp_path):
    path = tmp_path / "record.npz"
    write_record(path)
    record = load_sleep_record(path, channels=["EEG_C4-R"])
    assert record.signals.shape == (5, 6000, 1)
    flow = load_sleep_record(path, channels=["Flow"])
    assert flow.signals.shape == (5, 750, 1)
    assert flow.sampling_rate == 25
    with pytest.raises(ValueError, match="missing channels"):
        read_record_metadata(path, channels=["missing"])
    with pytest.raises(ValueError, match="same positive sampling interval"):
        read_record_metadata(path, channels=["EEG_C3-R", "Flow"])
    with pytest.raises(ValueError, match="expected 100"):
        read_record_metadata(path, sampling_rate=100)


def test_unlabeled_records_are_not_selected_for_supervised_use(tmp_path):
    unlabeled = tmp_path / "a.npz"
    labeled = tmp_path / "b.npz"
    write_record(unlabeled, labeled=False)
    write_record(labeled)
    assert find_example_file(tmp_path) == labeled
    with pytest.raises(ValueError, match="labels are unavailable"):
        load_sleep_record(unlabeled)
    record = load_sleep_record(unlabeled, require_stages=False)
    assert record.stages is None
    assert record.signals.shape == (5, 6000, 4)


def test_exclusion_keeps_signal_label_alignment_and_empty_shape():
    signal = np.arange(5 * 3 * 2).reshape(5, 3, 2)
    labels = np.asarray([0, -1, 2, -1, 4])
    x, y = ExcludeLabels([-1])(signal, labels)
    np.testing.assert_array_equal(x, signal[[0, 2, 4]])
    np.testing.assert_array_equal(y, [0, 2, 4])
    x, y = ExcludeLabels([0, -1, 2, 4])(signal, labels)
    assert x.shape == (0, 3, 2)
    assert y.shape == (0,)


def test_dataset_is_lazy_and_sequences_do_not_cross_unknown(tmp_path, monkeypatch):
    path = tmp_path / "record.npz"
    write_record(path)
    calls = []
    original = dataset_module.load_sleep_record

    def load(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(dataset_module, "load_sleep_record", load)
    dataset = SleepStageDataset(
        [path],
        seq_length=2,
        transform=Compose([ExcludeLabels([-1]), ZScoreNormalize()]),
    )
    assert calls == []
    assert len(dataset) == 2  # [2,3] と [3,4]；[0,2] は連続していない
    x, y = dataset[0]
    assert x.shape == (2, 6000, 4)
    assert int(y) == 1
    _, y2 = dataset[1]
    assert int(y2) == 4
    assert len(calls) == 1
    assert torch.isfinite(x).all()
    assert dataset.__getstate__()["_cache"] is None


def test_record_sampler_visits_every_sample_once_in_record_groups(tmp_path):
    paths = [tmp_path / "a.npz", tmp_path / "b.npz"]
    for path in paths:
        write_record(path)
    dataset = SleepStageDataset(paths, transform=ExcludeLabels([-1]))
    indices = list(RecordShuffleSampler(dataset))
    assert sorted(indices) == list(range(len(dataset)))
    file_indices = [dataset.epochs[index][0] for index in indices]
    assert sum(a != b for a, b in zip(file_indices, file_indices[1:])) == 1


def test_existing_model_accepts_dataset_shape(tmp_path):
    path = tmp_path / "record.npz"
    write_record(path)
    dataset = SleepStageDataset([path], transform=ZScoreNormalize())
    x, _ = dataset[0]
    torch.set_num_threads(1)
    model = SleepCAM(input_channels=4).eval()
    with torch.no_grad():
        output = model(x.unsqueeze(0))
    assert output.shape == (1, 5)
