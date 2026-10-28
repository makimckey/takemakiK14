"""授業用の30秒エポックを Sleep-Data-Hub Loader から取り出す。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .sleep_loader import load_npz, make_torch_xy_from_loaded

DATA_DIR = Path("/data/ise_data")
DEFAULT_CHANNELS = ("EEG_C3-R", "EEG_C4-R", "EOG_LOC-R", "EMG_L-EMG_R")
EPOCH_SECONDS = 30


@dataclass
class RecordMetadata:
    participant_id: str
    cohort_id: str
    record_id: str
    signal_info: np.ndarray
    sampling_rate: float
    n_epochs: int
    stages: Optional[np.ndarray]


@dataclass
class SleepRecord:
    signals: np.ndarray  # (エポック, データポイント, チャンネル)
    metadata: RecordMetadata

    @property
    def stages(self):
        return self.metadata.stages

    @property
    def channel_names(self):
        return self.metadata.signal_info[:, 0].tolist()

    @property
    def sampling_rate(self):
        return self.metadata.sampling_rate


def read_record_metadata(
    path,
    *,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    sampling_rate=None,
    require_stages=True,
) -> RecordMetadata:
    """信号本体を展開せず、索引の作成に必要な小さい配列だけを読む。"""
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("channels must contain unique channel names")
    with np.load(path, allow_pickle=False) as data:
        info = data["signal_info"]
        names = info[:, 0].tolist()
        missing = [name for name in channels if name not in names]
        if missing:
            raise ValueError(f"{path}: missing channels {missing}; available={names}")
        indices = [names.index(name) for name in channels]
        intervals = data["signal_sampling_interval_ms"][indices]
        if np.any(intervals <= 0) or len(set(intervals.tolist())) != 1:
            raise ValueError(
                f"{path}: selected channels must have the same positive sampling interval"
            )
        fs = 1000.0 / int(intervals[0])
        if sampling_rate is not None and not np.isclose(fs, sampling_rate):
            raise ValueError(
                f"{path}: sampling rate is {fs} Hz, expected {sampling_rate} Hz"
            )
        samples_per_epoch = EPOCH_SECONDS * fs
        if not float(samples_per_epoch).is_integer():
            raise ValueError(
                "A 30-second epoch must contain an integer number of samples"
            )
        lengths = np.diff(data["signal_offsets"])[indices]
        n_epochs = int(min(lengths) // int(samples_per_epoch))
        stages = data["stages"].astype(np.int64) if "stages" in data else None
        if require_stages and (stages is None or not int(data["usage_mask"]) & 2):
            raise ValueError(
                f"{path}: sleep-stage labels are unavailable; select supervised records"
            )
        if require_stages:
            n_epochs = min(n_epochs, len(stages))
        if n_epochs == 0:
            raise ValueError(f"{path}: no complete 30-second epochs")
        return RecordMetadata(
            participant_id=str(data["participant_id"].item()),
            cohort_id=str(data["cohort_id"].item()),
            record_id=str(data["record_id"].item()),
            signal_info=info[indices].copy(),
            sampling_rate=fs,
            n_epochs=n_epochs,
            stages=stages[:n_epochs].copy() if require_stages else None,
        )


def load_sleep_record(
    path,
    *,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    sampling_rate=None,
    require_stages=True,
) -> SleepRecord:
    """名前で選んだ信号をLoaderで読み、授業・モデル共通の (N,T,C) にする。"""
    meta = read_record_metadata(
        path,
        channels=channels,
        sampling_rate=sampling_rate,
        require_stages=require_stages,
    )
    purpose = "supervised_stage" if require_stages else "self_supervised"
    # Flowなども、明示的に名前を指定した場合はLoaderに追加要求する。
    purpose += "".join(f"+signal({name})" for name in channels)
    loaded = load_npz(path, purpose=purpose, allow_pickle=False)
    try:
        xs, ys, _, _ = make_torch_xy_from_loaded(
            loaded,
            x_channel_sets=[list(channels)],
            window_sec=EPOCH_SECONDS,
            stride_sec=EPOCH_SECONDS,
            y_specs=["stages(latest)"] if require_stages else [],
        )
        signals = xs[0][: meta.n_epochs].numpy().transpose(0, 2, 1)
        if require_stages:
            np.testing.assert_array_equal(ys[0][: meta.n_epochs].numpy(), meta.stages)
        return SleepRecord(signals, meta)
    finally:
        loaded.release()


def find_example_file(data_dir=DATA_DIR) -> Path:
    """正解ステージのある記録を選ぶ。ファイル名を授業資料に固定しない。"""
    for path in sorted(Path(data_dir).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "stages" in data and int(data["usage_mask"]) & 3 == 3:
                return path
    raise FileNotFoundError(f"No supervised sleep-stage NPZ found in {data_dir}")
