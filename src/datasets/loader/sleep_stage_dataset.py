"""Sleep-Data-Hub標準形式を既存モデルの (N,T,C) 入力につなぐ。"""

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .standard_sleep_data import (
    DEFAULT_CHANNELS,
    load_sleep_record,
    read_record_metadata,
)


def _excluded_classes(transform):
    if transform is None:
        return []
    if hasattr(transform, "exclude_classes"):
        return list(transform.exclude_classes)
    return [
        label
        for child in getattr(transform, "transforms", [])
        for label in _excluded_classes(child)
    ]


class SleepStageDataset(Dataset):
    def __init__(
        self,
        npz_files,
        seq_length=1,
        target_idx=-1,
        transform=None,
        channels=DEFAULT_CHANNELS,
        sampling_rate=200,
    ):
        if not npz_files:
            raise ValueError("npz_files must not be empty")
        if seq_length < 1 or not -seq_length <= target_idx < seq_length:
            raise ValueError("Invalid seq_length or target_idx")
        self.npz_files = list(npz_files)
        self.seq_length = seq_length
        self.target_idx = target_idx
        self.transform = transform
        self.channels = list(channels)
        self.sampling_rate = sampling_rate
        self._cache = None
        self.metadata = []
        self.labels = []
        self.epoch_numbers = []
        self.epochs = []
        self.record_sample_indices = []
        excluded = _excluded_classes(transform)

        # 索引の作成では信号本体を読まない。全コホートの一括展開を避ける。
        for file_idx, path in enumerate(self.npz_files):
            meta = read_record_metadata(
                path, channels=channels, sampling_rate=sampling_rate
            )
            keep = np.flatnonzero(~np.isin(meta.stages, excluded))
            self.metadata.append(meta)
            self.labels.append(meta.stages[keep])
            self.epoch_numbers.append(keep)
            first_sample = len(self.epochs)
            for start in range(len(keep) - seq_length + 1):
                # Unknownなどの除外区間をまたいだ連続エポックを作らない。
                if seq_length == 1 or np.all(
                    np.diff(keep[start : start + seq_length]) == 1
                ):
                    self.epochs.append((file_idx, start))
            self.record_sample_indices.append(range(first_sample, len(self.epochs)))

        self.signal_info = self.metadata[0].signal_info
        self.participant_ids = [m.participant_id for m in self.metadata]

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        file_idx, epoch_idx = self.epochs[idx]
        if self._cache is None or self._cache[0] != file_idx:
            # 各ワーカーは直近の1記録だけ保持する。
            self._cache = None
            record = load_sleep_record(
                self.npz_files[file_idx],
                channels=self.channels,
                sampling_rate=self.sampling_rate,
            )
            signal, label = record.signals, record.stages
            if self.transform:
                signal, label = self.transform(signal, label)
            if len(signal) != len(self.labels[file_idx]) or not np.array_equal(
                label, self.labels[file_idx]
            ):
                raise ValueError(
                    "Transforms must preserve epoch order except for ExcludeLabels"
                )
            self._cache = (file_idx, signal, label)
        _, signal, label = self._cache
        window = signal[epoch_idx : epoch_idx + self.seq_length]
        target = label[epoch_idx : epoch_idx + self.seq_length][self.target_idx]
        return torch.from_numpy(np.ascontiguousarray(window)), torch.tensor(
            target, dtype=torch.long
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state


class RecordShuffleSampler(Sampler):
    """記録順と記録内エポックをシャッフルし、同じNPZの再展開を抑える。"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for file_idx in torch.randperm(
            len(self.dataset.record_sample_indices)
        ).tolist():
            indices = self.dataset.record_sample_indices[file_idx]
            for offset in torch.randperm(len(indices)).tolist():
                yield indices[offset]
