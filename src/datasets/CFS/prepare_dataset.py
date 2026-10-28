import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

import scipy
import numpy as np
import pandas as pd
from loguru import logger
from pyedflib import EdfReader


EPOCH_PER_SEC = 30
WAKE_EDGE_EPOCHS = 30

TARGET_CHANNELS = ["C3", "C4", "LOC", "ROC"]
TARGET_SAMPLING_RATE = 200
TARGET_SIGNAL_DTYPE = np.float32

LABEL_TO_STAGE = {
    "Wake|0": 0,
    "REM sleep|5": 1,
    "Stage 1 sleep|1": 2,
    "Stage 2 sleep|2": 3,
    "Stage 3 sleep|3": 4,
}
TARGET_LABEL_DTYPE = int

METADATA_CHANNEL_INFO = [
    ["C3-Mave", ""],
    ["C4-Mave", ""],
    ["LOC-Mave", ""],
    ["ROC-Mave", ""],
]


def get_git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()[:7]
        )
    except Exception:
        return "unknown"


def process_edf_files(edf_path, ann_path, participant_id, meta_csv_path, save_path):
    logger.info(f"Processing {edf_path.name} and {ann_path.name}")

    ################################################################################
    # 1. Prepare biological signals (e.g, EEG, EMG, EOG)
    ################################################################################
    with EdfReader(str(edf_path)) as edf_f:
        signal_names = edf_f.getSignalLabels()
        logger.info(f"Available signal channels: {signal_names}")

        # Load original signals
        origin_signals, origin_signal_fs = {}, {}
        for signal_idx, label in enumerate(signal_names):
            origin_signals[label] = edf_f.readSignal(signal_idx)
            origin_signal_fs[label] = edf_f.getSampleFrequency(signal_idx)
            logger.info(
                f"Load {label}: {origin_signals[label].shape}, {origin_signal_fs[label]}Hz"
            )

        # Create mastoid average reference
        assert "M1" in origin_signals.keys()
        assert "M2" in origin_signals.keys()
        mave_signal = (origin_signals["M1"] + origin_signals["M2"]) / 2

        # Create residual signals (channel - mastoid average)
        residual_signals = {}
        for channel in TARGET_CHANNELS:
            residual_name = f"{channel}-Mave"
            residual_signals[residual_name] = origin_signals[channel] - mave_signal
            logger.info(
                f"Created {residual_name}: {residual_signals[residual_name].shape}"
            )

        # Upsample signals to TARGET_SAMPLING RATE
        for channel in TARGET_CHANNELS:
            origin_fs = origin_signal_fs[channel]
            if origin_fs == TARGET_SAMPLING_RATE:
                continue

            channel_name = f"{channel}-Mave"
            origin_signal = residual_signals[channel_name]
            new_signal_len = int(len(origin_signal) * TARGET_SAMPLING_RATE / origin_fs)
            signal = scipy.signal.resample(origin_signal, new_signal_len)
            residual_signals[channel_name] = signal.astype(TARGET_SIGNAL_DTYPE)

        # Accumulate them as a numpy ndarray
        all_signals = []
        for channel in TARGET_CHANNELS:
            channel_name = f"{channel}-Mave"
            all_signals.append(residual_signals[channel_name])
        all_signals = np.stack(all_signals, axis=1)

        # Format signals as (n_epochs, samples_per_epoch, n_channels)
        samples_per_epoch = EPOCH_PER_SEC * TARGET_SAMPLING_RATE
        n_epochs = all_signals.shape[0] // samples_per_epoch
        reshaped_signals = all_signals[: n_epochs * samples_per_epoch].reshape(
            n_epochs, samples_per_epoch, len(TARGET_CHANNELS)
        )
        logger.info(f"Reshaped: {reshaped_signals.shape} {reshaped_signals.dtype}")

    ################################################################################
    # 2. Prepare stage labels (e.g, Wake, N2)
    ################################################################################
    with open(ann_path) as ann_f:
        # Extract Stage Scoring Events
        root = ET.parse(ann_f).getroot()
        events = root.find("ScoredEvents").findall("ScoredEvent")

        stage_events = []
        for e in events:
            if e.find("EventType").text != "Stages|Stages":
                continue

            event_info = {
                "label": e.find("EventConcept").text,
                "duration": float(e.find("Duration").text),
            }
            stage_events.append(event_info)
        stage_event_df = pd.DataFrame(stage_events)

        # Create stage label for each epoch
        stage_labels = []
        for label, duration in zip(stage_event_df.label, stage_event_df.duration):
            logger.debug(f"label: {label}, duration: {duration}")

            num_epochs = int(duration) // EPOCH_PER_SEC
            stage_label = LABEL_TO_STAGE.get(label, -1)

            for _ in range(num_epochs):
                stage_labels.append(stage_label)

        stage_labels = np.array(stage_labels).astype(TARGET_LABEL_DTYPE)

    ################################################################################
    # 3. Adjust signal & label information
    ################################################################################
    # Remove annotations that are not in the signal
    # Use the number of epochs from the reshaped signal instead of original data records
    labels = np.array(stage_labels[: reshaped_signals.shape[0]])
    logger.info(f"Labels: {len(labels)}")

    # Extract sleep periods while keeping some wake epochs for context
    # The dataset includes ~20 hours of recordings, mostly during wakefulness
    # Keep a buffer of wake epochs around the actual sleep periods
    sleep_epoch_indices = np.where(labels != 0)[0]  # Non-wake stages
    first_sleep_idx = sleep_epoch_indices[0]
    last_sleep_idx = sleep_epoch_indices[-1]

    # Define extraction window with wake buffer
    start_idx = max(first_sleep_idx - WAKE_EDGE_EPOCHS, 0)
    end_idx = min(last_sleep_idx + WAKE_EDGE_EPOCHS, len(labels) - 1)
    logger.info(f"Extracting sleeping epochs from {start_idx} to {end_idx}")

    # Extract the selected portion
    selected_signals = reshaped_signals[start_idx:end_idx]
    selected_labels = labels[start_idx:end_idx]
    logger.info(f"Selected signals: {selected_signals.shape} {selected_signals.dtype}")
    logger.info(f"Selected labels: {selected_labels.shape} {selected_labels.dtype}")

    # Apply consistent data types
    final_signals = selected_signals.astype(TARGET_SIGNAL_DTYPE)
    final_labels = selected_labels.astype(TARGET_LABEL_DTYPE)
    logger.info(f"Final signals shape: {final_signals.shape} {final_signals.dtype}")
    logger.info(f"Final labels shape: {final_labels.shape} {final_labels.dtype}")

    ################################################################################
    # 4. Prepare meta information
    ################################################################################
    with EdfReader(str(edf_path)) as edf_f:
        start_time = str(edf_f.getStartdatetime())

    meta_csv_df = pd.read_csv(meta_csv_path)
    participant_record = meta_csv_df[meta_csv_df["nsrrid"] == participant_id]
    participant_age = participant_record["age"].values[0]
    participant_ahi = participant_record["ahi"].values[0]

    # NOTE:
    # In the CFS dataset, male is coded as 1, whereas in the laboratory’s data format, male is coded as 0.
    # Therefore, in the code below, the format is adjusted to match the laboratory’s standard.
    # ref: https://sleepdata.org/datasets/cfs/variables/sex
    participant_sex = participant_record["sex"].values[0]
    participant_sex = 0 if participant_sex == 1 else 1

    # Save preprocessed data as compressed npz
    npz_content = {
        "signal": final_signals,
        "signal_info": METADATA_CHANNEL_INFO,
        "stages": final_labels,
        "participant_id": participant_id,
        "age": participant_age,
        "ahi": participant_ahi,
        "sex": participant_sex,
        "datasets": "cfs",
        "scoring_rules": "AASM",
        "git_hash": get_git_hash(),
        "original_filename": [edf_path.name, ann_path.name],
        "start_time": start_time,
    }
    np.savez_compressed(save_path, **npz_content)
    logger.info(f"Saved to {save_path}")


@logger.catch
def main(args):
    # Setup logger
    logger.remove()
    logger.add(sys.stderr)
    logger.add(args.output_dir / f"prepare_dataset@{datetime.now().isoformat()}.log")
    logger.info("Starting the dataset preparation process")

    # Select signal & label files
    edfs = list(args.src_dir.glob("*.edf"))
    anns = list(args.src_dir.glob("*.xml"))
    logger.info(f"Found {len(edfs)} EDF files in {args.src_dir}")
    logger.info(f"Found {len(anns)} annotation files in {args.src_dir}")

    # Match signal & label files
    def get_nsrrid_from_edf_path(file):
        return file.stem.split("-")[-1]

    def get_nsrrid_from_ann_path(file):
        return file.stem.split("-")[-2]

    edf_ids = set(get_nsrrid_from_edf_path(f) for f in edfs)
    ann_ids = set(get_nsrrid_from_ann_path(f) for f in anns)
    common_ids = edf_ids & ann_ids
    logger.info(f"{len(common_ids)} files have both EDF and annotation files")

    # Do preprocessing for each patient data
    for nsrrid in sorted(common_ids):
        matchedf_edfs = [edf for edf in edfs if nsrrid in edf.stem]
        matchedf_anns = [ann for ann in anns if nsrrid in ann.stem]
        edf_path = matchedf_edfs[0]
        ann_path = matchedf_anns[0]

        participant_id = int(nsrrid)
        savepath = args.output_dir / f"{participant_id}.npz"

        try:
            process_edf_files(
                edf_path,
                ann_path,
                participant_id,
                args.meta_csv,
                savepath,
            )
        except Exception as e:
            logger.exception(f"Failed to process nsrrid {nsrrid}: {e}")
            continue

    logger.info("Dataset preparation has been completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--src-dir",
        type=Path,
        default="./edf",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default="./npz",
    )
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default="./edf/cfs-visit5-dataset-0.7.0.csv",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)
    main(args)
