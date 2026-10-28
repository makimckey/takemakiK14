#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sleep_loader.py

Sleep-Data-Hub unified loader + CLI (single-file distribution).

Modes are auto-detected from the first positional argument:
- If <path> is an .npz file (or a file): NPZ mode (detailed debug output, optional torch windowing/y)
- If <path> is a cohort directory (contains standardized_data/<cohort_id>.sqlite): Cohort mode
  (SELECT ... WHERE -> bulk NPZ load -> minimal listing)

Cohort mode prints (one line per loaded record):
  <record_id>.npz  pid=<participant_id>  rnum=<0..>  <cols...>  usage_mask=<int> have=[...]

Notes
- In cohort mode, participant-local record_num is assigned in the order returned by the SQL query
  (no extra sorting), per your request.
- Purpose tokens separated by '+':
    self_supervised, supervised_stage, supervised_arousal, supervised_apnea, signal(NAME)
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sqlite3
import sys
import warnings
from dataclasses import dataclass
try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd


# =============================================================================
# usage bits / purpose parsing
# =============================================================================

_TASK_TO_BIT = {
    "self_supervised": 0,
    "supervised_stage": 1,
    "supervised_arousal": 2,
    "supervised_apnea": 3,
    # alias
    "supervised_apnea_hypopnea": 3,
}

_BIT_TO_TASK_CANON = {
    0: "self_supervised",
    1: "supervised_stage",
    2: "supervised_arousal",
    3: "supervised_apnea",
}

_SIGNAL_EXTRA_RE = re.compile(r"^signal\((.+)\)$", re.IGNORECASE)
_Y_SPEC_RE = re.compile(r"^(stages|arousal|apnea)\(([^\)]*)\)$", re.IGNORECASE)


def _decode_tasks_from_mask(mask: int) -> List[str]:
    out: List[str] = []
    for bit, canon in _BIT_TO_TASK_CANON.items():
        if int(mask) & (1 << bit):
            out.append(canon)
    return out


def _is_eeg_eog_emg(name: str) -> bool:
    n = str(name)
    return n.startswith("EEG_") or n.startswith("EOG_") or (n == "EMG_Chin") or n.startswith("EMG_")


def parse_purpose(purpose: str) -> Tuple[Set[str], List[str]]:
    if purpose is None:
        purpose = ""
    p = str(purpose).strip()
    if p == "":
        return {"self_supervised"}, []

    tokens = [t.strip() for t in p.split("+") if t.strip()]
    tasks: Set[str] = set()
    extra: List[str] = []

    for t in tokens:
        m = _SIGNAL_EXTRA_RE.match(t)
        if m:
            extra.append(m.group(1).strip())
            continue

        if t not in _TASK_TO_BIT:
            raise ValueError(
                f"Unknown purpose token: {t!r}. Allowed tasks={sorted(_TASK_TO_BIT.keys())} and signal(NAME)."
            )
        if t == "supervised_apnea_hypopnea":
            t = "supervised_apnea"
        tasks.add(t)

    # supervised implies self_supervised
    if any(x.startswith("supervised_") for x in tasks):
        tasks.add("self_supervised")

    return tasks, extra


def _require_mask_for_tasks(tasks: Set[str]) -> int:
    m = 0
    for t in tasks:
        m |= (1 << _TASK_TO_BIT[t])
    return int(m)


# =============================================================================
# Single-record NPZ loader
# =============================================================================


@dataclass
class LoadedNPZ:
    path: Path
    dfs_by_dt_ms: Dict[int, pd.DataFrame]          # key: dt_ms
    sampling_interval_ms: Dict[str, int]           # per-signal dt_ms
    usage_mask: int
    signal_info: np.ndarray
    stages: Optional[pd.Series]
    requested_purpose: str
    requested_tasks: List[str]
    requested_extra_signals: List[str]

    def release(self) -> None:
        try:
            self.dfs_by_dt_ms.clear()
        except Exception:
            pass
        try:
            self.sampling_interval_ms.clear()
        except Exception:
            pass
        self.stages = None
        self.signal_info = np.empty((0, 0))
        gc.collect()


def release_loaded(loaded: LoadedNPZ) -> None:
    loaded.release()


def _read_required(z: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in z.files:
        raise KeyError(f"NPZ missing required key: {key}")
    return z[key]


def load_npz(npz_path: Union[str, Path], *, purpose: str = "self_supervised", allow_pickle: bool = True) -> LoadedNPZ:
    p = Path(npz_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"NPZ not found: {p}")

    tasks, extra_signals = parse_purpose(purpose)

    with np.load(str(p), allow_pickle=allow_pickle) as z:
        signal_concat = np.asarray(_read_required(z, "signal_concat"), dtype=np.float32).reshape(-1)
        signal_offsets = np.asarray(_read_required(z, "signal_offsets"), dtype=np.int64).reshape(-1)
        signal_info = np.asarray(_read_required(z, "signal_info"))
        dt_ms_arr = np.asarray(_read_required(z, "signal_sampling_interval_ms"), dtype=np.int32).reshape(-1)
        usage_mask = int(np.asarray(_read_required(z, "usage_mask")).reshape(()))

        req_mask = _require_mask_for_tasks(tasks)
        if (usage_mask & req_mask) != req_mask:
            raise ValueError(
                "NPZ does not satisfy requested purpose.\n"
                f"  usage_mask={usage_mask} have={_decode_tasks_from_mask(usage_mask)}\n"
                f"  requested_tasks={sorted(tasks)} required_mask={req_mask}"
            )

        names_all: List[str] = [str(x) for x in signal_info[:, 0].tolist()]
        name_to_idx = {n: i for i, n in enumerate(names_all)}

        want: List[str] = []
        if "self_supervised" in tasks:
            want.extend([n for n in names_all if _is_eeg_eog_emg(n)])
        if "supervised_arousal" in tasks:
            want.append("Arousal")
        if "supervised_apnea" in tasks:
            want.append("ApneaHypopnea")
        for s in extra_signals:
            if s and s not in want:
                want.append(s)

        missing = [n for n in want if n not in name_to_idx]
        if missing:
            raise KeyError(f"Requested signals not found in NPZ: {missing}. Available(head)={names_all[:25]}")

        arrays_by_dt: Dict[int, Dict[str, np.ndarray]] = {}
        dt_map: Dict[str, int] = {}

        for n in want:
            i = name_to_idx[n]
            a = int(signal_offsets[i]); b = int(signal_offsets[i + 1])
            x = signal_concat[a:b]
            dt = int(dt_ms_arr[i])
            dt_map[n] = dt
            arrays_by_dt.setdefault(dt, {})[n] = x

        dfs_by_dt: Dict[int, pd.DataFrame] = {}
        for dt, ch_map in sorted(arrays_by_dt.items(), key=lambda kv: kv[0]):
            lengths = {k: int(v.shape[0]) for k, v in ch_map.items()}
            max_len = max(lengths.values()) if lengths else 0
            data = {}
            for name, x in ch_map.items():
                if len(x) == max_len:
                    data[name] = x.astype(np.float32, copy=False)
                else:
                    tmp = np.full((max_len,), np.nan, dtype=np.float32)
                    tmp[: len(x)] = x.astype(np.float32, copy=False)
                    data[name] = tmp
            dfs_by_dt[dt] = pd.DataFrame(data)

        stages_ser: Optional[pd.Series] = None
        if "supervised_stage" in tasks:
            if "stages" not in z.files:
                raise KeyError("Requested supervised_stage but NPZ key 'stages' is missing.")
            st = np.asarray(z["stages"], dtype=np.int32).reshape(-1)
            stages_ser = pd.Series(st, name="stages")

    del signal_concat, signal_offsets, dt_ms_arr
    gc.collect()

    return LoadedNPZ(
        path=p,
        dfs_by_dt_ms=dfs_by_dt,
        sampling_interval_ms=dt_map,
        usage_mask=usage_mask,
        signal_info=signal_info,
        stages=stages_ser,
        requested_purpose=str(purpose),
        requested_tasks=sorted(tasks),
        requested_extra_signals=list(extra_signals),
    )


# =============================================================================
# Cohort-level selection + bulk load
# =============================================================================


@dataclass
class CohortLoaded:
    cohort_id: str
    sqlite_path: Path
    standardized_dir: Path
    hits_df: pd.DataFrame
    loaded: Dict[Tuple[str, int], LoadedNPZ]  # (participant_id, record_num) -> LoadedNPZ
    scoring_rule: str = "Unknown"
    biosignal_signature: Tuple[str, ...] = ()
    common_signal_names: Tuple[str, ...] = ()
    common_signal_names_by_type: Dict[str, Tuple[str, ...]] = None

    def release(self) -> None:
        for v in self.loaded.values():
            try:
                v.release()
            except Exception:
                pass
        self.loaded.clear()
        gc.collect()


@dataclass
class MultiCohortLoaded:
    cohort_ids: List[str]
    sqlite_paths: Dict[str, Path]
    standardized_dirs: Dict[str, Path]
    hits_df: pd.DataFrame
    loaded: Dict[Tuple[str, int], LoadedNPZ]  # (namespaced_participant_id, record_num) -> LoadedNPZ
    record_key_to_record_id: Dict[Tuple[str, int], str]
    scoring_rules: Dict[str, str]
    biosignal_signatures: Dict[str, Tuple[str, ...]]
    common_signal_names: Tuple[str, ...] = ()
    common_signal_names_by_type: Dict[str, Tuple[str, ...]] = None

    @property
    def cohort_id(self) -> str:
        return "+".join(self.cohort_ids)

    def release(self) -> None:
        for v in self.loaded.values():
            try:
                v.release()
            except Exception:
                pass
        self.loaded.clear()
        self.record_key_to_record_id.clear()
        gc.collect()


def _normalize_scoring_rule_family(scoring_rule: str) -> str:
    s = str(scoring_rule or "").strip().lower()
    if s == "":
        return "unknown"
    if "aasm" in s:
        return "AASM"
    if "r&k" in s or "r and k" in s or "rechtschaffen" in s or "kales" in s:
        return "R&K"
    if "not applicable" in s:
        return "Not Applicable"
    return str(scoring_rule or "Unknown").strip() or "Unknown"


def _extract_biosignal_signature_from_loaded(loaded: Dict[Tuple[str, int], LoadedNPZ]) -> Tuple[str, ...]:
    names: Set[str] = set()
    for obj in loaded.values():
        try:
            arr = np.asarray(obj.signal_info)
            if arr.ndim != 2 or arr.shape[1] < 1:
                continue
            for x in arr[:, 0].tolist():
                n = str(x)
                if _is_eeg_eog_emg(n):
                    names.add(n)
        except Exception:
            continue
    return tuple(sorted(names))


def _read_single_value(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> Optional[Any]:
    cur = conn.cursor()
    cur.execute(sql, list(params))
    row = cur.fetchone()
    return None if row is None else row[0]


def _read_cohort_scoring_rule(sqlite_path: Path) -> str:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        v = _read_single_value(conn, "SELECT scoring_rule FROM Cohort LIMIT 1")
    except Exception:
        v = None
    finally:
        conn.close()
    return str(v) if v not in (None, "") else "Unknown"


def _default_mix_rule_config_path() -> Path:
    return Path(__file__).resolve().with_name("cohort_mix_rules.toml")


def _parse_simple_toml_value(raw: str) -> Any:
    raw = raw.strip()

    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False

    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]

    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []

        vals = []
        buf = []
        in_quote = False
        quote_char = ""

        for ch in inner:
            if ch in ("'", '"'):
                if in_quote and ch == quote_char:
                    in_quote = False
                    quote_char = ""
                elif not in_quote:
                    in_quote = True
                    quote_char = ch
                buf.append(ch)
            elif ch == "," and not in_quote:
                vals.append(_parse_simple_toml_value("".join(buf).strip()))
                buf = []
            else:
                buf.append(ch)

        if buf:
            vals.append(_parse_simple_toml_value("".join(buf).strip()))
        return vals

    try:
        return int(raw)
    except Exception:
        pass

    try:
        return float(raw)
    except Exception:
        pass

    return raw


def _parse_simple_toml_fallback(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current: Optional[Dict[str, Any]] = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        if line.startswith("[[") and line.endswith("]]"):
            name = line[2:-2].strip()
            arr = data.setdefault(name, [])
            if not isinstance(arr, list):
                raise RuntimeError(f"Invalid TOML structure for [[{name}]]")
            current = {}
            arr.append(current)
            continue

        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            sec = data.setdefault(name, {})
            if not isinstance(sec, dict):
                raise RuntimeError(f"Invalid TOML structure for [{name}]")
            current = sec
            continue

        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value_obj = _parse_simple_toml_value(value.strip())
            if current is None:
                data[key] = value_obj
            else:
                current[key] = value_obj
            continue

        raise RuntimeError(f"Unsupported TOML line in fallback parser: {raw_line}")

    return data


def _load_mix_rule_config(config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    if config_path is False:
        return {"validation": {"enabled": False}}

    path = _default_mix_rule_config_path() if config_path is None else Path(config_path).expanduser().resolve()
    if not path.exists():
        if config_path is None:
            return {
                "validation": {"enabled": True},
                "shared_signal_requirements": {"EEG": "error", "EOG": "warning", "EMG": "warning"},
            }
        raise FileNotFoundError(f"cohort mix rule config not found: {path}")

    if tomllib is not None:
        with path.open("rb") as f:
            data = tomllib.load(f)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = _parse_simple_toml_fallback(f.read())

    data["__config_path__"] = str(path)
    return data


def _signals_by_type(sig_names: Sequence[str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {"EEG": set(), "EOG": set(), "EMG": set()}
    for name in sig_names:
        s = str(name)
        if s.startswith("EEG_"):
            out["EEG"].add(s)
        elif s.startswith("EOG_"):
            out["EOG"].add(s)
        elif s.startswith("EMG_") or s == "EMG_Chin":
            out["EMG"].add(s)
    return out


def _compute_common_signal_names(children: Sequence[CohortLoaded]) -> Dict[str, Tuple[str, ...]]:
    by_type: Optional[Dict[str, Set[str]]] = None
    for child in children:
        child_map = _signals_by_type(child.biosignal_signature)
        if by_type is None:
            by_type = {k: set(v) for k, v in child_map.items()}
        else:
            for k in ("EEG", "EOG", "EMG"):
                by_type[k] &= set(child_map.get(k, set()))
    if by_type is None:
        by_type = {"EEG": set(), "EOG": set(), "EMG": set()}
    return {k: tuple(sorted(v)) for k, v in by_type.items()}


def _validate_explicit_forbidden_pairs(children: Sequence[CohortLoaded], config: Dict[str, Any]) -> None:
    rules = config.get("forbid_pair") or []
    if not isinstance(rules, list) or not rules:
        return
    loaded_ids = {str(c.cohort_id) for c in children}
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        cohorts = rule.get("cohorts") or []
        if not isinstance(cohorts, list) or len(cohorts) < 2:
            continue
        cohort_set = {str(x) for x in cohorts}
        if cohort_set.issubset(loaded_ids):
            reason = str(rule.get("reason") or "Forbidden cohort combination")
            raise ValueError(f"Forbidden cohort combination detected: {sorted(cohort_set)}. {reason}")


def _validate_child_compatibility(
    children: Sequence[CohortLoaded],
    *,
    purpose: str,
    config_path: Optional[Union[str, Path]],
) -> Dict[str, Tuple[str, ...]]:
    if len(children) <= 1:
        if children:
            return _compute_common_signal_names(children)
        return {"EEG": (), "EOG": (), "EMG": ()}

    config = _load_mix_rule_config(config_path)
    validation = config.get("validation") or {}
    if not bool(validation.get("enabled", True)):
        return _compute_common_signal_names(children)

    _validate_explicit_forbidden_pairs(children, config)

    shared_by_type = _compute_common_signal_names(children)
    severity_map = config.get("shared_signal_requirements") or {"EEG": "error", "EOG": "warning", "EMG": "warning"}
    for sig_type in ("EEG", "EOG", "EMG"):
        shared = tuple(shared_by_type.get(sig_type, ()))
        severity = str(severity_map.get(sig_type, "ignore")).strip().lower()
        if shared or severity in {"ignore", "none", "off", "disabled"}:
            continue
        per_child = []
        for child in children:
            child_names = _signals_by_type(child.biosignal_signature).get(sig_type, set())
            per_child.append(f"{child.cohort_id}: [{', '.join(sorted(child_names)) or '<none>'}]")
        msg = (
            f"Incompatible cohort mix detected: no shared {sig_type} channels across cohorts. "
            + " | ".join(per_child)
        )
        if severity == "error":
            raise ValueError(msg)
        if severity == "warning":
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    check_stage_rule = bool(validation.get("check_scoring_rule_for_supervised_stage", True))
    tasks, _ = parse_purpose(purpose)
    if check_stage_rule and "supervised_stage" in tasks:
        fam_to_ids: Dict[str, List[str]] = {}
        for child in children:
            fam = _normalize_scoring_rule_family(child.scoring_rule)
            fam_to_ids.setdefault(fam, []).append(str(child.cohort_id))
        if len(fam_to_ids) > 1:
            parts = [f"{fam}: {ids}" for fam, ids in fam_to_ids.items()]
            raise ValueError(
                "Incompatible cohort mix detected for purpose=supervised_stage: scoring_rule families differ across cohorts. "
                + " | ".join(parts)
            )

    return shared_by_type


def _candidate_cohort_roots() -> Sequence[Path]:
    """Ordered search roots for resolving a cohort by cohort_id (local first)."""
    return [
        Path("/data/sleep_cohorts"),
        Path("/data/Sleep-Data-Hub/cohorts"),
    ]


def _resolve_cohort_input(
    cohort: Union[str, Path],
    *,
    cohort_id: Optional[str] = None,
) -> Tuple[Path, Optional[str]]:
    """
    Resolve either:
      - an explicit path, or
      - a cohort_id-like string

    Returns:
      (resolved_cohort_dir, inferred_or_given_cohort_id)
    """
    raw = Path(cohort).expanduser()

    if raw.exists():
        return raw.resolve(), (str(cohort_id) if cohort_id else None)

    cid = str(cohort_id) if cohort_id else str(cohort)
    tried: List[str] = []
    for root in _candidate_cohort_roots():
        cand = (root / cid).expanduser().resolve()
        tried.append(str(cand))
        if cand.exists():
            return cand, cid

    msg = (
        "Could not resolve cohort from input.\n"
        f"  input={cohort}\n"
        f"  interpreted_cohort_id={cid}\n"
        "  tried candidate directories:\n"
        + "\n".join(f"    - {p}" for p in tried)
    )
    raise FileNotFoundError(msg)


def _resolve_standardized_paths(
    cohort_dir: Union[str, Path],
    *,
    cohort_id: Optional[str] = None,
) -> Tuple[str, Path, Path]:
    cd = Path(cohort_dir).expanduser().resolve()
    if not cd.exists():
        raise FileNotFoundError(f"cohort_dir not found: {cd}")

    cid = str(cohort_id) if cohort_id else cd.name

    std_candidate_1 = cd / "standardized_data"
    sqlite_candidate_1 = std_candidate_1 / f"{cid}.sqlite"

    std_candidate_2 = cd
    inferred_cid_2 = str(cohort_id) if cohort_id else cd.parent.name
    sqlite_candidate_2 = std_candidate_2 / f"{inferred_cid_2}.sqlite"

    std_candidate_3 = cd
    sqlite_candidate_3 = std_candidate_3 / f"{cid}.sqlite"

    if std_candidate_1.exists() and sqlite_candidate_1.exists():
        return cid, std_candidate_1, sqlite_candidate_1

    if cd.name == "standardized_data" and sqlite_candidate_2.exists():
        return inferred_cid_2, std_candidate_2, sqlite_candidate_2

    if sqlite_candidate_3.exists():
        return cid, std_candidate_3, sqlite_candidate_3

    msg = (
        "Could not resolve standardized data layout.\n"
        f"  cohort_dir={cd}\n"
        f"  tried:\n"
        f"    1) {sqlite_candidate_1}\n"
        f"    2) {sqlite_candidate_2}\n"
        f"    3) {sqlite_candidate_3}\n"
        "Expected one of the following layouts:\n"
        "  - <cohort_dir>/standardized_data/<cohort_id>.sqlite\n"
        "  - <cohort_dir> itself is standardized_data\n"
        "  - <cohort_dir>/<cohort_id>.sqlite  (workstation local cache layout)"
    )
    raise FileNotFoundError(msg)


def _build_record_select_sql(*, where: str, order_by: str, limit: Optional[int] = None) -> str:
    sql = f"""
    SELECT
      r.record_id AS record_id,
      r.participant_id AS participant_id,
      p.record_num AS participant_record_count,
      r.signal_file_relpath AS signal_file_relpath,
      r.stage_file_relpath AS stage_file_relpath,
      r.scorer AS scorer,
      r.trim_policy AS trim_policy,
      r.trim_start_clock AS trim_start_clock,
      r.trim_duration_sec AS trim_duration_sec
    FROM Record r
    JOIN Participant p USING(participant_id)
    WHERE {where}
    ORDER BY {order_by}
    """.strip()
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"
    return sql


def _run_record_select_query(
    sqlite_path: Union[str, Path],
    *,
    where: str,
    params: Sequence[Any],
    order_by: str,
    limit: Optional[int],
) -> pd.DataFrame:
    sql = _build_record_select_sql(where=where, order_by=order_by, limit=limit)
    conn = sqlite3.connect(str(sqlite_path))
    try:
        return pd.read_sql_query(sql, conn, params=list(params))
    finally:
        conn.close()


def _annotate_hits_df_with_context(hits_df: pd.DataFrame, *, cohort_id: str, sqlite_path: Path, standardized_dir: Path) -> pd.DataFrame:
    df = hits_df.copy()
    df.insert(0, "cohort_id", str(cohort_id))
    df.insert(1, "sqlite_path", str(sqlite_path))
    df.insert(2, "standardized_dir", str(standardized_dir))
    return df


def select_records(
    cohort_dir: Union[str, Path],
    *,
    where: str = "1=1",
    params: Sequence[Any] = (),
    cohort_id: Optional[str] = None,
    limit: Optional[int] = None,
    order_by: str = "r.record_id ASC",
) -> pd.DataFrame:
    resolved_cohort_dir, resolved_input_cid = _resolve_cohort_input(
        cohort_dir,
        cohort_id=cohort_id,
    )
    cid, std, sqlite_path = _resolve_standardized_paths(
        resolved_cohort_dir,
        cohort_id=resolved_input_cid,
    )

    sql = f"""
    SELECT
      r.record_id AS record_id,
      r.participant_id AS participant_id,
      p.record_num AS participant_record_count,
      r.signal_file_relpath AS signal_file_relpath,
      r.stage_file_relpath AS stage_file_relpath,
      r.scorer AS scorer,
      r.trim_policy AS trim_policy,
      r.trim_start_clock AS trim_start_clock,
      r.trim_duration_sec AS trim_duration_sec
    FROM Record r
    JOIN Participant p USING(participant_id)
    WHERE {where}
    ORDER BY {order_by}
    """.strip()

    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"

    conn = sqlite3.connect(str(sqlite_path))
    try:
        df = pd.read_sql_query(sql, conn, params=list(params))
    finally:
        conn.close()

    df.insert(0, "cohort_id", cid)
    df.insert(1, "sqlite_path", str(sqlite_path))
    df.insert(2, "standardized_dir", str(std))
    return df

def _assign_record_num_in_selection_order(hits_df: pd.DataFrame) -> pd.DataFrame:
    """Assign record_num=0.. per participant in the order returned by SQL (no extra sorting)."""
    if hits_df.empty:
        hits_df["record_num"] = pd.Series([], dtype=np.int64)
        return hits_df

    df = hits_df.copy().reset_index(drop=True)
    # cumcount preserves the current row order within each participant group
    df["record_num"] = df.groupby("participant_id", sort=False).cumcount().astype(np.int64)
    return df


def _npz_path_for_record(standardized_dir: Path, record_id: str) -> Path:
    return standardized_dir / f"{record_id}.npz"


def _annotate_hits_df_with_npz_status(hits_df: pd.DataFrame, *, standardized_dir: Path) -> pd.DataFrame:
    df = hits_df.copy()
    df["npz_path"] = df["record_id"].map(lambda rid: str(_npz_path_for_record(standardized_dir, str(rid))))
    df["npz_exists"] = df["npz_path"].map(lambda p: Path(p).exists())
    return df


def _load_records_from_hits(
    hits_df: pd.DataFrame,
    *,
    standardized_dir: Path,
    purpose: str,
    allow_pickle: bool,
) -> Tuple[Dict[Tuple[str, int], LoadedNPZ], List[str], List[Tuple[str, str]]]:
    loaded: Dict[Tuple[str, int], LoadedNPZ] = {}
    missing_npz: List[str] = []
    failed: List[Tuple[str, str]] = []

    for _, row in hits_df.iterrows():
        rid = str(row["record_id"])
        pid = str(row["participant_id"])
        rnum = int(row["record_num"])
        npz_path = _npz_path_for_record(standardized_dir, rid)
        if not npz_path.exists():
            missing_npz.append(str(npz_path))
            continue
        try:
            obj = load_npz(npz_path, purpose=purpose, allow_pickle=allow_pickle)
            loaded[(pid, rnum)] = obj
        except Exception as e:
            failed.append((rid, str(e)))

    return loaded, missing_npz, failed


def _print_load_cohort_summary(
    *,
    cohort_id: str,
    sqlite_path: Path,
    standardized_dir: Path,
    hits_count: int,
    loaded_count: int,
    missing_npz: Sequence[str],
    failed: Sequence[Tuple[str, str]],
) -> None:
    print(f"[sleep_loader] cohort_id={cohort_id}")
    print(f"[sleep_loader] sqlite={sqlite_path}")
    print(f"[sleep_loader] standardized_dir={standardized_dir}")
    print(f"[sleep_loader] hits={hits_count} loaded={loaded_count} missing_npz={len(missing_npz)} failed={len(failed)}")
    if missing_npz:
        print(f"[sleep_loader] missing_npz(head): {list(missing_npz)[:5]}")
    if failed:
        print(f"[sleep_loader] failed(head): {list(failed)[:3]}")


def _namespace_value_for_multi(cohort_id: str, value: Any) -> str:
    return f"{cohort_id}::{value}"


def _namespace_hits_df_for_multi(hits_df: pd.DataFrame, *, cohort_id: str) -> pd.DataFrame:
    df = hits_df.copy()
    if "participant_id" in df.columns:
        df["participant_id"] = df["participant_id"].map(lambda x: _namespace_value_for_multi(cohort_id, x))
    if "record_id" in df.columns:
        df["source_record_id"] = df["record_id"].astype(str)
        df["record_id"] = df["record_id"].map(lambda x: _namespace_value_for_multi(cohort_id, x))
    return df


def load_cohorts(
    cohort_dirs: Sequence[Union[str, Path]],
    *,
    where: str = "1=1",
    params: Sequence[Any] = (),
    purpose: str = "self_supervised",
    cohort_ids: Optional[Sequence[Optional[str]]] = None,
    limit: Optional[int] = None,
    order_by: str = "r.record_id ASC",
    allow_pickle: bool = True,
    verbose: bool = True,
    compatibility_config: Optional[Union[str, Path]] = None,
) -> MultiCohortLoaded:
    if not cohort_dirs:
        raise ValueError("cohort_dirs must contain at least one cohort")

    if cohort_ids is not None and len(cohort_ids) != len(cohort_dirs):
        raise ValueError(
            f"cohort_ids length mismatch: len(cohort_dirs)={len(cohort_dirs)} len(cohort_ids)={len(cohort_ids)}"
        )

    loaded_children: List[CohortLoaded] = []
    for i, cohort_dir in enumerate(cohort_dirs):
        child = load_cohort(
            cohort_dir,
            where=where,
            params=params,
            purpose=purpose,
            cohort_id=None if cohort_ids is None else cohort_ids[i],
            limit=limit,
            order_by=order_by,
            allow_pickle=allow_pickle,
            verbose=verbose,
        )
        loaded_children.append(child)

    seen_cohort_ids: Set[str] = set()
    for child in loaded_children:
        if child.cohort_id in seen_cohort_ids:
            raise ValueError(
                f"Duplicate cohort_id detected while merging cohorts: {child.cohort_id!r}. "
                "Please rename one cohort or load them separately."
            )
        seen_cohort_ids.add(child.cohort_id)

    common_signal_names_by_type = _validate_child_compatibility(
        loaded_children,
        purpose=purpose,
        config_path=compatibility_config,
    )
    common_signal_names = tuple(sorted({x for vals in common_signal_names_by_type.values() for x in vals}))

    merged_hits_parts: List[pd.DataFrame] = []
    merged_loaded: Dict[Tuple[str, int], LoadedNPZ] = {}
    record_key_to_record_id: Dict[Tuple[str, int], str] = {}
    sqlite_paths: Dict[str, Path] = {}
    standardized_dirs: Dict[str, Path] = {}
    scoring_rules: Dict[str, str] = {}
    biosignal_signatures: Dict[str, Tuple[str, ...]] = {}

    for child in loaded_children:
        cid = str(child.cohort_id)
        sqlite_paths[cid] = child.sqlite_path
        standardized_dirs[cid] = child.standardized_dir
        scoring_rules[cid] = child.scoring_rule
        biosignal_signatures[cid] = tuple(child.biosignal_signature)
        merged_hits_parts.append(_namespace_hits_df_for_multi(child.hits_df, cohort_id=cid))

        for (pid, rnum), obj in child.loaded.items():
            namespaced_pid = _namespace_value_for_multi(cid, pid)
            new_key = (namespaced_pid, int(rnum))
            if new_key in merged_loaded:
                raise ValueError(
                    "Duplicate loaded key detected after namespacing. "
                    f"cohort_id={cid} participant_id={pid} record_num={rnum}"
                )
            merged_loaded[new_key] = obj
            record_key_to_record_id[new_key] = _namespace_value_for_multi(cid, obj.path.stem)

    merged_hits = pd.concat(merged_hits_parts, axis=0, ignore_index=True) if merged_hits_parts else pd.DataFrame()
    return MultiCohortLoaded(
        cohort_ids=[str(child.cohort_id) for child in loaded_children],
        sqlite_paths=sqlite_paths,
        standardized_dirs=standardized_dirs,
        hits_df=merged_hits,
        loaded=merged_loaded,
        record_key_to_record_id=record_key_to_record_id,
        scoring_rules=scoring_rules,
        biosignal_signatures=biosignal_signatures,
    )


def load_cohort(
    cohort_dir: Union[str, Path],
    *,
    where: str = "1=1",
    params: Sequence[Any] = (),
    purpose: str = "self_supervised",
    cohort_id: Optional[str] = None,
    limit: Optional[int] = None,
    order_by: str = "r.record_id ASC",
    allow_pickle: bool = True,
    verbose: bool = True,
    compatibility_config: Optional[Union[str, Path]] = None,
) -> CohortLoaded:
    # compatibility_config is accepted for API symmetry with load_cohorts;
    # single-cohort loading does not require cross-cohort validation.
    _ = compatibility_config

    resolved_cohort_dir, resolved_input_cid = _resolve_cohort_input(
        cohort_dir,
        cohort_id=cohort_id,
    )
    cid, std, sqlite_path = _resolve_standardized_paths(
        resolved_cohort_dir,
        cohort_id=resolved_input_cid,
    )

    hits = select_records(
        resolved_cohort_dir,
        where=where,
        params=params,
        cohort_id=cid,
        limit=limit,
        order_by=order_by,
    )
    hits = _assign_record_num_in_selection_order(hits)

    loaded, missing_npz, failed = _load_records_from_hits(
        hits,
        standardized_dir=std,
        purpose=purpose,
        allow_pickle=allow_pickle,
    )
    hits = _annotate_hits_df_with_npz_status(hits, standardized_dir=std)

    if verbose:
        _print_load_cohort_summary(
            cohort_id=cid,
            sqlite_path=sqlite_path,
            standardized_dir=std,
            hits_count=len(hits),
            loaded_count=len(loaded),
            missing_npz=missing_npz,
            failed=failed,
        )

    scoring_rule = _read_cohort_scoring_rule(sqlite_path)
    biosignal_signature = _extract_biosignal_signature_from_loaded(loaded)

    return CohortLoaded(
        cohort_id=cid,
        sqlite_path=sqlite_path,
        standardized_dir=std,
        hits_df=hits,
        loaded=loaded,
        scoring_rule=scoring_rule,
        biosignal_signature=biosignal_signature,
    )


# =============================================================================
# Optional: build a long DataFrame for a specific dt_ms group and channel set
# =============================================================================


def build_long_df(
    cohort: CohortLoaded,
    *,
    dt_ms: int,
    channels: Sequence[str],
    drop_all_nan_rows: bool = True,
) -> pd.DataFrame:
    chs = [str(c) for c in channels]
    parts: List[pd.DataFrame] = []

    for (pid, rnum), obj in cohort.loaded.items():
        if dt_ms not in obj.dfs_by_dt_ms:
            continue
        df = obj.dfs_by_dt_ms[dt_ms]
        miss = [c for c in chs if c not in df.columns]
        if miss:
            raise KeyError(f"Missing channels in record pid={pid} rnum={rnum}: {miss}")

        sub = df[chs].copy()
        if drop_all_nan_rows:
            sub = sub.dropna(how="all")
        sub.index = pd.MultiIndex.from_product(
            [[pid], [rnum], sub.index.astype(np.int64)],
            names=["participant_id", "record_num", "t"],
        )
        parts.append(sub)

    if not parts:
        return pd.DataFrame(columns=chs)

    return pd.concat(parts, axis=0)


# =============================================================================
# NPZ debug torch helpers (kept from original file)
# =============================================================================


def _split_top_level_commas(s: str) -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    depth = 0
    in_sq = False
    in_dq = False
    esc = False
    for ch in s:
        if esc:
            buf.append(ch); esc = False; continue
        if ch == "\\":
            buf.append(ch); esc = True; continue
        if ch == "'" and not in_dq:
            in_sq = not in_sq; buf.append(ch); continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq; buf.append(ch); continue
        if not in_sq and not in_dq:
            if ch in "[({":
                depth += 1
            elif ch in "])}":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                out.append("".join(buf).strip()); buf = []; continue
        buf.append(ch)
    out.append("".join(buf).strip())
    return out


def parse_channel_sets_arg(channel_sets_str: str) -> List[Optional[List[str]]]:
    s = (channel_sets_str or "").strip()
    if not s:
        raise ValueError("channel_sets is empty.")
    parts = _split_top_level_commas(s)
    out: List[Optional[List[str]]] = []
    for i, part in enumerate(parts):
        if part == "":
            out.append(None); continue
        try:
            v = json.loads(part)
        except Exception as e:
            raise ValueError(
                f"Failed to parse channel_sets element #{i+1}: {part!r}. Use JSON list like [\"EEG_C3-R\"]."
            ) from e
        if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
            raise ValueError(f"channel_sets element #{i+1} must be a JSON list of strings, got: {v!r}")
        out.append([str(x) for x in v])
    return out


def parse_y_specs_arg(y_specs_str: str) -> List[str]:
    s = (y_specs_str or "").strip()
    if not s:
        return []
    try:
        v = json.loads(s)
    except Exception as e:
        raise ValueError('y_specs must be JSON list, e.g. \"[\\"stages(latest)\\",\\"arousal(epoch,any)\\"]\"') from e
    if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
        raise ValueError(f"y_specs must be a JSON list of strings, got: {v!r}")
    specs = [str(x).strip() for x in v if str(x).strip()]
    for sp in specs:
        m = _Y_SPEC_RE.match(sp)
        if not m:
            raise ValueError(f"Unknown y spec format: {sp!r}")
        kind = m.group(1).lower()
        args = [a.strip().lower() for a in m.group(2).split(",") if a.strip()]
        mode = args[0] if args else ""
        agg = args[1] if len(args) >= 2 else ""

        if kind == "stages":
            if mode not in ("latest", "center", "vote"):
                raise ValueError(f"stages() mode must be latest|center|vote, got {mode!r} in {sp!r}")
            continue

        if mode not in ("timepoint", "epoch"):
            raise ValueError(f"{kind}() mode must be timepoint|epoch, got {mode!r} in {sp!r}")
        if mode == "epoch":
            if agg == "":
                agg = "any"
            if agg not in ("any", "count", "ratio"):
                raise ValueError(f"{kind}(epoch,agg) agg must be any|count|ratio, got {agg!r} in {sp!r}")

    return specs


def _to_seconds_samples(seconds: float, dt_ms: int) -> int:
    return int(round((float(seconds) * 1000.0) / float(dt_ms)))


def make_torch_windows_from_loaded(
    loaded: LoadedNPZ,
    *,
    channel_sets: Sequence[Optional[Sequence[str]]],
    window_sec: float,
    stride_sec: float,
    dtype: str = "float32",
):
    try:
        import torch  # type: ignore
    except Exception as e:
        raise ImportError("torch is required.") from e

    x_list: List[Optional["torch.Tensor"]] = []
    info_list: List[Optional[dict]] = []

    dt_map = loaded.sampling_interval_ms

    for chs in channel_sets:
        if not chs:
            x_list.append(None); info_list.append(None); continue

        ch_list = [str(c) for c in chs]
        missing = [c for c in ch_list if c not in dt_map]
        if missing:
            raise KeyError(f"Channels not found / not loaded: {missing}")

        dts = {dt_map[c] for c in ch_list}
        if len(dts) != 1:
            raise ValueError(f"Sampling interval mismatch within channel_set={ch_list}: dt_ms={sorted(dts)}")
        dt = int(next(iter(dts)))
        df = loaded.dfs_by_dt_ms[dt]

        win = _to_seconds_samples(window_sec, dt)
        stride = _to_seconds_samples(stride_sec, dt)
        if stride <= 0:
            raise ValueError("stride_samples must be >0")

        arrs = []
        lengths = []
        for c in ch_list:
            x = df[c].to_numpy(dtype=np.float32, copy=False)
            if np.isnan(x).any():
                finite = np.isfinite(x)
                if finite.any():
                    last = int(np.where(finite)[0][-1]) + 1
                    x = x[:last]
            arrs.append(x); lengths.append(len(x))
        L = int(min(lengths)) if lengths else 0

        if L < win:
            tens = torch.empty((0, len(ch_list), win), dtype=getattr(torch, dtype))
            x_list.append(tens)
            info_list.append({"dt_ms": dt, "fs_hz": 1000.0/dt, "window_samples": win, "stride_samples": stride,
                              "n_windows": 0, "channels": ch_list, "signal_length_samples": L})
            continue

        nwin = 1 + (L - win) // stride
        data = np.empty((nwin, len(ch_list), win), dtype=np.float32)
        for i in range(nwin):
            a = i * stride; b = a + win
            for j, x in enumerate(arrs):
                data[i, j, :] = x[a:b]
        tens = torch.from_numpy(data).to(dtype=getattr(torch, dtype))
        x_list.append(tens)
        info_list.append({"dt_ms": dt, "fs_hz": 1000.0/dt, "window_samples": win, "stride_samples": stride,
                          "n_windows": int(nwin), "channels": ch_list, "signal_length_samples": L})

    return x_list, info_list


def _epoch_indices_for_window(start_sec: float, window_sec: float, epoch_sec: float) -> List[int]:
    end_sec = start_sec + window_sec
    k0 = int(start_sec // epoch_sec)
    k1 = int((max(end_sec - 1e-9, 0.0)) // epoch_sec)
    if k1 < k0:
        return []
    return list(range(k0, k1 + 1))


def _stage_label_for_window(stages: np.ndarray, start_sec: float, window_sec: float, *, mode: str, epoch_sec: float) -> int:
    if stages.size == 0:
        return -1
    mode = mode.lower()
    if mode == "latest":
        end_sec = start_sec + window_sec
        k = int((max(end_sec - 1e-9, 0.0)) // epoch_sec)
        return int(stages[k]) if 0 <= k < len(stages) else -1
    if mode == "center":
        center = start_sec + window_sec/2.0
        k = int(center // epoch_sec)
        return int(stages[k]) if 0 <= k < len(stages) else -1
    if mode == "vote":
        ks = _epoch_indices_for_window(start_sec, window_sec, epoch_sec)
        vals = [int(stages[k]) for k in ks if 0 <= k < len(stages)]
        if not vals:
            return -1
        from collections import Counter
        c = Counter(vals)
        return int(sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])
    raise ValueError(mode)


def _binary_epoch_label_from_timepoint_signal(sig: np.ndarray, *, dt_ms: int, epoch_sec: float) -> np.ndarray:
    samples_per_epoch = int(round(epoch_sec*1000.0/dt_ms))
    n_epochs = (len(sig) + samples_per_epoch - 1)//samples_per_epoch
    out = np.zeros((n_epochs,), dtype=np.int64)
    for k in range(n_epochs):
        a = k*samples_per_epoch
        b = min((k+1)*samples_per_epoch, len(sig))
        if b > a and np.nanmax(sig[a:b]) > 0:
            out[k] = 1
    return out


def make_torch_xy_from_loaded(
    loaded: LoadedNPZ,
    *,
    x_channel_sets: Sequence[Optional[Sequence[str]]],
    window_sec: float,
    stride_sec: float,
    y_specs: Sequence[str] = (),
    epoch_sec: float = 30.0,
):
    try:
        import torch  # type: ignore
    except Exception as e:
        raise ImportError("torch is required.") from e

    x_list, x_info_list = make_torch_windows_from_loaded(
        loaded, channel_sets=x_channel_sets, window_sec=window_sec, stride_sec=stride_sec
    )

    ref_info = None
    ref_nwin = 0
    for x, info in zip(x_list, x_info_list):
        if x is not None and info is not None:
            ref_info = info
            ref_nwin = int(x.shape[0])
            break

    if ref_info is None:
        start_secs = np.array([], dtype=np.float64)
    else:
        dt_ref = int(ref_info["dt_ms"])
        stride_samples = int(ref_info["stride_samples"])
        start_samples = np.arange(ref_nwin, dtype=np.int64)*stride_samples
        start_secs = start_samples.astype(np.float64) * (dt_ref/1000.0)

    have_mask = int(loaded.usage_mask)
    have_stage = bool(have_mask & (1 << 1))
    have_arousal = bool(have_mask & (1 << 2))
    have_apnea = bool(have_mask & (1 << 3))

    stages_np = loaded.stages.to_numpy(dtype=np.int32, copy=False) if loaded.stages is not None else None

    def _get_timepoint(name: str) -> Tuple[np.ndarray, int]:
        dt = int(loaded.sampling_interval_ms[name])
        x = loaded.dfs_by_dt_ms[dt][name].to_numpy(dtype=np.float32, copy=False)
        if np.isnan(x).any():
            finite = np.isfinite(x)
            if finite.any():
                x = x[: int(np.where(finite)[0][-1]) + 1]
        return x, dt

    y_list: List[Optional["torch.Tensor"]] = []
    y_info_list: List[Optional[dict]] = []

    for sp in y_specs:
        m = _Y_SPEC_RE.match(sp)
        if not m:
            raise ValueError(f"Invalid y spec: {sp}")
        kind = m.group(1).lower()
        args = [a.strip().lower() for a in m.group(2).split(",") if a.strip()]
        mode = args[0] if args else ""
        agg = args[1] if len(args) >= 2 else ""

        if kind == "stages":
            if not have_stage or stages_np is None:
                y_list.append(None); y_info_list.append({"spec": sp, "available": False}); continue
            labels = np.array([
                _stage_label_for_window(stages_np, float(s), window_sec, mode=mode, epoch_sec=epoch_sec)
                for s in start_secs
            ], dtype=np.int64)
            y_list.append(torch.from_numpy(labels))
            y_info_list.append({"spec": sp, "available": True, "shape": list(labels.shape)})
            continue

        if kind == "arousal":
            if not have_arousal:
                y_list.append(None); y_info_list.append({"spec": sp, "available": False}); continue
            sig, dt = _get_timepoint("Arousal")
            if mode == "timepoint":
                # align to window length in reference dt grid (simple: per-window slice)
                win = _to_seconds_samples(window_sec, dt)
                stride = _to_seconds_samples(stride_sec, dt)
                nwin = len(start_secs)
                arr = np.empty((nwin, win), dtype=np.float32)
                for i in range(nwin):
                    a = i*stride; b = a+win
                    if b <= len(sig):
                        arr[i, :] = sig[a:b]
                    else:
                        tmp = np.full((win,), np.nan, dtype=np.float32)
                        sl = sig[a:len(sig)]
                        tmp[: len(sl)] = sl
                        arr[i, :] = tmp
                y_list.append(torch.from_numpy(arr))
                y_info_list.append({"spec": sp, "available": True, "shape": list(arr.shape)})
                continue
            # epoch mode
            if agg == "":
                agg = "any"
            epoch_bin = _binary_epoch_label_from_timepoint_signal(sig, dt_ms=dt, epoch_sec=epoch_sec)
            out = np.zeros((len(start_secs),), dtype=np.float32 if agg == "ratio" else np.int64)
            for i, s in enumerate(start_secs):
                ks = _epoch_indices_for_window(float(s), window_sec, epoch_sec)
                vals = [int(epoch_bin[k]) for k in ks if 0 <= k < len(epoch_bin)]
                if not vals:
                    out[i] = 0
                elif agg == "any":
                    out[i] = int(any(v > 0 for v in vals))
                elif agg == "count":
                    out[i] = int(sum(v > 0 for v in vals))
                elif agg == "ratio":
                    out[i] = float(sum(v > 0 for v in vals) / len(vals))
            y_list.append(torch.from_numpy(out))
            y_info_list.append({"spec": sp, "available": True, "shape": list(out.shape)})
            continue

        if kind == "apnea":
            if not have_apnea:
                y_list.append(None); y_info_list.append({"spec": sp, "available": False}); continue
            sig, dt = _get_timepoint("ApneaHypopnea")
            if mode == "timepoint":
                win = _to_seconds_samples(window_sec, dt)
                stride = _to_seconds_samples(stride_sec, dt)
                nwin = len(start_secs)
                arr = np.empty((nwin, win), dtype=np.float32)
                for i in range(nwin):
                    a = i*stride; b = a+win
                    if b <= len(sig):
                        arr[i, :] = sig[a:b]
                    else:
                        tmp = np.full((win,), np.nan, dtype=np.float32)
                        sl = sig[a:len(sig)]
                        tmp[: len(sl)] = sl
                        arr[i, :] = tmp
                y_list.append(torch.from_numpy(arr))
                y_info_list.append({"spec": sp, "available": True, "shape": list(arr.shape)})
                continue
            if agg == "":
                agg = "any"
            epoch_bin = _binary_epoch_label_from_timepoint_signal(sig, dt_ms=dt, epoch_sec=epoch_sec)
            out = np.zeros((len(start_secs),), dtype=np.float32 if agg == "ratio" else np.int64)
            for i, s in enumerate(start_secs):
                ks = _epoch_indices_for_window(float(s), window_sec, epoch_sec)
                vals = [int(epoch_bin[k]) for k in ks if 0 <= k < len(epoch_bin)]
                if not vals:
                    out[i] = 0
                elif agg == "any":
                    out[i] = int(any(v > 0 for v in vals))
                elif agg == "count":
                    out[i] = int(sum(v > 0 for v in vals))
                elif agg == "ratio":
                    out[i] = float(sum(v > 0 for v in vals) / len(vals))
            y_list.append(torch.from_numpy(out))
            y_info_list.append({"spec": sp, "available": True, "shape": list(out.shape)})
            continue

        raise ValueError(f"Unsupported y spec: {sp}")

    return x_list, y_list, x_info_list, y_info_list


# =============================================================================

# =============================================================================
# Subject-wise Cross Validation (participant-level split) + cohort -> torch tensors
# =============================================================================

@dataclass(frozen=True)
class CVConfig:
    """Configuration for participant-wise K-fold split."""

    n_folds: int = 5
    seed: int = 0
    shuffle_subjects: bool = True


@dataclass
class CrossValidationSetting:
    """Participant -> fold assignment (0..n_folds-1)."""

    config: CVConfig
    participant_to_fold: Dict[str, int]

    @property
    def n_folds(self) -> int:
        return int(self.config.n_folds)


def _list_participants_from_cohort(cohort: "CohortLoaded") -> List[str]:
    return sorted({pid for (pid, _rnum) in cohort.loaded.keys()})


def _shuffle_participants_if_needed(participants: Sequence[str], *, seed: int, shuffle: bool) -> List[str]:
    pids = [str(pid) for pid in participants]
    if not bool(shuffle):
        return pids
    rng = np.random.default_rng(int(seed))
    rng.shuffle(pids)
    return pids


def _assign_participants_to_folds(participants: Sequence[str], *, n_folds: int) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for i, pid in enumerate(participants):
        mapping[str(pid)] = int(i % int(n_folds))
    return mapping


def build_cv_setting(cohort: "CohortLoaded", config: CVConfig) -> CrossValidationSetting:
    """Deterministically assign each participant to a fold (subject-wise)."""

    pids = _list_participants_from_cohort(cohort)

    if int(config.n_folds) <= 1:
        raise ValueError("n_folds must be >= 2")
    if len(pids) < int(config.n_folds):
        raise ValueError(f"Too few participants ({len(pids)}) for n_folds={config.n_folds}")

    pids = _shuffle_participants_if_needed(
        pids,
        seed=int(config.seed),
        shuffle=bool(config.shuffle_subjects),
    )
    mapping = _assign_participants_to_folds(pids, n_folds=int(config.n_folds))
    return CrossValidationSetting(config=config, participant_to_fold=mapping)


def print_cv_summary(cohort: "CohortLoaded", cv: CrossValidationSetting) -> None:
    counts = {i: 0 for i in range(cv.n_folds)}
    for _pid, f in cv.participant_to_fold.items():
        counts[int(f)] += 1

    print("[sleep_loader][cv] n_participants=", len(cv.participant_to_fold))
    for i in range(cv.n_folds):
        print(f"[sleep_loader][cv] fold={i} n_participants={counts[i]}")


def _iter_records_for_participants(cohort: "CohortLoaded", participants: Sequence[str]):
    pset = set(participants)
    for (pid, rnum), obj in cohort.loaded.items():
        if pid in pset:
            yield pid, rnum, obj


def _participants_for_fold(cv: CrossValidationSetting, *, fold_idx: int) -> Tuple[List[str], List[str]]:
    test_pids = [pid for pid, f in cv.participant_to_fold.items() if f == fold_idx]
    train_pids = [pid for pid, f in cv.participant_to_fold.items() if f != fold_idx]
    return train_pids, test_pids


def _make_xy_for_loaded_record(
    obj: LoadedNPZ,
    *,
    channel_set: Optional[Sequence[str]],
    window_sec: float,
    stride_sec: float,
    y_spec: str,
    epoch_sec: float,
):
    x_list, y_list, _x_info_list, _y_info_list = make_torch_xy_from_loaded(
        obj,
        x_channel_sets=[channel_set],
        window_sec=window_sec,
        stride_sec=stride_sec,
        y_specs=[y_spec],
        epoch_sec=epoch_sec,
    )
    x0 = x_list[0]
    y0 = y_list[0]
    if x0 is None:
        raise RuntimeError("X generation returned None (channel_set was empty?)")
    if y0 is None:
        raise RuntimeError(f"Y generation returned None (maybe unavailable in usage_mask?): y_spec={y_spec}")

    if hasattr(x0, "shape") and hasattr(y0, "shape") and int(x0.shape[0]) != int(y0.shape[0]):
        raise AssertionError(
            "X/Y window count mismatch detected.\n"
            f"  npz={obj.path}\n"
            f"  x.shape={tuple(x0.shape)}\n"
            f"  y.shape={tuple(y0.shape)}\n"
            "This indicates a bug or inconsistency in window/label alignment.\n"
            "Please contact Horie or fix the code and submit a Pull Request."
        )
    return x0, y0


def _assert_consistent_feature_shape(
    *,
    expected_x_feat_shape: Optional[Tuple[int, ...]],
    x0: Any,
    obj: LoadedNPZ,
    context: str,
) -> Tuple[int, ...]:
    got_shape = tuple(x0.shape[1:])
    if expected_x_feat_shape is None:
        return got_shape
    if got_shape != expected_x_feat_shape:
        raise AssertionError(
            f"Inconsistent X feature shape {context}.\n"
            f"  expected x.shape[1:]={expected_x_feat_shape}\n"
            f"  got x.shape[1:]={got_shape}\n"
            f"  npz={obj.path}\n"
            "This likely means channel_set/window parameters differ across records.\n"
            "Please contact Horie or fix the code and submit a Pull Request."
        )
    return expected_x_feat_shape


def _resolve_effective_channel_set(cohort: "CohortLoaded", channel_set: Optional[Sequence[str]]) -> List[str]:
    if channel_set is None:
        common = tuple(getattr(cohort, "common_signal_names", ()) or ())
        if common:
            return list(common)
        raise ValueError("channel_set is None, but no available common_signal_names were found in cohort.")

    resolved = [str(x) for x in channel_set]
    common = tuple(getattr(cohort, "common_signal_names", ()) or ())
    if common:
        missing = [ch for ch in resolved if ch not in set(common)]
        if missing:
            raise ValueError(
                "Requested channel_set contains channels that are not shared across all loaded cohorts: "
                f"missing={missing} available_common={list(common)}"
            )
    return resolved


def _collect_xy_for_participants(
    cohort: "CohortLoaded",
    participants: Sequence[str],
    *,
    channel_set: Optional[Sequence[str]],
    window_sec: float,
    stride_sec: float,
    y_spec: str,
    epoch_sec: float,
    include_y: bool,
    expected_x_feat_shape: Optional[Tuple[int, ...]] = None,
    shape_context: str,
):
    x_list: List[Any] = []
    y_list: List[Any] = []
    record_keys: List[Tuple[str, int]] = []
    resolved_channel_set = _resolve_effective_channel_set(cohort, channel_set)

    for pid, rnum, obj in _iter_records_for_participants(cohort, participants):
        x0, y0 = _make_xy_for_loaded_record(
            obj,
            channel_set=resolved_channel_set,
            window_sec=window_sec,
            stride_sec=stride_sec,
            y_spec=y_spec,
            epoch_sec=epoch_sec,
        )
        expected_x_feat_shape = _assert_consistent_feature_shape(
            expected_x_feat_shape=expected_x_feat_shape,
            x0=x0,
            obj=obj,
            context=shape_context,
        )
        x_list.append(x0)
        record_keys.append((pid, rnum))
        if include_y:
            y_list.append(y0)

    return x_list, y_list, record_keys, expected_x_feat_shape




@dataclass
class IndexedTensorDataset:
    x: Any
    y: Any
    record_index: Any
    participant_index: Any
    epoch_index_within_record: Any
    record_ids: List[str]
    participant_ids: List[str]

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def to_pandas(self) -> pd.DataFrame:
        def _to_numpy(arr):
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            return np.asarray(arr)

        return pd.DataFrame({
            "sample_index": np.arange(len(self), dtype=np.int64),
            "record_index": _to_numpy(self.record_index).astype(np.int64, copy=False),
            "participant_index": _to_numpy(self.participant_index).astype(np.int64, copy=False),
            "epoch_index_within_record": _to_numpy(self.epoch_index_within_record).astype(np.int64, copy=False),
            "record_id": [self.record_ids[int(i)] for i in _to_numpy(self.record_index)],
            "participant_id": [self.participant_ids[int(i)] for i in _to_numpy(self.participant_index)],
        })


@dataclass
class TrainTestTensorBundle:
    train: IndexedTensorDataset
    test: IndexedTensorDataset

    def __iter__(self):
        yield self.train.x
        yield self.train.y
        yield self.test.x
        yield self.test.y


def torch_tensor_to_tf_compatible(x: Any):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def torch_label_to_tf_compatible(y: Any):
    if hasattr(y, "detach"):
        return y.detach().cpu().numpy()
    return np.asarray(y)


def summarize_grouped_predictions(
    y_true: Any,
    y_pred: Any,
    dataset: IndexedTensorDataset,
    *,
    level: str = "record",
) -> pd.DataFrame:
    level_norm = str(level).strip().lower()
    if level_norm not in {"record", "participant"}:
        raise ValueError(f"level must be 'record' or 'participant' (got {level!r})")

    def _to_numpy(arr):
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        return np.asarray(arr)

    y_true_np = _to_numpy(y_true).reshape(-1)
    y_pred_np = _to_numpy(y_pred).reshape(-1)
    if y_true_np.shape[0] != len(dataset) or y_pred_np.shape[0] != len(dataset):
        raise ValueError(
            "Prediction length mismatch. "
            f"len(dataset)={len(dataset)} len(y_true)={y_true_np.shape[0]} len(y_pred)={y_pred_np.shape[0]}"
        )

    meta = dataset.to_pandas()
    meta["y_true"] = y_true_np
    meta["y_pred"] = y_pred_np
    meta["correct"] = (y_true_np == y_pred_np).astype(np.int32)

    key = "record_id" if level_norm == "record" else "participant_id"
    out = (
        meta.groupby(key, sort=False)
        .agg(
            n_samples=("correct", "size"),
            accuracy=("correct", "mean"),
            record_index=("record_index", "first"),
            participant_index=("participant_index", "first"),
        )
        .reset_index()
    )
    return out


def _build_indexed_tensor_dataset(
    x_list: Sequence[Any],
    y_list: Sequence[Any],
    *,
    participants: Sequence[str],
    cohort: "CohortLoaded",
    record_keys: Sequence[Tuple[str, int]],
):
    try:
        import torch  # type: ignore
    except Exception as e:
        raise ImportError("torch is required.") from e

    x = torch.cat(list(x_list), dim=0)
    y = torch.cat(list(y_list), dim=0) if y_list else torch.empty((0,), dtype=torch.long)

    participant_ids = list(participants)
    participant_to_index = {pid: i for i, pid in enumerate(participant_ids)}

    record_ids: List[str] = []
    record_index_parts = []
    participant_index_parts = []
    epoch_index_parts = []

    for rec_i, ((pid, rnum), x0) in enumerate(zip(record_keys, x_list)):
        obj = cohort.loaded[(pid, rnum)]
        if hasattr(cohort, "record_key_to_record_id"):
            record_id = str(getattr(cohort, "record_key_to_record_id")[(pid, rnum)])
        else:
            record_id = obj.path.stem
        record_ids.append(record_id)
        n = int(x0.shape[0])
        record_index_parts.append(torch.full((n,), rec_i, dtype=torch.long))
        participant_index_parts.append(torch.full((n,), participant_to_index[pid], dtype=torch.long))
        epoch_index_parts.append(torch.arange(n, dtype=torch.long))

    record_index = torch.cat(record_index_parts, dim=0) if record_index_parts else torch.empty((0,), dtype=torch.long)
    participant_index = torch.cat(participant_index_parts, dim=0) if participant_index_parts else torch.empty((0,), dtype=torch.long)
    epoch_index_within_record = torch.cat(epoch_index_parts, dim=0) if epoch_index_parts else torch.empty((0,), dtype=torch.long)

    if int(x.shape[0]) != int(record_index.shape[0]):
        raise AssertionError(
            "Index length mismatch while building IndexedTensorDataset. "
            f"x.shape[0]={int(x.shape[0])} record_index.shape[0]={int(record_index.shape[0])}"
        )

    return IndexedTensorDataset(
        x=x,
        y=y,
        record_index=record_index,
        participant_index=participant_index,
        epoch_index_within_record=epoch_index_within_record,
        record_ids=record_ids,
        participant_ids=participant_ids,
    )


def load_training_testing_data(
    cohort: "CohortLoaded",
    cv: CrossValidationSetting,
    *,
    fold_idx: int,
    channel_set: Optional[Sequence[str]],
    window_sec: float,
    stride_sec: float,
    y_spec: str,
    epoch_sec: float = 30.0,
    include_test_y: bool = True,
    normalization: Optional[str] = None,
):
    """Subject-wise K-fold split (participant-wise; no leakage across train/test).

    Returns:
        TrainTestTensorBundle

    Backward compatibility:
        The returned bundle is iterable, so the following still works:
            train_x, train_y, test_x, test_y = load_training_testing_data(...)
    """
    try:
        import torch  # type: ignore
    except Exception as e:
        raise ImportError("torch is required.") from e

    if fold_idx < 0 or fold_idx >= cv.n_folds:
        raise ValueError(f"fold_idx out of range: {fold_idx} (n_folds={cv.n_folds})")

    train_pids, test_pids = _participants_for_fold(cv, fold_idx=fold_idx)
    norm_mode = _resolve_normalization_mode(normalization)

    expected_x_feat_shape: Optional[Tuple[int, ...]] = None

    train_x_list, train_y_list, train_record_keys, expected_x_feat_shape = _collect_xy_for_participants(
        cohort,
        train_pids,
        channel_set=channel_set,
        window_sec=window_sec,
        stride_sec=stride_sec,
        y_spec=y_spec,
        epoch_sec=epoch_sec,
        include_y=True,
        expected_x_feat_shape=expected_x_feat_shape,
        shape_context="across records",
    )
    if not train_x_list:
        raise RuntimeError(f"No training records found for fold={fold_idx}")

    test_x_list, test_y_list, test_record_keys, expected_x_feat_shape = _collect_xy_for_participants(
        cohort,
        test_pids,
        channel_set=channel_set,
        window_sec=window_sec,
        stride_sec=stride_sec,
        y_spec=y_spec,
        epoch_sec=epoch_sec,
        include_y=bool(include_test_y),
        expected_x_feat_shape=expected_x_feat_shape,
        shape_context="between train/test records",
    )
    if not test_x_list:
        raise RuntimeError(f"No testing records found for fold={fold_idx}")

    if norm_mode == "epoch-wise":
        train_x_list = [_normalize_x_tensor(x, mode=norm_mode) for x in train_x_list]
        test_x_list = [_normalize_x_tensor(x, mode=norm_mode) for x in test_x_list]
    elif norm_mode == "record-wise":
        train_x_list = [_normalize_x_tensor_recordwise(x) for x in train_x_list]
        test_x_list = [_normalize_x_tensor_recordwise(x) for x in test_x_list]

    train_ds = _build_indexed_tensor_dataset(
        train_x_list,
        train_y_list,
        participants=train_pids,
        cohort=cohort,
        record_keys=train_record_keys,
    )
    test_ds = _build_indexed_tensor_dataset(
        test_x_list,
        test_y_list,
        participants=test_pids,
        cohort=cohort,
        record_keys=test_record_keys,
    )

    if not include_test_y:
        test_ds.y = torch.empty((0,), dtype=torch.long)

    if norm_mode == "train-wise":
        train_x, test_x, _norm_info = _apply_trainwise_normalization(train_ds.x, test_ds.x)
        train_ds.x = train_x
        test_ds.x = test_x
    elif norm_mode == "none":
        pass

    return TrainTestTensorBundle(train=train_ds, test=test_ds)


def _resolve_normalization_mode(normalization: Optional[str]) -> str:
    if normalization is None:
        return "none"
    mode = str(normalization).strip().lower()
    aliases = {
        "": "none",
        "none": "none",
        "off": "none",
        "false": "none",
        "no": "none",
        "epoch-wise": "epoch-wise",
        "epochwise": "epoch-wise",
        "per_epoch": "epoch-wise",
        "record-wise": "record-wise",
        "recordwise": "record-wise",
        "per_record": "record-wise",
        "train-wise": "train-wise",
        "trainwise": "train-wise",
        "per_train": "train-wise",
    }
    if mode not in aliases:
        raise ValueError(
            "normalization must be one of: none, epoch-wise, record-wise, train-wise "
            f"(got {normalization!r})"
        )
    return aliases[mode]


def _normalize_x_tensor(x, *, mode: str, eps: float = 1e-6):
    try:
        import torch  # type: ignore
    except Exception as e:
        raise ImportError("torch is required.") from e

    if x.ndim != 3:
        raise ValueError(
            "Normalization currently expects X shaped [N, C, L]. "
            f"Got x.shape={tuple(x.shape)}"
        )

    x = x.to(dtype=torch.float32)

    if mode == "epoch-wise":
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True, unbiased=False)
    else:
        raise ValueError(f"Unsupported per-tensor normalization mode: {mode}")

    std = torch.where(std < eps, torch.ones_like(std), std)
    return (x - mean) / std


def _normalize_x_tensor_recordwise(x, *, eps: float = 1e-6):
    """Record-wise z-normalization for one record tensor shaped [N, C, L].

    Statistics are computed independently for each record and channel,
    using all windows/epochs and all time samples belonging to that record.
    """
    try:
        import torch  # type: ignore
    except Exception as e:
        raise ImportError("torch is required.") from e

    if x.ndim != 3:
        raise ValueError(
            "Record-wise normalization currently expects X shaped [N, C, L]. "
            f"Got x.shape={tuple(x.shape)}"
        )

    x = x.to(dtype=torch.float32)
    mean = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True, unbiased=False)
    std = torch.where(std < eps, torch.ones_like(std), std)
    return (x - mean) / std


def _apply_trainwise_normalization(train_x, test_x, *, eps: float = 1e-6):
    try:
        import torch  # type: ignore
    except Exception as e:
        raise ImportError("torch is required.") from e

    if train_x.ndim != 3 or test_x.ndim != 3:
        raise ValueError(
            "Normalization currently expects X shaped [N, C, L]. "
            f"Got train_x.shape={tuple(train_x.shape)}, test_x.shape={tuple(test_x.shape)}"
        )

    train_x = train_x.to(dtype=torch.float32)
    test_x = test_x.to(dtype=torch.float32)

    mean = train_x.mean(dim=(0, 2), keepdim=True)
    std = train_x.std(dim=(0, 2), keepdim=True, unbiased=False)
    std = torch.where(std < eps, torch.ones_like(std), std)

    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    return train_x, test_x, {"mode": "train-wise", "mean": mean, "std": std}



def _resolve_cli_input(
    raw_input: Union[str, Path],
    *,
    cohort_id: Optional[str] = None,
) -> Tuple[Path, str, Optional[str]]:
    """
    Resolve CLI input as either:
      - existing file path
      - existing directory path
      - cohort_id searched under candidate roots

    Returns:
      (resolved_path, kind, resolved_cohort_id)

    kind:
      - "file"
      - "dir"
    """
    p = Path(raw_input).expanduser()

    if p.exists():
        rp = p.resolve()
        if rp.is_file():
            return rp, "file", None
        if rp.is_dir():
            return rp, "dir", (str(cohort_id) if cohort_id else None)
        raise FileNotFoundError(f"Path exists but is neither file nor directory: {rp}")

    rp, cid = _resolve_cohort_input(raw_input, cohort_id=cohort_id)
    if rp.is_dir():
        return rp, "dir", cid

    raise FileNotFoundError(f"Resolved CLI input is not a directory: {rp}")


def _cli_npz(npz_path: Path, args: argparse.Namespace) -> int:
    loaded = load_npz(npz_path, purpose=args.purpose)

    print(f"[sleep_loader] npz={loaded.path}")
    print(f"[sleep_loader] requested_purpose={loaded.requested_purpose}")
    print(f"[sleep_loader] requested_tasks={loaded.requested_tasks}")
    print(f"[sleep_loader] extra_signals={loaded.requested_extra_signals}")
    print(f"[sleep_loader] usage_mask={loaded.usage_mask} have={_decode_tasks_from_mask(int(loaded.usage_mask))}")

    channel_names = []
    try:
        if getattr(loaded, "signal_info", None) is not None and loaded.signal_info.size > 0:
            channel_names = [str(x) for x in loaded.signal_info[:, 0].tolist()]
    except Exception:
        channel_names = []
    print(f"[sleep_loader] n_channels={len(channel_names)}")
    if channel_names:
        print(f"[sleep_loader] channels(head)={channel_names[:10]}")

    print(f"[sleep_loader] dt_groups={sorted(int(k) for k in loaded.dfs_by_dt_ms.keys())}")
    for dt in sorted(int(k) for k in loaded.dfs_by_dt_ms.keys()):
        df = loaded.dfs_by_dt_ms[dt]
        print(f"[sleep_loader] dt_ms={dt} shape={tuple(df.shape)} columns(head)={list(df.columns[:10])}")

    if loaded.stages is not None:
        print(f"[sleep_loader] stages_len={len(loaded.stages)}")
    else:
        print("[sleep_loader] stages_len=NA")

    channel_sets = []
    if getattr(args, 'channel_sets', ''):
        raw = str(args.channel_sets).strip()
        if raw:
            channel_sets = json.loads(f"[{raw}]")

    y_specs = json.loads(str(getattr(args, 'y_specs', '[]')))
    if channel_sets or y_specs:
        x_list, y_list, x_info_list, y_info_list = make_torch_xy_from_loaded(
            loaded,
            x_channel_sets=channel_sets if channel_sets else [None],
            window_sec=float(args.window_sec),
            stride_sec=float(args.stride_sec),
            y_specs=y_specs,
            epoch_sec=float(args.epoch_sec),
        )
        for i, (x, info) in enumerate(zip(x_list, x_info_list)):
            shape = None if x is None else tuple(x.shape)
            print(f"[sleep_loader][x{i}] shape={shape} info={info}")
        for i, (y, info) in enumerate(zip(y_list, y_info_list)):
            shape = None if y is None else tuple(y.shape)
            print(f"[sleep_loader][y{i}] shape={shape} info={info}")

    return 0


def _cli_cohort(cohort_dir: Path, args: argparse.Namespace, cohort_id: Optional[str] = None) -> int:
    multi_raw = str(getattr(args, "cohorts", "") or "").strip()
    if multi_raw:
        cohort_inputs = [x.strip() for x in multi_raw.split(",") if x.strip()]
        if str(cohort_dir) not in cohort_inputs and str(cohort_id or "") not in cohort_inputs:
            cohort_inputs = [str(cohort_dir)] + cohort_inputs
        loaded = load_cohorts(
            cohort_inputs,
            where=args.where,
            purpose=args.purpose,
            limit=args.limit,
            order_by=args.order_by,
            verbose=True,
            compatibility_config=args.compatibility_config,
        )
    else:
        loaded = load_cohort(
            cohort_dir,
            where=args.where,
            purpose=args.purpose,
            cohort_id=cohort_id,
            limit=args.limit,
            order_by=args.order_by,
            verbose=True,
            compatibility_config=args.compatibility_config,
        )

    print(loaded.hits_df.head(int(args.head)).to_string(index=False))

    if args.show:
        cols = [c.strip() for c in str(args.show).split(',') if c.strip()]
        existing = [c for c in cols if c in loaded.hits_df.columns]
        if existing:
            print("\n[selected columns]")
            print(loaded.hits_df[existing].head(int(args.head)).to_string(index=False))

    return 0

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="sleep_loader.py",
        description=(
            "Unified CLI: pass an .npz file, a cohort directory, or a cohort ID "
            "for auto-resolution (local first)."
        ),
    )

    ap.add_argument(
        "path",
        nargs="?",
        default=None,
        help="NPZ file path, cohort directory path, cohort ID, or literal mode token: npz|cohort",
    )
    ap.add_argument(
        "path2",
        nargs="?",
        default=None,
        help="Path or cohort ID when first argument is an explicit mode token (npz|cohort)",
    )

    ap.add_argument("--purpose", default="self_supervised", help="purpose tokens (e.g., supervised_stage+signal(SpO2))")
    ap.add_argument("--where", default="1=1", help="SQL WHERE clause (cohort mode)")
    ap.add_argument("--limit", type=int, default=None, help="LIMIT for cohort SQL")
    ap.add_argument("--order_by", default="r.record_id ASC", help="ORDER BY for cohort SQL")
    ap.add_argument("--head", type=int, default=50, help="Print first N loaded records (cohort mode)")
    ap.add_argument("--show", default="", help="Comma-separated columns to print from Record row (cohort mode)")
    ap.add_argument(
        "--cohorts",
        default="",
        help="Comma-separated cohort directories / cohort IDs to merge and load together in cohort mode",
    )

    ap.add_argument(
        "--compatibility-config",
        default=None,
        help="Path to TOML config for cohort compatibility checks (default: cohort_mix_rules.toml). Use an empty string to disable.",
    )

    ap.add_argument(
        "--channel_sets",
        default="",
        help=(
            "Comma-separated JSON lists. Example: '[\"EEG_C3-R\",\"EEG_C4-R\"]', '[\"EOG_LOC-R\"]'"
        ),
    )
    ap.add_argument("--window_sec", type=float, default=30.0)
    ap.add_argument("--stride_sec", type=float, default=30.0)
    ap.add_argument("--y_specs", default="[]", help="JSON list, e.g. '[\"stages(vote)\", \"arousal(epoch,any)\"]'")
    ap.add_argument("--epoch_sec", type=float, default=30.0)

    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.compatibility_config == "":
        args.compatibility_config = False

    primary_input = args.path
    if primary_input is None:
        cohorts_csv = str(args.cohorts).strip()
        if cohorts_csv:
            primary_input = cohorts_csv.split(",")[0].strip()
        else:
            raise SystemExit("sleep_loader.py: error: path is required unless --cohorts is provided")

    mode_token = str(primary_input).strip().lower()

    try:
        if mode_token in {"cohort", "npz"}:
            if not args.path2:
                raise SystemExit("sleep_loader.py: error: missing <path_or_cohort_id> after explicit mode token")

            target_input = args.path2

            if mode_token == "npz":
                p = Path(target_input).expanduser().resolve()
                if not p.exists() or not p.is_file():
                    raise FileNotFoundError(f"NPZ path not found: {p}")
                return _cli_npz(p, args)

            p, kind, resolved_cid = _resolve_cli_input(target_input)
            if kind != "dir":
                raise ValueError(f"Explicit cohort mode requires a directory/cohort_id, got file: {p}")
            return _cli_cohort(p, args, cohort_id=resolved_cid)

        p, kind, resolved_cid = _resolve_cli_input(primary_input)

        if kind == "file":
            return _cli_npz(p, args)
        if kind == "dir":
            return _cli_cohort(p, args, cohort_id=resolved_cid)

        raise RuntimeError(f"Unexpected resolved kind: {kind}")

    except SystemExit:
        raise
    except Exception as e:
        print(f"[sleep_loader][ERROR] {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
