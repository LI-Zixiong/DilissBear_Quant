"""
Tests for chronological splits and future-leakage prevention.

These tests use synthetic panel data to verify:
- train/valid/test date boundaries are respected
- sequence windows do not include future rows
- next-period targets are mapped to the next global return date
"""

import numpy as np
import pandas as pd

from src.experiment.returns import build_returns_frame_from_next_target
from src.data.dataset_builder import PanelDatasetBuilder


def _make_panel_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    stock_ids = ["A", "B", "C"]

    rows = []

    for date_idx, date in enumerate(dates):
        for stock_idx, stock_id in enumerate(stock_ids):
            row_id = date_idx * 10 + stock_idx

            rows.append(
                {
                    "date": date,
                    "stock_id": stock_id,
                    "factor_1": float(row_id),
                    "factor_2": float(row_id) + 0.5,
                    "target": float(row_id) + 1.0,
                    "row_id": float(row_id),
                    "industry_code": f"IND_{stock_idx}",
                }
            )

    return pd.DataFrame(rows)


def _make_builder(seq_len: int = 3) -> PanelDatasetBuilder:
    return PanelDatasetBuilder(
        feature_cols=["factor_1", "factor_2"],
        target_col="target",
        date_col="date",
        stock_col="stock_id",
        seq_len=seq_len,
        meta_cols=["row_id", "industry_code"],
    )


def test_split_built_dataset_respects_chronological_boundaries() -> None:
    df = _make_panel_df()
    builder = _make_builder()

    dataset = builder.build_tabular_dataset(df)

    train_data, valid_data, test_data = builder.split_built_dataset(
        dataset=dataset,
        train_end="2020-01-03",
        valid_end="2020-01-05",
    )

    train_dates = pd.to_datetime(train_data.meta["date"])
    valid_dates = pd.to_datetime(valid_data.meta["date"])
    test_dates = pd.to_datetime(test_data.meta["date"])

    assert train_dates.max() <= pd.Timestamp("2020-01-03")

    assert valid_dates.min() > pd.Timestamp("2020-01-03")
    assert valid_dates.max() <= pd.Timestamp("2020-01-05")

    assert test_dates.min() > pd.Timestamp("2020-01-05")

    assert set(train_dates.unique()).isdisjoint(set(valid_dates.unique()))
    assert set(train_dates.unique()).isdisjoint(set(test_dates.unique()))
    assert set(valid_dates.unique()).isdisjoint(set(test_dates.unique()))


def test_sequence_windows_do_not_include_future_rows() -> None:
    df = _make_panel_df()
    seq_len = 3
    builder = _make_builder(seq_len=seq_len)

    dataset = builder.build_sequence_dataset(df)

    assert dataset.X.ndim == 3
    assert dataset.X.shape[1] == seq_len

    sample_row_ids = dataset.meta["row_id"].to_numpy(dtype=float)
    window_row_ids = dataset.X[:, :, 0]

    for sample_row_id, window_ids in zip(sample_row_ids, window_row_ids):
        # The last row in the sequence should correspond to the sample itself.
        assert window_ids[-1] == sample_row_id

        # No future row should appear inside the historical window.
        assert (window_ids <= sample_row_id).all()

        # All rows in a sequence should belong to the same synthetic stock.
        assert (np.mod(window_ids, 10) == np.mod(sample_row_id, 10)).all()


def test_sequence_split_respects_sample_dates() -> None:
    df = _make_panel_df()
    seq_len = 3
    builder = _make_builder(seq_len=seq_len)

    dataset = builder.build_sequence_dataset(df)

    train_data, valid_data, test_data = builder.split_built_dataset(
        dataset=dataset,
        train_end="2020-01-04",
        valid_end="2020-01-06",
    )

    train_dates = pd.to_datetime(train_data.meta["date"])
    valid_dates = pd.to_datetime(valid_data.meta["date"])
    test_dates = pd.to_datetime(test_data.meta["date"])

    assert train_dates.max() <= pd.Timestamp("2020-01-04")

    assert valid_dates.min() > pd.Timestamp("2020-01-04")
    assert valid_dates.max() <= pd.Timestamp("2020-01-06")

    assert test_dates.min() > pd.Timestamp("2020-01-06")


def test_returns_frame_maps_target_to_next_global_date() -> None:
    df = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-03",
            ],
            "stock_id": ["A", "B", "A", "A", "B"],
            "target": [0.10, 0.20, 0.30, 0.40, 0.50],
        }
    )

    returns_df = build_returns_frame_from_next_target(
        df=df,
        date_col="date",
        stock_col="stock_id",
        target_col="target",
        return_col="return_1d",
    )

    returns_df["date"] = pd.to_datetime(returns_df["date"])

    b_return = returns_df[
        (returns_df["date"] == pd.Timestamp("2020-01-02"))
        & (returns_df["stock_id"] == "B")
    ]

    assert len(b_return) == 1
    assert b_return["return_1d"].iloc[0] == 0.20


def test_returns_frame_drops_last_global_date_targets() -> None:
    df = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
            ],
            "stock_id": ["A", "B", "A", "B"],
            "target": [0.10, 0.20, 0.30, 0.40],
        }
    )

    returns_df = build_returns_frame_from_next_target(
        df=df,
        date_col="date",
        stock_col="stock_id",
        target_col="target",
        return_col="return_1d",
    )

    returns_df["date"] = pd.to_datetime(returns_df["date"])

    assert set(returns_df["date"].unique()) == {pd.Timestamp("2020-01-02")}
    assert len(returns_df) == 2
    assert not returns_df.duplicated(subset=["date", "stock_id"]).any()


def test_returns_frame_has_unique_date_stock_rows() -> None:
    df = _make_panel_df()

    returns_df = build_returns_frame_from_next_target(
        df=df,
        date_col="date",
        stock_col="stock_id",
        target_col="target",
        return_col="return_1d",
    )

    assert not returns_df.duplicated(subset=["date", "stock_id"]).any()
    assert (pd.to_datetime(returns_df["date"]) > pd.Timestamp("2020-01-01")).all()
