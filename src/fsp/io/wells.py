"""
Injection well data loading and normalisation.
Port of FSP/core/utilities.jl prepare_well_data_for_pressure_scenario and helpers.
"""
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class ProcessedWellData:
    """Pre-processed single well ready for Theis calculations."""
    well_id: str
    latitude: float
    longitude: float
    start_date: date
    end_date: date
    start_year: int
    end_year: int
    days: np.ndarray    # days from injection start
    rates: np.ndarray   # bbl/day


def load_injection_wells(path: str, data_type: str) -> pd.DataFrame:
    """Load an injection wells CSV.  data_type drives column type for IDs."""
    if data_type == "injection_tool_data":
        return pd.read_csv(path, dtype={"API Number": str})
    else:
        return pd.read_csv(path, dtype={"WellID": str})


def get_date_bounds(df: pd.DataFrame) -> Tuple[date, date]:
    """Return (earliest_date, latest_date) across all injection records.

    Port of Julia Utilities.get_date_bounds.
    Works for annual_fsp, monthly_fsp, and injection_tool_data formats.
    """
    if "StartYear" in df.columns:
        start = date(int(df["StartYear"].min()), 1, 1)
        end = date(int(df["EndYear"].max()), 12, 31)
    elif "Year" in df.columns:
        min_year = int(df["Year"].min())
        max_year = int(df["Year"].max())
        min_month = int(df[df["Year"] == min_year]["Month"].min())
        max_month = int(df[df["Year"] == max_year]["Month"].max())
        start = date(min_year, min_month, 1)
        last_day = _last_day_of_month(max_year, max_month)
        end = date(max_year, max_month, last_day)
    elif "Date of Injection" in df.columns:
        parsed = _parse_dates_column(df["Date of Injection"])
        start = min(parsed)
        end = max(parsed)
    else:
        raise ValueError("Cannot determine date bounds from injection well data")
    return start, end


def preprocess_well_data(df: pd.DataFrame, data_type: str) -> dict:
    """Pre-process injection DataFrame into a dict of ProcessedWellData keyed by well_id.

    Port of Julia preprocess_well_data.
    """
    well_id_col = "API Number" if data_type == "injection_tool_data" else "WellID"
    lat_col = "Surface Latitude" if data_type == "injection_tool_data" else "Latitude(WGS84)"
    lon_col = "Surface Longitude" if data_type == "injection_tool_data" else "Longitude(WGS84)"

    well_info = {}

    grouped_df = df.copy()
    grouped_df["_well_id_str"] = grouped_df[well_id_col].astype(str)

    for well_id, well_data in grouped_df.groupby("_well_id_str", sort=False):
        well_data = well_data.drop(columns=["_well_id_str"]).copy()
        if well_data.empty:
            continue

        lat = float(well_data.iloc[0][lat_col])
        lon = float(well_data.iloc[0][lon_col])

        if data_type == "annual_fsp":
            sy = int(well_data.iloc[0]["StartYear"])
            ey = int(well_data.iloc[0]["EndYear"])
            start_d = date(sy, 1, 1)
            end_d = date(ey - 1, 12, 31)

        elif data_type == "monthly_fsp":
            min_year = int(well_data["Year"].min())
            max_year = int(well_data["Year"].max())
            min_month = int(well_data[well_data["Year"] == min_year]["Month"].min())
            max_month = int(well_data[well_data["Year"] == max_year]["Month"].max())
            start_d = date(min_year, min_month, 1)
            ldom = _last_day_of_month(max_year, max_month)
            end_d = date(max_year, max_month, ldom)

        elif data_type == "injection_tool_data":
            dates = _parse_dates_column(well_data["Date of Injection"])
            start_d = min(dates)
            end_d = max(dates)

        else:
            continue

        well_info[well_id] = ProcessedWellData(
            well_id=well_id,
            latitude=lat,
            longitude=lon,
            start_date=start_d,
            end_date=end_d,
            start_year=start_d.year,
            end_year=end_d.year,
            days=np.array([], dtype=float),
            rates=np.array([], dtype=float),
        )
        # Store reference to well_data for later processing
        well_info[well_id]._raw_data = well_data

    return well_info


def normalize_wells_to_well_data(well_info: dict, data_type: str,
                                   cutoff_date: date) -> List[ProcessedWellData]:
    """Convert pre-processed wells to ProcessedWellData with populated days/rates arrays.

    Port of Julia prepare_well_data_for_pressure_scenario.
    cutoff_date = Dec 31 of (analysis_year - 1).
    """
    result = []
    for well_id, wd in well_info.items():
        if wd.start_date > cutoff_date:
            continue

        actual_end = min(wd.end_date, cutoff_date)
        raw = wd._raw_data

        days, rates = _prepare_days_rates(raw, wd.start_date, actual_end,
                                           data_type, cutoff_date)
        if len(days) == 0:
            continue

        result.append(ProcessedWellData(
            well_id=well_id,
            latitude=wd.latitude,
            longitude=wd.longitude,
            start_date=wd.start_date,
            end_date=actual_end,
            start_year=wd.start_year,
            end_year=actual_end.year,
            days=days,
            rates=rates,
        ))
    return result


def _prepare_days_rates(well_data: pd.DataFrame, start_date: date,
                         end_date: date, data_type: str,
                         cutoff_date: date) -> Tuple[np.ndarray, np.ndarray]:
    """Build (days, rates) arrays from filtered well data.

    days are counted from start_date (day 1 = first day).
    """
    if data_type == "annual_fsp":
        if "InjectionRate(bbl/day)" not in well_data.columns:
            return np.array([]), np.array([])
        rate = float(well_data.iloc[0]["InjectionRate(bbl/day)"])
        total_days = (end_date - start_date).days + 1
        days = np.array([1.0, float(total_days)])
        rates = np.array([rate, 0.0])
        return days, rates

    elif data_type == "monthly_fsp":
        return _monthly_fsp_days_rates(well_data, start_date, end_date)

    elif data_type == "injection_tool_data":
        return _injection_tool_days_rates(well_data, start_date, end_date)

    return np.array([]), np.array([])


def _monthly_fsp_days_rates(well_data: pd.DataFrame, start_date: date,
                              end_date: date) -> Tuple[np.ndarray, np.ndarray]:
    """Convert monthly injection volumes to step-change days/rates arrays."""
    rate_col = None
    for c in ["InjectionRate(bbl/month)", "Injection Rate (bbl/month)", "MonthlyInjectionRate"]:
        if c in well_data.columns:
            rate_col = c
            break
    if rate_col is None:
        return np.array([]), np.array([])

    days_list = []
    rates_list = []

    for _, row in well_data.iterrows():
        try:
            yr = int(row["Year"])
            mo = int(row["Month"])
            monthly_vol = float(row[rate_col])
        except (ValueError, KeyError):
            continue

        month_start = date(yr, mo, 1)
        if month_start > end_date:
            continue

        days_in_month = _last_day_of_month(yr, mo)
        daily_rate = monthly_vol / days_in_month

        day_offset = float((month_start - start_date).days + 1)
        days_list.append(day_offset)
        rates_list.append(daily_rate)

    if not days_list:
        return np.array([]), np.array([])

    # Sort by day
    order = np.argsort(days_list)
    return np.array(days_list)[order], np.array(rates_list)[order]


def _injection_tool_days_rates(well_data: pd.DataFrame, start_date: date,
                                 end_date: date) -> Tuple[np.ndarray, np.ndarray]:
    """Convert injection tool data to days/rates arrays."""
    dates = _parse_dates_column(well_data["Date of Injection"])

    # Determine rate column
    rate_col = None
    for c in ["Monthly Injection Volume (BBLs)", "Annual Injection Volume (BBLs)",
              "Injection Volume (BBL)", "BPD"]:
        if c in well_data.columns:
            rate_col = c
            break
    if rate_col is None:
        return np.array([]), np.array([])

    is_monthly = "Monthly" in rate_col

    days_list = []
    rates_list = []

    for (d, row) in zip(dates, well_data.itertuples()):
        if d > end_date:
            continue
        try:
            vol = float(getattr(row, rate_col.replace(" ", "_").replace("(", "").replace(")", "")))
        except AttributeError:
            try:
                idx = well_data.columns.get_loc(rate_col)
                vol = float(row[idx + 1])  # +1 because itertuples includes index at [0]
            except Exception:
                continue

        if is_monthly:
            days_in_month = _last_day_of_month(d.year, d.month)
            rate = vol / days_in_month
        else:
            rate = vol / 365.0

        day_offset = float((d - start_date).days + 1)
        days_list.append(day_offset)
        rates_list.append(rate)

    if not days_list:
        return np.array([]), np.array([])

    order = np.argsort(days_list)
    return np.array(days_list)[order], np.array(rates_list)[order]


def injection_rate_data_to_d3_bbl_day(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Convert injection well DataFrame to D3-compatible time-series (bbl/day).

    Port of Julia injection_rate_data_to_d3_bbl_day.
    Returns both legacy Python columns and Julia-compatible graph columns.
    """
    rows = []

    if data_type == "annual_fsp":
        for _, row in df.iterrows():
            wid = str(row["WellID"])
            rate = float(row["InjectionRate(bbl/day)"])
            sy = int(row["StartYear"])
            ey = int(row["EndYear"])
            for yr in range(sy, ey + 1):
                d = date(yr, 1, 1)
                rows.append({
                    "WellID": wid,
                    "date": d.strftime("%Y-%m-%d"),
                    "rate_bbl_day": rate,
                    "InjectionRate(bbl/day)": rate,
                    "Timestamp": float(datetime(d.year, d.month, d.day).timestamp() * 1000.0),
                })

    elif data_type == "monthly_fsp":
        rate_col = None
        for c in ["InjectionRate(bbl/month)", "Injection Rate (bbl/month)"]:
            if c in df.columns:
                rate_col = c
                break
        if rate_col is None:
            return pd.DataFrame(rows)

        for _, row in df.iterrows():
            wid = str(row["WellID"])
            yr = int(row["Year"])
            mo = int(row["Month"])
            vol = float(row[rate_col])
            days_in_month = _last_day_of_month(yr, mo)
            daily_rate = vol / days_in_month
            d = date(yr, mo, 1)
            rows.append({
                "WellID": wid,
                "date": d.strftime("%Y-%m-%d"),
                "rate_bbl_day": daily_rate,
                "InjectionRate(bbl/day)": daily_rate,
                "Timestamp": float(datetime(d.year, d.month, d.day).timestamp() * 1000.0),
            })

    elif data_type == "injection_tool_data":
        for _, row in df.iterrows():
            wid = str(row.get("API Number", row.get("UWI", "Unknown")))
            d_str = str(row.get("Date of Injection", ""))
            try:
                d = _parse_single_date(d_str)
            except Exception:
                continue
            for c in ["Monthly Injection Volume (BBLs)", "Annual Injection Volume (BBLs)"]:
                if c in df.columns:
                    vol = float(row[c])
                    is_monthly = "Monthly" in c
                    dpm = _last_day_of_month(d.year, d.month) if is_monthly else 365
                    rows.append({
                        "WellID": wid,
                        "date": d.strftime("%Y-%m-%d"),
                        "rate_bbl_day": vol / dpm,
                        "InjectionRate(bbl/day)": vol / dpm,
                        "Timestamp": float(datetime(d.year, d.month, d.day).timestamp() * 1000.0),
                    })
                    break

    return pd.DataFrame(rows)


def _parse_dates_column(col: pd.Series) -> list:
    """Parse a date column to list of date objects."""
    parsed_series = pd.to_datetime(col, errors="coerce", format="mixed")
    parsed = []
    for original, parsed_value in zip(col, parsed_series):
        if isinstance(original, date):
            parsed.append(original)
        elif pd.notna(parsed_value):
            parsed.append(parsed_value.date())
        else:
            parsed.append(_parse_single_date(str(original)))
    return parsed


def _parse_single_date(s: str) -> date:
    """Try common date formats."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"):
        try:
            import datetime
            return datetime.datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Cannot parse date: {s!r}")


def _last_day_of_month(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]
