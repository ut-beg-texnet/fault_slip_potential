"""
Injection well data loading and normalisation.
Port of FSP/core/utilities.jl prepare_well_data_for_pressure_scenario and helpers.
"""
import calendar
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Portal true values for extrapolate_injection_rates. Missing/None/false -> False.
_EXTRAPOLATE_TRUE_STRINGS = {"true", "yes", "1"}


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


def coerce_extrapolate_injection_rates(value) -> bool:
    """True only for explicit portal true values; missing/None/false stays false."""
    if isinstance(value, str):
        return value.strip().lower() in _EXTRAPOLATE_TRUE_STRINGS
    return value is True or value == 1


def resolve_extrapolate_injection_rates(
    get_param: Callable[[int, str], object],
    current_step: int,
) -> bool:
    """Read extrapolate_injection_rates from current step, then Step 4, then Step 1."""
    for step in (current_step, 3, 0):
        value = get_param(step, "extrapolate_injection_rates")
        if value is not None:
            return coerce_extrapolate_injection_rates(value)
    return False


def normalize_wells_to_well_data(
    well_info: dict,
    data_type: str,
    cutoff_date: date,
    extrapolate_injection_rates: bool = False,
) -> List[ProcessedWellData]:
    """Convert pre-processed wells to ProcessedWellData with populated days/rates arrays.

    Port of Julia prepare_well_data_for_pressure_scenario.
    cutoff_date = Dec 31 of (analysis_year - 1).
    extrapolate_injection_rates continues the last monthly rate to cutoff when True.
    """
    result = []
    for well_id, wd in well_info.items():
        if wd.start_date > cutoff_date:
            continue

        actual_end = min(wd.end_date, cutoff_date)
        raw = wd._raw_data

        days, rates = _prepare_days_rates(
            raw, wd.start_date, actual_end, data_type, cutoff_date,
            extrapolate_injection_rates,
        )
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


def _prepare_days_rates(
    well_data: pd.DataFrame,
    start_date: date,
    end_date: date,
    data_type: str,
    cutoff_date: date,
    extrapolate_injection_rates: bool,
) -> Tuple[np.ndarray, np.ndarray]:
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
        return _monthly_fsp_days_rates(
            well_data, start_date, cutoff_date, extrapolate_injection_rates,
        )

    elif data_type == "injection_tool_data":
        return _injection_tool_days_rates(
            well_data, start_date, cutoff_date, extrapolate_injection_rates,
        )

    return np.array([]), np.array([])


def _next_month_start(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _monthly_rate_column(well_data: pd.DataFrame) -> Optional[str]:
    for col in ["InjectionRate(bbl/month)", "Injection Rate (bbl/month)", "MonthlyInjectionRate"]:
        if col in well_data.columns:
            return col
    return None


def _monthly_volumes_from_rows(well_data: pd.DataFrame, rate_col: str) -> Dict[Tuple[int, int], float]:
    """Last listed volume wins for a repeated (year, month)."""
    volume_by_month: Dict[Tuple[int, int], float] = {}
    ordered = well_data.sort_values(["Year", "Month"], kind="mergesort")
    for _, row in ordered.iterrows():
        try:
            year = int(row["Year"])
            month = int(row["Month"])
            volume = float(row[rate_col])
        except (ValueError, KeyError, TypeError):
            continue
        volume_by_month[(year, month)] = volume
    return volume_by_month


def _monthly_step_series(
    volume_by_month: Dict[Tuple[int, int], float],
    start_date: date,
    cutoff_date: date,
    extrapolate_injection_rates: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """MATLAB SpreadsheetStrings2WellData mapped onto Python start-of-interval days.

    Gap months are rate 0. Consecutive listed months with the same volume keep the
    first month's bbl/day. When extrapolate_injection_rates is False, a trailing 0
    starts at the month after the last data month.
    """
    included = {
        key: volume
        for key, volume in volume_by_month.items()
        if date(key[0], key[1], 1) <= cutoff_date
    }
    if not included:
        return np.array([]), np.array([])

    last_data_key = max(included)
    last_data_month = date(last_data_key[0], last_data_key[1], 1)
    current = date(min(included)[0], min(included)[1], 1)

    days_list: List[float] = []
    rates_list: List[float] = []
    previous_rate: Optional[float] = None
    previous_volume: Optional[float] = None
    previous_had_data = False

    def emit(step_date: date, rate: float) -> None:
        nonlocal previous_rate
        if step_date > cutoff_date:
            return
        step_date = max(step_date, start_date)
        if previous_rate is None or rate != previous_rate:
            days_list.append(float((step_date - start_date).days + 1))
            rates_list.append(rate)
            previous_rate = rate

    while current <= last_data_month:
        key = (current.year, current.month)
        if key in included:
            volume = float(included[key])
            rate = volume / _last_day_of_month(current.year, current.month)
            if not (previous_had_data and volume == previous_volume):
                emit(current, rate)
            previous_had_data = True
            previous_volume = volume
        else:
            emit(current, 0.0)
            previous_had_data = False
            previous_volume = None
        current = _next_month_start(current)

    if not extrapolate_injection_rates:
        shut_in = _next_month_start(last_data_month)
        if shut_in <= cutoff_date:
            emit(shut_in, 0.0)

    if not days_list:
        return np.array([]), np.array([])
    return np.asarray(days_list, dtype=float), np.asarray(rates_list, dtype=float)


def _monthly_fsp_days_rates(
    well_data: pd.DataFrame,
    start_date: date,
    cutoff_date: date,
    extrapolate_injection_rates: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert monthly injection volumes to MATLAB-style step-change days/rates."""
    rate_col = _monthly_rate_column(well_data)
    if rate_col is None:
        return np.array([]), np.array([])
    volume_by_month = _monthly_volumes_from_rows(well_data, rate_col)
    return _monthly_step_series(
        volume_by_month, start_date, cutoff_date, extrapolate_injection_rates,
    )


def _injection_tool_volume_column(well_data: pd.DataFrame) -> Optional[str]:
    for col in ["Monthly Injection Volume (BBLs)", "Annual Injection Volume (BBLs)",
                "Injection Volume (BBL)", "BPD"]:
        if col in well_data.columns:
            return col
    return None


def _injection_tool_days_rates(
    well_data: pd.DataFrame,
    start_date: date,
    cutoff_date: date,
    extrapolate_injection_rates: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert injection tool data to days/rates arrays."""
    dates = _parse_dates_column(well_data["Date of Injection"])
    rate_col = _injection_tool_volume_column(well_data)
    if rate_col is None:
        return np.array([]), np.array([])

    volumes = pd.to_numeric(well_data[rate_col], errors="coerce")
    is_monthly = "Monthly" in rate_col

    if is_monthly:
        dated_volumes = sorted(
            (
                (injection_date, float(volume))
                for injection_date, volume in zip(dates, volumes)
                if pd.notna(volume) and date(injection_date.year, injection_date.month, 1) <= cutoff_date
            ),
            key=lambda item: item[0],
        )
        volume_by_month: Dict[Tuple[int, int], float] = {}
        for injection_date, volume in dated_volumes:
            volume_by_month[(injection_date.year, injection_date.month)] = volume
        return _monthly_step_series(
            volume_by_month, start_date, cutoff_date, extrapolate_injection_rates,
        )

    days_list = []
    rates_list = []
    for injection_date, volume in zip(dates, volumes):
        if pd.isna(volume) or injection_date > cutoff_date:
            continue
        days_list.append(float((injection_date - start_date).days + 1))
        rates_list.append(float(volume) / 365.0)

    if not days_list:
        return np.array([]), np.array([])
    order = np.argsort(days_list)
    return np.array(days_list)[order], np.array(rates_list)[order]


def _d3_rate_row(well_id: str, step_date: date, rate: float) -> dict:
    return {
        "WellID": well_id,
        "date": step_date.strftime("%Y-%m-%d"),
        "rate_bbl_day": rate,
        "InjectionRate(bbl/day)": rate,
        "Timestamp": float(datetime(step_date.year, step_date.month, step_date.day).timestamp() * 1000.0),
    }


def _d3_rows_from_monthly_volumes(
    well_id: str,
    volume_by_month: Dict[Tuple[int, int], float],
) -> List[dict]:
    """Graph series: gap zeros between first and last data month; no tail past last data."""
    if not volume_by_month:
        return []
    last_key = max(volume_by_month)
    last_day = date(last_key[0], last_key[1], _last_day_of_month(last_key[0], last_key[1]))
    first_key = min(volume_by_month)
    start_date = date(first_key[0], first_key[1], 1)
    days, rates = _monthly_step_series(
        volume_by_month, start_date, last_day, extrapolate_injection_rates=True,
    )
    rows = []
    for day_offset, rate in zip(days, rates):
        step_date = start_date + timedelta(days=int(day_offset) - 1)
        rows.append(_d3_rate_row(well_id, step_date, rate))
    return rows


def injection_rate_data_to_d3_bbl_day(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Convert injection well DataFrame to D3-compatible time-series (bbl/day).

    Port of Julia injection_rate_data_to_d3_bbl_day.
    Returns both legacy Python columns and Julia-compatible graph columns.
    Monthly series use the MATLAB reconstruction (gap zeros, same-volume merge).
    """
    rows = []

    if data_type == "annual_fsp":
        for _, row in df.iterrows():
            well_id = str(row["WellID"])
            rate = float(row["InjectionRate(bbl/day)"])
            start_year = int(row["StartYear"])
            end_year = int(row["EndYear"])
            for year in range(start_year, end_year):
                rows.append(_d3_rate_row(well_id, date(year, 1, 1), rate))
            rows.append(_d3_rate_row(well_id, date(end_year, 1, 1), 0.0))

    elif data_type == "monthly_fsp":
        rate_col = _monthly_rate_column(df)
        if rate_col is None:
            return pd.DataFrame(rows)
        for well_id, group in df.groupby(df["WellID"].astype(str), sort=False):
            volume_by_month = _monthly_volumes_from_rows(group, rate_col)
            rows.extend(_d3_rows_from_monthly_volumes(str(well_id), volume_by_month))

    elif data_type == "injection_tool_data":
        well_id_col = "API Number" if "API Number" in df.columns else "UWI"
        rate_col = _injection_tool_volume_column(df)
        if rate_col is None:
            return pd.DataFrame(rows)
        is_monthly = "Monthly" in rate_col
        for well_id, group in df.groupby(df[well_id_col].astype(str), sort=False):
            if is_monthly:
                dates = _parse_dates_column(group["Date of Injection"])
                volumes = pd.to_numeric(group[rate_col], errors="coerce")
                dated_volumes = sorted(
                    (
                        (injection_date, float(volume))
                        for injection_date, volume in zip(dates, volumes)
                        if pd.notna(volume)
                    ),
                    key=lambda item: item[0],
                )
                volume_by_month: Dict[Tuple[int, int], float] = {}
                for injection_date, volume in dated_volumes:
                    volume_by_month[(injection_date.year, injection_date.month)] = volume
                rows.extend(_d3_rows_from_monthly_volumes(str(well_id), volume_by_month))
            else:
                dates = _parse_dates_column(group["Date of Injection"])
                volumes = pd.to_numeric(group[rate_col], errors="coerce")
                for injection_date, volume in zip(dates, volumes):
                    if pd.isna(volume):
                        continue
                    rows.append(_d3_rate_row(str(well_id), injection_date, float(volume) / 365.0))

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
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Cannot parse date: {s!r}")


def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]
