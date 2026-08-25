"""Unit tests for MATLAB-style monthly injection-rate reconstruction."""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from fsp.io.wells import (
    coerce_extrapolate_injection_rates,
    injection_rate_data_to_d3_bbl_day,
    normalize_wells_to_well_data,
    preprocess_well_data,
    resolve_extrapolate_injection_rates,
)


def _monthly_frame(rows):
    """rows: list of (well_id, year, month, volume)."""
    return pd.DataFrame(
        [
            {
                "WellID": well_id,
                "Latitude(WGS84)": 31.2,
                "Longitude(WGS84)": -103.7,
                "Year": year,
                "Month": month,
                "InjectionRate(bbl/month)": volume,
            }
            for well_id, year, month, volume in rows
        ]
    )


def _normalize(rows, cutoff, extrapolate):
    frame = _monthly_frame(rows)
    well_info = preprocess_well_data(frame, "monthly_fsp")
    return normalize_wells_to_well_data(
        well_info, "monthly_fsp", cutoff, extrapolate_injection_rates=extrapolate,
    )


class TestCoerceExtrapolateFlag:
    def test_explicit_true_values(self):
        assert coerce_extrapolate_injection_rates(True) is True
        assert coerce_extrapolate_injection_rates(1) is True
        assert coerce_extrapolate_injection_rates("true") is True
        assert coerce_extrapolate_injection_rates("YES") is True
        assert coerce_extrapolate_injection_rates("1") is True

    def test_missing_and_false_are_off(self):
        assert coerce_extrapolate_injection_rates(None) is False
        assert coerce_extrapolate_injection_rates(False) is False
        assert coerce_extrapolate_injection_rates(0) is False
        assert coerce_extrapolate_injection_rates("false") is False
        assert coerce_extrapolate_injection_rates("") is False

    def test_resolve_uses_first_non_none_step_and_does_not_default_true(self):
        values = {4: None, 3: None, 0: None}
        assert resolve_extrapolate_injection_rates(
            lambda step, _name: values.get(step), 4,
        ) is False

        values[3] = "false"
        assert resolve_extrapolate_injection_rates(
            lambda step, _name: values.get(step), 4,
        ) is False

        values[4] = "true"
        assert resolve_extrapolate_injection_rates(
            lambda step, _name: values.get(step), 4,
        ) is True
        assert resolve_extrapolate_injection_rates(
            lambda step, _name: values.get(step), 4, "monthly_fsp",
        ) is True

    def test_resolve_ignores_flag_unless_monthly_fsp(self):
        get_param = lambda step, _name: "true"
        assert resolve_extrapolate_injection_rates(get_param, 3, "annual_fsp") is False
        assert resolve_extrapolate_injection_rates(
            get_param, 3, "injection_tool_data",
        ) is False
        assert resolve_extrapolate_injection_rates(get_param, 3, "monthly_fsp") is True


class TestMonthlyRateSeries:
    def test_gap_month_is_zero_not_carried_forward(self):
        wells = _normalize(
            [("W1", 2018, 1, 31000.0), ("W1", 2018, 3, 31000.0)],
            date(2018, 12, 31),
            extrapolate=False,
        )
        well = wells[0]
        # Jan 1 = day 1, Feb 1 = day 32, Mar 1 = day 60
        assert well.days[0] == pytest.approx(1.0)
        assert well.rates[0] == pytest.approx(31000.0 / 31.0)
        feb_index = int(np.where(well.days == 32.0)[0][0])
        assert well.rates[feb_index] == pytest.approx(0.0)
        mar_index = int(np.where(well.days == 60.0)[0][0])
        assert well.rates[mar_index] == pytest.approx(31000.0 / 31.0)

    def test_same_volume_keeps_january_daily_rate(self):
        wells = _normalize(
            [("W1", 2018, 1, 31000.0), ("W1", 2018, 2, 31000.0)],
            date(2018, 12, 31),
            extrapolate=False,
        )
        well = wells[0]
        assert well.days[0] == pytest.approx(1.0)
        assert well.rates[0] == pytest.approx(31000.0 / 31.0)
        assert 32.0 not in well.days
        assert 31000.0 / 28.0 not in well.rates

    def test_extrapolate_on_has_no_trailing_zero(self):
        wells = _normalize(
            [("W1", 2018, 1, 31000.0)],
            date(2018, 12, 31),
            extrapolate=True,
        )
        well = wells[0]
        assert well.rates[-1] == pytest.approx(31000.0 / 31.0)
        assert well.rates[-1] != pytest.approx(0.0)

    def test_extrapolate_off_shuts_in_after_last_data_month(self):
        wells = _normalize(
            [("W1", 2018, 1, 31000.0)],
            date(2018, 12, 31),
            extrapolate=False,
        )
        well = wells[0]
        assert well.days[-1] == pytest.approx(32.0)
        assert well.rates[-1] == pytest.approx(0.0)

    def test_default_normalize_does_not_extrapolate(self):
        frame = _monthly_frame([("W1", 2018, 1, 31000.0)])
        well_info = preprocess_well_data(frame, "monthly_fsp")
        wells = normalize_wells_to_well_data(well_info, "monthly_fsp", date(2018, 12, 31))
        assert wells[0].rates[-1] == pytest.approx(0.0)

    def test_months_after_cutoff_are_dropped(self):
        wells = _normalize(
            [("W1", 2018, 1, 31000.0), ("W1", 2019, 1, 99999.0)],
            date(2018, 12, 31),
            extrapolate=True,
        )
        well = wells[0]
        assert well.rates[0] == pytest.approx(31000.0 / 31.0)
        assert 99999.0 / 31.0 not in well.rates

    def test_leap_year_february_uses_29_days(self):
        wells = _normalize(
            [("W1", 2020, 2, 2900.0)],
            date(2020, 12, 31),
            extrapolate=True,
        )
        well = wells[0]
        assert well.rates[0] == pytest.approx(100.0)


class TestInjectionRateGraph:
    def test_annual_graph_drops_to_zero_on_end_year(self):
        frame = pd.DataFrame(
            [
                {
                    "WellID": "W1",
                    "Latitude(WGS84)": 31.0,
                    "Longitude(WGS84)": -103.0,
                    "StartYear": 2018,
                    "EndYear": 2020,
                    "InjectionRate(bbl/day)": 1000.0,
                }
            ]
        )
        graph = injection_rate_data_to_d3_bbl_day(frame, "annual_fsp")
        by_date = dict(zip(graph["date"], graph["rate_bbl_day"]))
        assert by_date["2018-01-01"] == pytest.approx(1000.0)
        assert by_date["2019-01-01"] == pytest.approx(1000.0)
        assert by_date["2020-01-01"] == pytest.approx(0.0)
        assert "2021-01-01" not in by_date

    def test_monthly_graph_inserts_gap_zeros(self):
        frame = _monthly_frame(
            [("W1", 2018, 1, 31000.0), ("W1", 2018, 3, 31000.0)]
        )
        graph = injection_rate_data_to_d3_bbl_day(frame, "monthly_fsp")
        by_date = dict(zip(graph["date"], graph["rate_bbl_day"]))
        assert by_date["2018-01-01"] == pytest.approx(31000.0 / 31.0)
        assert by_date["2018-02-01"] == pytest.approx(0.0)
        assert by_date["2018-03-01"] == pytest.approx(31000.0 / 31.0)


class TestNonMonthlyFormatsIgnoreExtrapolate:
    def test_injection_tool_monthly_shuts_in_even_if_flag_true(self):
        frame = pd.DataFrame(
            [
                {
                    "API Number": "W1",
                    "Surface Latitude": 31.2,
                    "Surface Longitude": -103.7,
                    "Date of Injection": "2018-01-15",
                    "Monthly Injection Volume (BBLs)": 31000.0,
                }
            ]
        )
        well_info = preprocess_well_data(frame, "injection_tool_data")
        wells = normalize_wells_to_well_data(
            well_info, "injection_tool_data", date(2018, 12, 31),
            extrapolate_injection_rates=True,
        )
        well = wells[0]
        # Start date is 2018-01-15, so Feb 1 shut-in is day 18.
        assert well.days[-1] == pytest.approx(18.0)
        assert well.rates[-1] == pytest.approx(0.0)

    def test_annual_uses_end_year_not_portal_flag(self):
        frame = pd.DataFrame(
            [
                {
                    "WellID": "W1",
                    "Latitude(WGS84)": 31.0,
                    "Longitude(WGS84)": -103.0,
                    "StartYear": 2018,
                    "EndYear": 2019,
                    "InjectionRate(bbl/day)": 1000.0,
                }
            ]
        )
        well_info = preprocess_well_data(frame, "annual_fsp")
        wells = normalize_wells_to_well_data(
            well_info, "annual_fsp", date(2020, 12, 31),
            extrapolate_injection_rates=True,
        )
        well = wells[0]
        assert well.rates[0] == pytest.approx(1000.0)
        assert well.rates[-1] == pytest.approx(0.0)
        assert well.end_date == date(2018, 12, 31)
