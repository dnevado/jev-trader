import numpy as np
import pandas as pd
import pytest

from jevbt.features.fundamentals import build_user_prompt, compute_ratios, pe_ratio, point_in_time


def statements():
    idx = pd.to_datetime(["2022-09-24", "2023-09-30", "2024-09-28"]).rename("date")
    acc = pd.to_datetime(["2022-10-28 18:00", "2023-11-02 18:00", "2024-11-01 06:00"])
    income = pd.DataFrame({
        "fiscal_year": ["2022", "2023", "2024"], "accepted_date": acc,
        "revenue": [100.0, 110.0, 99.0], "gross_profit": [40.0, 44.0, 40.0],
        "operating_income": [30.0, 33.0, 20.0], "net_income": [25.0, 27.0, 15.0], "eps_diluted": [1.0, 1.1, 0.6],
    }, index=idx)
    balance = pd.DataFrame({"accepted_date": acc, "total_debt": [50.0, 60.0, 70.0],
                            "total_stockholders_equity": [100.0, 120.0, -5.0]}, index=idx)
    # Cash flow of FY2024 accepted a day later than the other two statements.
    cashflow = pd.DataFrame({"accepted_date": acc + pd.to_timedelta([0, 0, 1], unit="D"),
                             "free_cash_flow": [20.0, 22.0, 9.9]}, index=idx)
    return income, balance, cashflow


def test_compute_ratios():
    r = compute_ratios(*statements())
    assert list(r["fiscal_year"]) == ["2022", "2023", "2024"]
    assert np.isnan(r["revenue_growth_yoy"].iloc[0])
    assert r["revenue_growth_yoy"].iloc[1] == pytest.approx(0.10)
    assert r["revenue_growth_yoy"].iloc[2] == pytest.approx(-0.10)
    assert r["gross_margin"].iloc[1] == pytest.approx(0.40)
    assert r["operating_margin"].iloc[2] == pytest.approx(20 / 99)
    assert r["fcf_margin"].iloc[2] == pytest.approx(0.10)
    assert r["debt_to_equity"].iloc[1] == pytest.approx(0.5)
    assert np.isnan(r["debt_to_equity"].iloc[2])  # negative equity
    assert r["known_at"].iloc[2] == pd.Timestamp("2024-11-02 06:00")


def test_point_in_time_uses_publication_not_period_end():
    r = compute_ratios(*statements())
    # After FY2024's period end but before its statements are all public → still FY2022-2023.
    assert list(point_in_time(r, "2024-11-01")["fiscal_year"]) == ["2022", "2023"]
    assert list(point_in_time(r, "2024-11-02")["fiscal_year"]) == ["2022", "2023"]
    assert list(point_in_time(r, "2024-11-03")["fiscal_year"]) == ["2022", "2023", "2024"]
    assert point_in_time(r, "2022-10-28").empty
    assert list(point_in_time(r, "2025-01-01", years=2)["fiscal_year"]) == ["2023", "2024"]


def test_pe_ratio():
    assert pe_ratio(30.0, 1.5) == pytest.approx(20.0)
    assert np.isnan(pe_ratio(30.0, -1.0))
    assert np.isnan(pe_ratio(30.0, 0.0))
    assert np.isnan(pe_ratio(30.0, float("nan")))


def test_user_prompt_is_anchored_to_filing_date():
    r = compute_ratios(*statements())
    prompt = build_user_prompt("the company", point_in_time(r, "2025-01-01"))
    assert "Analysis date: 2024-11-02" in prompt
    assert "FY2024" in prompt and "FY2022" in prompt
    assert "no data" in prompt  # growth of the first year, D/E with negative equity
    assert "not available" in prompt
