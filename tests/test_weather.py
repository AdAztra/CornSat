import numpy as np
import pandas as pd

from cornsat.weather import add_features, compute_edd30, compute_gdd


def test_gdd_edd_examples():
    assert compute_gdd([35], [20]).iloc[0] == 15  # (30+20)/2-10
    assert compute_edd30([35]).iloc[0] == 5
    assert compute_gdd([8], [2]).iloc[0] == 0
    assert compute_edd30([8]).iloc[0] == 0
    assert compute_gdd([25], [15]).iloc[0] == 10


def _frame(start, end, tmax=25.0, tmin=15.0, precip=1.0):
    d = pd.date_range(start, end)
    return pd.DataFrame({"date": d, "tmax_c": tmax, "tmin_c": tmin, "precip_mm": precip})


def test_cumulative_reset_and_season_bounds():
    f = add_features(_frame("2010-03-30", "2011-04-02"))

    def row(s):
        return f[f.date == s].iloc[0]

    assert np.isnan(row("2010-03-31").gdd_cum)
    assert row("2010-04-01").gdd_cum == 10
    assert row("2010-04-03").gdd_cum == 30
    assert np.isnan(row("2010-11-01").gdd_cum)
    assert row("2011-04-01").gdd_cum == 10  # reset
    assert row("2011-04-01").precip_cum == 1


def test_missing_is_nan():
    df = _frame("2010-06-01", "2010-06-03")
    df.loc[1, "tmax_c"] = np.nan
    f = add_features(df)
    assert np.isnan(f.gdd[1]) and np.isnan(f.edd30[1])
    assert not np.isnan(f.gdd[0])
    # -999 sentinel replaced by NaN, as fetch_point does
    raw = _frame("2010-06-01", "2010-06-02", tmax=-999.0).replace(-999, np.nan)
    assert add_features(raw).gdd.isna().all()


def test_available_date_after_date():
    f = add_features(_frame("2010-06-01", "2010-06-05"))
    assert (f.available_date > f.date).all()
    assert (f.available_date - f.date == pd.Timedelta(days=1)).all()
