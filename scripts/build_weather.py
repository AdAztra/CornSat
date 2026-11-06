"""Build weather features and print a US summary by year."""

import pandas as pd

from cornsat.config import YEARS
from cornsat.weather import build_weather


def main() -> None:
    w = build_weather()
    us = w[w.state == "US"]
    rows = []
    for y in YEARS:
        d = us[us.date.dt.year == y]
        jul = d[d.date.dt.month == 7]
        jja = d[d.date.dt.month.isin([6, 7, 8])]
        rows.append(
            {
                "year": y,
                "jul_tmax_mean": jul.tmax_c.mean(),
                "jul_edd30": jul.edd30.sum(),
                "jja_precip_mm": jja.precip_mm.sum(),
                "jja_days": len(jja),
            }
        )
    tbl = pd.DataFrame(rows).set_index("year").round(1)
    print(tbl.to_string())
    print("rows:", len(w), "| last date:", w.date.max().date())


if __name__ == "__main__":
    main()
