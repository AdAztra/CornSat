"""Phase 8 report figures -> reports/figures/phase8_*.png. Usage: uv run python scripts/make_phase8_figures.py"""
import glob
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cornsat import panel as P
from cornsat.config import DATA_INTERIM, DATA_PROCESSED, ROOT

warnings.filterwarnings("ignore")
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GREEN, ORANGE, GREY = "#2a78d6", "#1baf7a", "#eb6834", "#8a8984"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 1.6,
})


def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(p.name, p.stat().st_size // 1024, "KB")


h = pd.read_parquet(DATA_PROCESSED / "hls_state.parquet")
m = pd.read_parquet(DATA_PROCESSED / "modis_state.parquet")
m["end_date"] = pd.to_datetime(m["end_date"]).astype("datetime64[ns]")
mm = (m.groupby(["state", "end_date"], as_index=False)["ndvi_mean"].mean()
      .rename(columns={"ndvi_mean": "modis", "end_date": "mend"}).sort_values("mend"))

# matched pairs: every HLS composite with the nearest MODIS composite end date within 4 days
j = []
for st, g in h.groupby("state"):
    j.append(pd.merge_asof(g.sort_values("end_date"), mm[mm["state"] == st].drop(columns="state"),
                           left_on="end_date", right_on="mend", direction="nearest",
                           tolerance=pd.Timedelta(days=4)))
j = pd.concat(j).dropna(subset=["modis", "ndvi_mean"])
r = np.corrcoef(j["ndvi_mean"], j["modis"])[0, 1]

# --- 1. seasonal curves -------------------------------------------------------------------
cases = [("IA", 2020, "IA 2020 (derecho Aug 10)"), ("IL", 2023, "IL 2023 (dry summer)"),
         ("NE", 2024, "NE 2024"), ("IN", 2016, "IN 2016 (thin early coverage)")]
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
for ax, (st, yr, title) in zip(axes.flat, cases):
    hh = h[(h["state"] == st) & (h["year"] == yr)].sort_values("end_date")
    mo = m[(m["state"] == st) & (m["year"] == yr) & m["end_date"].dt.month.between(5, 10)]
    mo = mo.groupby("end_date", as_index=False)["ndvi_mean"].mean()
    ax.plot(mo["end_date"], mo["ndvi_mean"], color=BLUE, marker="o", ms=4, label="MODIS 250 m (S)")
    ax.plot(hh["end_date"], hh["ndvi_mean"], color=GREEN, lw=1.6, label="HLS 30 m (H)")
    lo = hh[hh["valid_frac"] < 0.5]
    ax.scatter(lo["end_date"], lo["ndvi_mean"], s=30, facecolor=SURF, edgecolor=GREEN, lw=1.4,
               zorder=3, label="HLS valid_frac < 0.5 (dropped)")
    if (st, yr) == ("IA", 2020):
        ax.axvline(pd.Timestamp("2020-08-10"), color=INK2, lw=1, ls="--")
        ax.annotate("derecho", (pd.Timestamp("2020-08-10"), 0.3), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.set_title(title, loc="left")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(pd.Timestamp(yr, 5, 10), pd.Timestamp(yr, 10, 5))
for ax in axes[:, 0]:
    ax.set_ylabel("state corn NDVI (composite end date)")
hnd, lbl = axes[0, 0].get_legend_handles_labels()
fig.suptitle("HLS and MODIS state corn NDVI follow the same seasonal curve", x=0.01, ha="left",
             fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.legend(hnd, lbl, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3, fontsize=8)
save(fig, "phase8_curves.png")

# --- 2. scatter -------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 5))
per = [((2016, 2017), "2016-17", ORANGE), ((2018, 2021), "2018-21", BLUE), ((2022, 2026), "2022-26", GREEN)]
for (a, b), lab, col in per:
    s = j[j["year"].between(a, b)]
    ax.scatter(s["modis"], s["ndvi_mean"], s=10, color=col, alpha=0.6, lw=0, label=f"{lab} (n={len(s)})")
ax.plot([0.15, 0.95], [0.15, 0.95], color=INK2, lw=1, ls="--")
ax.set_xlim(0.15, 0.95); ax.set_ylim(0.15, 0.95)
ax.set_xlabel("MODIS state NDVI"); ax.set_ylabel("HLS state NDVI")
ax.set_title(f"Matched composites: r = {r:.3f}, n = {len(j):,}", loc="left")
ax.text(0.17, 0.90, "dashed: 1:1. HLS is lower early in the season\n(bare soil between rows) and higher at peak",
        fontsize=8, color=INK2, va="top")
ax.legend(loc="lower right", fontsize=8, markerscale=2)
fig.tight_layout()
save(fig, "phase8_scatter.png")

# --- 3. coverage --------------------------------------------------------------------------------
sc = pd.concat([pd.read_parquet(f) for f in glob.glob(str(DATA_INTERIM / "hls" / "scenes" / "*" / "*.parquet"))])
gr = sc.groupby("granule_id").agg(read=("refl_read", "max"), season=("season", "first"), sensor=("sensor", "first"))
gr = gr[gr["read"]]
cnt = gr.groupby(["season", "sensor"]).size().unstack(fill_value=0)
vf = pd.concat([pd.read_parquet(DATA_PROCESSED / "hls_state_presample.parquet"), h]).groupby("year")["valid_frac"].mean()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
x = cnt.index.to_numpy()
ax.bar(x, cnt["L30"], color=BLUE, width=0.75, label="Landsat L30")
ax.bar(x, cnt.get("S30", 0), bottom=cnt["L30"], color=GREEN, width=0.75, label="Sentinel-2 S30",
       edgecolor=SURF, linewidth=1)
ax.axvline(2015.5, color=INK2, lw=1, ls="--")
ax.annotate("pre-sample\n(climatology only)", (2014, cnt.sum(axis=1).max() * 0.9), ha="center", fontsize=8, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([str(v)[2:] for v in x])
ax.set_ylabel("granules with reflectance read")
ax.set_title("Granules read per season (May 1-Sep 30)", loc="left")
ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 0.8))
ax = axes[1]
ax.plot(vf.index, vf.values, color=GREEN, marker="o", ms=5)
ax.axhline(0.5, color=INK2, lw=1, ls="--")
ax.annotate("composite gate 0.5", (2013, 0.51), fontsize=8, color=INK2)
for yv in (2016, 2017):
    ax.annotate(f"{vf[yv]:.2f}", (yv, vf[yv]), xytext=(4, -10), textcoords="offset points", fontsize=8)
ax.set_ylim(0.3, 1.0)
ax.set_xticks(vf.index); ax.set_xticklabels([str(v)[2:] for v in vf.index])
ax.set_ylabel("mean valid_frac of state composites")
ax.set_title("Clear-sky share of masked pixels", loc="left")
fig.tight_layout()
save(fig, "phase8_coverage.png")

# --- 4. RMSE by snapshot ----------------------------------------------------------------------
acc = pd.read_csv(REPORTS / "state_panel_accuracy.csv")
snaps = ["Jul01", "Jul15", "Aug01", "Aug15", "Sep01", "Sep15"]
sets = [("W+C", GREY, "W+C"), ("W+C+S", BLUE, "W+C+S (MODIS)"), ("W+C+H", GREEN, "W+C+H (HLS)")]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, sch, ttl in zip(axes, ["loso", "wf"], ["2016+ leave-one-season-out (10 seasons)",
                                                "2016+ walk-forward (7 test seasons, 2019-25)"]):
    a = acc[(acc["window"] == "2016+") & (acc["scheme"] == sch)].set_index(["set", "snapshot"])["rmse"]
    xs = np.arange(len(snaps))
    for s, col, lab in sets:
        y = [a[(s, sn)] for sn in snaps]
        ax.plot(xs, y, color=col, marker="o", ms=5, label=f"{lab}, all {a[(s, 'ALL')]:.2f}")
    ax.plot(xs, [a[("trend", sn)] for sn in snaps], color=INK2, lw=1, ls=":", label=f"trend only, all {a[('trend', 'ALL')]:.2f}")
    ax.plot(xs[3:], [a[("usda", sn)] for sn in snaps[3:]], color=ORANGE, lw=1.2, ls="--", marker="s", ms=4,
            label="USDA state forecast")
    ax.set_xticks(xs); ax.set_xticklabels(snaps)
    ax.set_title(ttl, loc="left")
    ax.legend(fontsize=7.5, loc="lower left")
axes[0].set_ylabel("RMSE of state yield surprise (bu/ac)")
axes[0].set_ylim(5.5, 14)
fig.suptitle("State panel RMSE by snapshot: H lowest in July-August, MODIS lowest at Sep 15", x=0.01, ha="left",
             fontweight="bold")
fig.tight_layout()
save(fig, "phase8_rmse_snapshots.png")

# --- 5. IA 2020 derecho -----------------------------------------------------------------------
fd = pd.read_parquet(DATA_PROCESSED / "features_daily.parquet")
ev = pd.Timestamp("2020-08-10")
fig, ax = plt.subplots(figsize=(9, 3.8))
for feat, col, lab, ls in [("ndvi_z", BLUE, "S: MODIS ndvi_z", "-"), ("hls_ndvi_z", GREEN, "H: HLS hls_ndvi_z", "-"),
                           ("ge", ORANGE, "C: good/excellent rating", "--")]:
    z = P.prior_season_z(fd, "IA", 2020, feat, min_prior=3, first_prior=2016)
    z = z[(z["date"] >= "2020-07-01") & (z["date"] <= "2020-10-15")]
    ax.plot(z["date"], z["z"], color=col, ls=ls, label=lab)
ev_t = pd.read_csv(REPORTS / "state_panel_events.csv")
ev_t = ev_t[(ev_t["state"] == "IA") & (ev_t["season"] == 2020) & ev_t["feature"].isin(["ndvi_z", "hls_ndvi_z", "ge"])]
for _, rr in ev_t.iterrows():
    d = pd.Timestamp(rr["first_cross_after_event"])
    col = {"ndvi_z": BLUE, "hls_ndvi_z": GREEN, "ge": ORANGE}[rr["feature"]]
    ax.scatter([d], [-1], s=50, color=col, edgecolor=SURF, lw=1.5, zorder=4)
    off = {"ndvi_z": (-30, 8), "hls_ndvi_z": (6, -14), "ge": (3, 6)}[rr["feature"]]
    ax.annotate(f"+{int(rr['days_after_event'])} d", (d, -1), xytext=off, textcoords="offset points",
                fontsize=8, color=INK2)
ax.axvline(ev, color=INK2, lw=1, ls="--")
ax.annotate("derecho Aug 10", (ev, 1.4), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2)
ax.axhline(-1, color=INK2, lw=0.8, ls=":")
ax.axhline(0, color=GRID, lw=1)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.set_ylabel("z vs same day 2016-2019 (as-of values)")
ax.set_title("Iowa 2020: the rating reacted 8 days after the derecho, both satellites 24-25 days after", loc="left")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
save(fig, "phase8_derecho_ia2020.png")
