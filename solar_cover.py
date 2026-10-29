"""
Parametric Rooftop Solar Shortfall Insurance — Gujarat
=======================================================

Prices a parametric cover for household rooftop solar that pays out when
satellite-derived monthly generation falls below a trigger.

Pipeline
  1. Fetch monthly irradiance + rainfall from NASA POWER for 5 Gujarat cities
  2. Convert irradiance -> modelled monthly generation for a 3 kW home system
  3. Define payout: trigger / exit on generation ratio vs monthly normal
  4. Burn analysis on the full history
  5. Monte Carlo pricing (fitted distributions + cross-city correlation)
  6. Premium = (expected loss + risk load) / (1 - expense ratio)
  7. Return periods, trigger sensitivity
  8. Portfolio: loss ratios, 1-in-200 capital
  9. Correlation with a rainfall-deficit crop cover (diversification test)

Run:   python solar_cover.py
Out:   outputs/results.json, outputs/summary.txt, outputs/*.png
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Configuration (all assumptions in one place — explain these in interviews)
# ----------------------------------------------------------------------------
CITIES = {
    "Ahmedabad": (23.02, 72.57),
    "Surat": (21.17, 72.83),
    "Rajkot": (22.30, 70.80),
    "Vadodara": (22.31, 73.18),
    "Bhuj": (23.25, 69.67),
}
START_YEAR, END_YEAR = 1984, 2024

SYSTEM_KW = 3.0            # typical household system
PERFORMANCE_RATIO = 0.77   # losses: inverter, temperature, wiring, soiling
TARIFF_INR = 6.0           # ₹/unit avoided — set to your DISCOM's rate

TRIGGER = 0.85             # pay when month's generation < 85% of normal
EXIT = 0.60                # max payout reached at 60% of normal
TRIGGER_GRID = [0.80, 0.85, 0.90]

RISK_LOAD = 0.25           # loading = 25% of annual loss std-dev
EXPENSE_RATIO = 0.20       # 20% of gross premium for distribution/admin
N_SIMS = 100_000
N_POLICIES = 10_000        # illustrative portfolio, equal split across cities
SEED = 42

# Rainfall-deficit crop cover (for diversification test)
KHARIF_MONTHS = [6, 7, 8, 9]
RAIN_TRIGGER_PCTL = 30
RAIN_EXIT_PCTL = 10

DATA_DIR = "data"
OUT_DIR = "outputs"
DAYS_IN_MONTH = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
API = "https://power.larc.nasa.gov/api/temporal/monthly/point"


def fetch_city(name, lat, lon):
    """Monthly all-sky GHI (kWh/m²/day) and rainfall (mm/day). Cached to CSV."""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,PRECTOTCORR",
        "community": "RE",
        "latitude": lat,
        "longitude": lon,
        "start": START_YEAR,
        "end": END_YEAR,
        "format": "JSON",
    }
    r = requests.get(API, params=params, timeout=90)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    rows = []
    for key, ghi in p["ALLSKY_SFC_SW_DWN"].items():
        year, month = int(key[:4]), int(key[4:])
        if month == 13:  # annual average row
            continue
        rows.append({
            "year": year, "month": month,
            "ghi": ghi, "rain": p["PRECTOTCORR"].get(key, np.nan),
        })
    df = pd.DataFrame(rows).replace(-999, np.nan)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    time.sleep(1)  # be polite to the API
    return df


def load_all():
    frames = []
    for name, (lat, lon) in CITIES.items():
        print(f"  loading {name} ...")
        df = fetch_city(name, lat, lon)
        df["city"] = name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True).dropna(subset=["ghi"])
    return data


# ----------------------------------------------------------------------------
# 2. Generation model + 3. payout
# ----------------------------------------------------------------------------
def add_generation(df):
    days = DAYS_IN_MONTH[df["month"].values - 1]
    df["gen_kwh"] = df["ghi"] * days * SYSTEM_KW * PERFORMANCE_RATIO
    normal = df.groupby(["city", "month"])["gen_kwh"].transform("mean")
    df["normal_kwh"] = normal
    df["ratio"] = df["gen_kwh"] / df["normal_kwh"]
    return df


def payout_from_ratio(ratio, normal_kwh, trigger=TRIGGER, exit_=EXIT):
    """Linear payout between trigger and exit, in ₹."""
    shortfall_kwh = np.clip(trigger - ratio, 0, trigger - exit_) * normal_kwh
    return shortfall_kwh * TARIFF_INR


def max_annual_payout(df):
    """Sum insured = payout if every month hits the exit."""
    n = df.groupby(["city", "month"])["normal_kwh"].first()
    return float(((TRIGGER - EXIT) * n * TARIFF_INR).groupby("city").sum().mean())


# ----------------------------------------------------------------------------
# 4. Burn analysis
# ----------------------------------------------------------------------------
def burn_analysis(df, trigger=TRIGGER):
    df = df.copy()
    df["payout"] = payout_from_ratio(df["ratio"], df["normal_kwh"], trigger)
    annual = df.groupby(["city", "year"])["payout"].sum().unstack("city")
    monthly_freq = (df.assign(hit=df["payout"] > 0)
                      .groupby(["city", "month"])["hit"].mean().unstack("month"))
    return df, annual, monthly_freq


# ----------------------------------------------------------------------------
# 5. Monte Carlo
# ----------------------------------------------------------------------------
def monte_carlo(df, trigger=TRIGGER, n_sims=N_SIMS, seed=SEED):
    """
    For each calendar month: fit a normal distribution to each city's ratio,
    estimate the cross-city correlation, draw correlated ratios.
    Months are simulated independently (monthly anomalies show weak serial
    correlation; check this with the printed lag-1 autocorrelation).
    Returns simulated annual payouts, shape (n_sims, n_cities).
    """
    rng = np.random.default_rng(seed)
    cities = list(CITIES)
    annual = np.zeros((n_sims, len(cities)))
    fit_info = {}
    for m in range(1, 13):
        sub = df[df["month"] == m].pivot(index="year", columns="city",
                                         values="ratio")[cities].dropna()
        mu, sd = sub.mean().values, sub.std(ddof=1).values
        corr = np.nan_to_num(sub.corr().values, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        cov = corr * np.outer(sd, sd)
        draws = rng.multivariate_normal(mu, cov, size=n_sims, method="eigh")
        normal = (df[df["month"] == m].groupby("city")["normal_kwh"]
                  .first()[cities].values)
        annual += payout_from_ratio(draws, normal, trigger)
        # normality check per city-month (Shapiro-Wilk p-values)
        fit_info[m] = {c: float(stats.shapiro(sub[c]).pvalue) for c in cities}
    return annual, fit_info


def price(sim_annual):
    """Per-policy premium, pooled across cities (equal weight)."""
    el = float(sim_annual.mean())
    sd = float(sim_annual.std())
    gross = (el + RISK_LOAD * sd) / (1 - EXPENSE_RATIO)
    return el, sd, gross


def return_periods(sim_per_policy, periods=(10, 25, 100)):
    return {f"1_in_{T}": float(np.quantile(sim_per_policy, 1 - 1 / T))
            for T in periods}


# ----------------------------------------------------------------------------
# 8. Portfolio
# ----------------------------------------------------------------------------
def portfolio(sim_annual, annual_hist, gross_premium):
    per_city = N_POLICIES / sim_annual.shape[1]
    port_loss = sim_annual.sum(axis=1) * per_city
    port_premium = gross_premium * N_POLICIES
    var_995 = float(np.quantile(port_loss, 0.995))      # 1-in-200
    capital = max(0.0, var_995 - port_premium * (1 - EXPENSE_RATIO))
    hist_lr = (annual_hist.mean(axis=1) / gross_premium)
    return {
        "policies": N_POLICIES,
        "annual_premium_inr": port_premium,
        "expected_loss_inr": float(port_loss.mean()),
        "expected_loss_ratio": float(port_loss.mean() / port_premium),
        "loss_1_in_200_inr": var_995,
        "capital_needed_1_in_200_inr": capital,
        "prob_loss_ratio_above_100pct": float((port_loss > port_premium).mean()),
        "hist_loss_ratio_mean": float(hist_lr.mean()),
        "hist_loss_ratio_worst": float(hist_lr.max()),
        "hist_worst_year": int(hist_lr.idxmax()),
    }, port_loss


# ----------------------------------------------------------------------------
# 9. Rainfall-deficit cover correlation
# ----------------------------------------------------------------------------
def rainfall_cover(df):
    k = df[df["month"].isin(KHARIF_MONTHS)].copy()
    k["rain_mm"] = k["rain"] * DAYS_IN_MONTH[k["month"].values - 1]
    seasonal = k.groupby(["city", "year"])["rain_mm"].sum().unstack("city")
    trig = seasonal.quantile(RAIN_TRIGGER_PCTL / 100)
    ext = seasonal.quantile(RAIN_EXIT_PCTL / 100)
    # payout fraction of sum insured, 0..1
    frac = ((trig - seasonal) / (trig - ext)).clip(0, 1)
    return seasonal, frac


def diversification(annual_solar, rain_frac):
    out = {}
    for c in annual_solar.columns:
        s = annual_solar[c].dropna()
        r = rain_frac[c].reindex(s.index).dropna()
        s = s.reindex(r.index)
        pear = stats.pearsonr(s, r)
        spear = stats.spearmanr(s, r)
        out[c] = {"pearson_r": float(pear.statistic), "pearson_p": float(pear.pvalue),
                  "spearman_rho": float(spear.statistic),
                  "spearman_p": float(spear.pvalue)}
    # pooled: average across cities per year
    s = annual_solar.mean(axis=1)
    r = rain_frac.mean(axis=1).reindex(s.index)
    m = s.notna() & r.notna()
    out["Gujarat_pooled"] = {
        "pearson_r": float(stats.pearsonr(s[m], r[m]).statistic),
        "pearson_p": float(stats.pearsonr(s[m], r[m]).pvalue),
        "spearman_rho": float(stats.spearmanr(s[m], r[m]).statistic),
        "spearman_p": float(stats.spearmanr(s[m], r[m]).pvalue),
    }
    return out


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plots(monthly_freq, annual_hist, port_loss, port_premium, sens, rain_frac):
    os.makedirs(OUT_DIR, exist_ok=True)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(monthly_freq.values * 100, cmap="Blues", aspect="auto")
    ax.set_xticks(range(12), months)
    ax.set_yticks(range(len(monthly_freq)), monthly_freq.index)
    for i in range(monthly_freq.shape[0]):
        for j in range(12):
            ax.text(j, i, f"{monthly_freq.values[i, j] * 100:.0f}",
                    ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="% of years with a payout")
    ax.set_title(f"Trigger frequency by month (trigger = {TRIGGER:.0%} of normal)")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/1_trigger_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    annual_hist.mean(axis=1).plot.bar(ax=ax, color="#3b6ea5")
    ax.set_ylabel("₹ payout per 3 kW policy")
    ax.set_title("Historical burn: average annual payout across cities")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/2_burn_history.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(port_loss / 1e6, bins=80, color="#3b6ea5")
    ax.axvline(port_premium / 1e6, color="#c0392b", ls="--", label="Premium income")
    ax.axvline(np.quantile(port_loss, 0.995) / 1e6, color="black", ls=":",
               label="1-in-200 loss")
    ax.set_xlabel("Portfolio annual payout (₹ million)")
    ax.set_title(f"Simulated portfolio losses ({N_POLICIES:,} policies)")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/3_portfolio_losses.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    t = [s["trigger"] for s in sens]
    ax.plot(t, [s["gross_premium_inr"] for s in sens], "o-", label="Gross premium")
    ax.plot(t, [s["expected_loss_inr"] for s in sens], "s--", label="Expected loss")
    ax.set_xlabel("Trigger (fraction of normal generation)")
    ax.set_ylabel("₹ per policy per year"); ax.legend()
    ax.set_title("Trigger sensitivity")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/4_trigger_sensitivity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    s = annual_hist.mean(axis=1)
    r = rain_frac.mean(axis=1).reindex(s.index)
    ax.scatter(r, s, color="#3b6ea5")
    ax.set_xlabel("Rainfall-deficit cover payout (fraction of sum insured)")
    ax.set_ylabel("Solar cover payout (₹/policy)")
    ax.set_title("Do the two covers pay in the same years?")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/5_solar_vs_rain.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("1. Loading NASA POWER data")
    df = add_generation(load_all())
    years = sorted(df["year"].unique())
    print(f"   {len(CITIES)} cities, {years[0]}-{years[-1]} ({len(years)} years)")

    # serial correlation check for the independence assumption
    lag1 = float(df.sort_values(["city", "year", "month"])
                   .groupby("city")["ratio"]
                   .apply(lambda x: x.autocorr(1)).mean())

    print("2. Burn analysis")
    df, annual_hist, monthly_freq = burn_analysis(df)
    burn_cost = float(annual_hist.values.mean())
    years_with_payout = float((annual_hist.mean(axis=1) > 0).mean())

    print("3. Monte Carlo pricing")
    sim, fit_info = monte_carlo(df)
    el, sd, gross = price(sim)
    per_policy = sim.reshape(-1)
    rp = return_periods(per_policy)

    print("4. Trigger sensitivity")
    sens = []
    for t in TRIGGER_GRID:
        _, ah, _ = burn_analysis(df, t)
        s_sim, _ = monte_carlo(df, t, n_sims=N_SIMS // 2)
        e, s, g = price(s_sim)
        sens.append({"trigger": t, "expected_loss_inr": e, "gross_premium_inr": g,
                     "premium_per_kw_inr": g / SYSTEM_KW,
                     "hist_payout_freq_monthly":
                         float((payout_from_ratio(df["ratio"], df["normal_kwh"], t) > 0).mean())})

    print("5. Portfolio & capital")
    port, port_loss = portfolio(sim, annual_hist, gross)

    print("6. Rainfall-deficit correlation")
    _, rain_frac = rainfall_cover(df)
    div = diversification(annual_hist, rain_frac)

    plots(monthly_freq, annual_hist, port_loss, port["annual_premium_inr"],
          sens, rain_frac)

    riskiest = monthly_freq.mean().sort_values(ascending=False).head(3)
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    city_premium = {c: float((sim[:, i].mean() + RISK_LOAD * sim[:, i].std())
                             / (1 - EXPENSE_RATIO))
                    for i, c in enumerate(CITIES)}
    nonnormal = sum(p < 0.05 for m in fit_info.values() for p in m.values())

    results = {
        "data": {"cities": list(CITIES), "first_year": int(years[0]),
                 "last_year": int(years[-1]), "n_years": len(years),
                 "city_months_modelled": int(len(df))},
        "assumptions": {"system_kw": SYSTEM_KW, "performance_ratio": PERFORMANCE_RATIO,
                        "tariff_inr_per_unit": TARIFF_INR, "trigger": TRIGGER,
                        "exit": EXIT, "risk_load": RISK_LOAD,
                        "expense_ratio": EXPENSE_RATIO, "n_sims": N_SIMS},
        "generation": {
            "avg_annual_kwh_per_policy": float(df.groupby(["city", "year"])["gen_kwh"]
                                               .sum().mean()),
            "kwh_per_kw_per_day": float(df["gen_kwh"].sum() / len(years) / len(CITIES)
                                        / SYSTEM_KW / 365.25),
        },
        "burn": {"burn_cost_inr_per_policy": burn_cost,
                 "share_of_years_with_any_payout": years_with_payout,
                 "riskiest_months": {month_names[m]: float(v)
                                     for m, v in riskiest.items()}},
        "pricing": {"expected_loss_inr": el, "loss_std_inr": sd,
                    "gross_premium_inr": gross,
                    "premium_per_kw_inr": gross / SYSTEM_KW,
                    "max_annual_payout_inr": max_annual_payout(df),
                    "premium_by_city_inr": city_premium,
                    "return_period_payouts_inr": rp},
        "model_checks": {"lag1_autocorr_of_monthly_ratio": lag1,
                         "city_months_failing_normality_p05": nonnormal,
                         "city_months_tested": 12 * len(CITIES),
                         "mc_vs_burn_gap_pct": float((el - burn_cost) / burn_cost * 100)
                         if burn_cost else None},
        "trigger_sensitivity": sens,
        "portfolio": port,
        "diversification_vs_rainfall_cover": div,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    pooled = div["Gujarat_pooled"]
    lines = [
        "=" * 64,
        "RESUME NUMBERS",
        "=" * 64,
        f"Cities: {len(CITIES)}   Years: {len(years)} ({years[0]}-{years[-1]})",
        f"Modelled generation: {results['generation']['kwh_per_kw_per_day']:.2f} kWh/kW/day",
        f"Riskiest months: {', '.join(results['burn']['riskiest_months'])}",
        f"Burn cost (historical): ₹{burn_cost:,.0f} per policy per year",
        f"Expected loss (Monte Carlo): ₹{el:,.0f}   (gap vs burn: "
        f"{results['model_checks']['mc_vs_burn_gap_pct']:+.1f}%)",
        f"Gross premium: ₹{gross:,.0f}/year for {SYSTEM_KW:.0f} kW  ->  "
        f"₹{gross / SYSTEM_KW:,.0f}/kW/year",
        "Return-period payouts per policy: " + ", ".join(
            f"{k.replace('_', ' ')} = ₹{v:,.0f}" for k, v in rp.items()),
        f"Portfolio expected loss ratio: {port['expected_loss_ratio']:.0%}  "
        f"| worst historical year {port['hist_worst_year']}: "
        f"{port['hist_loss_ratio_worst']:.0%}",
        f"1-in-200 capital for {N_POLICIES:,} policies: "
        f"₹{port['capital_needed_1_in_200_inr'] / 1e5:,.1f} lakh",
        f"Solar vs rainfall-deficit payouts (pooled): Spearman rho = "
        f"{pooled['spearman_rho']:+.2f} (p = {pooled['spearman_p']:.3f})",
        "",
        "Model checks:",
        f"  lag-1 autocorrelation of monthly ratio: {lag1:+.2f}",
        f"  city-months failing normality (p<0.05): {nonnormal}/{12 * len(CITIES)}",
        "=" * 64,
    ]
    text = "\n".join(lines)
    print("\n" + text)
    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
