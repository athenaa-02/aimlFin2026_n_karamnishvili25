"""
ddos_analysis.py
─────────────────────────────────────────────────────
Analyzes a web server log file to detect
DDoS attack intervals using:
  1. Requests-per-minute time series extraction
  2. Polynomial regression to model normal traffic baseline
  3. Residual analysis (z-score) to flag attack windows
  4. Visualizations saved as PNG files
─────────────────────────────────────────────────────
Usage:
    python ddos_analysis.py server.log
"""

import re
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────
LOG_FILE       = sys.argv[1] if len(sys.argv) > 1 else "server.log"
RESAMPLE_MIN   = "1min"          # time bucket size
ZSCORE_THRESH  = 3.0             # z-score threshold for anomaly
POLY_DEGREE    = 6               # polynomial degree for baseline
MIN_ATTACK_MIN = 2               # consecutive anomalous minutes = attack

# ── 1. Parse log ────────────────────────────────────
LOG_RE = re.compile(
    r'(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+)[^"]*" (\d{3}) (\d+)'
)


def parse_ts(raw):
    # Handles both formats:
    #   "22/Mar/2024:18:01:38 +0400"   (Apache standard)
    #   "2024-03-22 18:01:38+04:00"    (this log's format)
    try:
        raw = raw.strip()
        if raw[0].isdigit() and "-" in raw[:7]:
            # Format: "2024-03-22 18:01:38+04:00"
            ts_str = raw[:19]            # "2024-03-22 18:01:38"
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        else:
            # Format: "22/Mar/2024:18:01:38 +0400"
            ts_str = raw.split(" ")[0]   # "22/Mar/2024:18:01:38"
            return datetime.strptime(ts_str, "%d/%b/%Y:%H:%M:%S")
    except Exception:
        return None

rows = []
print(f"[1/5] Parsing {LOG_FILE} ...")
with open(LOG_FILE, "r", errors="replace") as f:
    for line in f:
        m = LOG_RE.match(line)
        if not m:
            continue
        ip, ts_raw, method, url, status, size = m.groups()
        ts = parse_ts(ts_raw)
        if ts:
            rows.append({"ts": ts, "ip": ip, "status": int(status), "size": int(size)})

df = pd.DataFrame(rows)
df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values("ts").reset_index(drop=True)
print(f"    Parsed {len(df):,} valid log entries  ({df['ts'].min()} → {df['ts'].max()})")

# ── 2. Aggregate to requests/minute ─────────────────
print("[2/5] Aggregating to requests/minute ...")
df.set_index("ts", inplace=True)
ts_counts   = df.resample(RESAMPLE_MIN).size().rename("requests")
ts_uniq_ips = df["ip"].resample(RESAMPLE_MIN).nunique().rename("unique_ips")
ts_err_rate = (df["status"] >= 400).resample(RESAMPLE_MIN).mean().rename("error_rate")

traffic = pd.concat([ts_counts, ts_uniq_ips, ts_err_rate], axis=1).fillna(0)
traffic.index.name = "minute"

# ── 3. Polynomial regression on full series ──────────
print("[3/5] Fitting polynomial regression (baseline) ...")

x_num = np.arange(len(traffic)).reshape(-1, 1)
y     = traffic["requests"].values.astype(float)

model = make_pipeline(PolynomialFeatures(POLY_DEGREE), LinearRegression())
model.fit(x_num, y)
y_pred = model.predict(x_num)
r2     = r2_score(y, y_pred)
print(f"    R² (full series, degree={POLY_DEGREE}): {r2:.4f}")

traffic["baseline"]  = y_pred
traffic["residual"]  = y - y_pred
resid_mean = traffic["residual"].mean()
resid_std  = traffic["residual"].std()
traffic["zscore"]    = (traffic["residual"] - resid_mean) / resid_std
traffic["anomaly"]   = traffic["zscore"] > ZSCORE_THRESH

# ── 4. Identify attack intervals ─────────────────────
print("[4/5] Identifying DDoS intervals ...")

# Label contiguous anomalous blocks
traffic["attack_block"] = (
    traffic["anomaly"]
    .astype(int)
    .groupby((traffic["anomaly"] != traffic["anomaly"].shift()).cumsum())
    .transform("sum")
) * traffic["anomaly"].astype(int)

attack_minutes = traffic[traffic["anomaly"]]
intervals = []
if not attack_minutes.empty:
    # Group consecutive minutes
    idx = attack_minutes.index.to_list()
    block_start = idx[0]
    block_end   = idx[0]
    for i in range(1, len(idx)):
        diff = (idx[i] - idx[i-1]).total_seconds() / 60
        if diff <= 2:
            block_end = idx[i]
        else:
            if (block_end - block_start).total_seconds() / 60 >= MIN_ATTACK_MIN:
                intervals.append((block_start, block_end))
            block_start = idx[i]
            block_end   = idx[i]
    if (block_end - block_start).total_seconds() / 60 >= MIN_ATTACK_MIN:
        intervals.append((block_start, block_end))

print(f"\n    ══ DETECTED DDoS INTERVAL(S) ══")
for s, e in intervals:
    dur = int((e - s).total_seconds() / 60) + 1
    peak = traffic.loc[s:e, "requests"].max()
    avg  = traffic.loc[s:e, "requests"].mean()
    print(f"    • {s.strftime('%H:%M')} → {e.strftime('%H:%M')}  ({dur} min)  peak={peak:.0f} req/min  avg={avg:.0f} req/min")

# ── 5. Visualizations ────────────────────────────────
print("\n[5/5] Generating plots ...")

plt.style.use("dark_background")
ACCENT  = "#4fc3f7"
ATTACK  = "#ef5350"
BASELINE= "#81c784"
NORMAL  = "#546e7a"

fig, axes = plt.subplots(4, 1, figsize=(16, 18), facecolor="#0d1117")
fig.suptitle("DDoS Attack Detection — Web Server Log Analysis",
             fontsize=16, color="white", y=0.98, fontweight="bold")

for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

# ── Plot 1: Requests/min + regression + attack bands ──
ax = axes[0]
ax.bar(traffic.index, traffic["requests"], width=0.0006, color=NORMAL, alpha=0.6, label="Requests/min")
ax.plot(traffic.index, traffic["baseline"], color=BASELINE, lw=2, label=f"Poly regression baseline (R²={r2:.3f})")
for s, e in intervals:
    ax.axvspan(s, e, color=ATTACK, alpha=0.25)
    ax.axvline(s, color=ATTACK, lw=1.5, ls="--")
    ax.axvline(e, color=ATTACK, lw=1.5, ls="--")
    dur = int((e - s).total_seconds() / 60) + 1
    mid = s + (e - s) / 2
    ax.annotate(f"DDoS\n{s.strftime('%H:%M')}–{e.strftime('%H:%M')}\n({dur} min)",
                xy=(mid, traffic.loc[s:e,"requests"].max()),
                xytext=(mid, traffic.loc[s:e,"requests"].max() * 1.05),
                color=ATTACK, fontsize=9, ha="center", fontweight="bold")
ax.set_title("Requests per Minute with Polynomial Regression Baseline")
ax.set_ylabel("Requests / min")
ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

# ── Plot 2: Z-score anomaly ──
ax = axes[1]
ax.plot(traffic.index, traffic["zscore"], color=ACCENT, lw=1.2, label="Z-score of residual")
ax.axhline(ZSCORE_THRESH, color=ATTACK, lw=1.5, ls="--", label=f"Threshold z={ZSCORE_THRESH}")
ax.fill_between(traffic.index, traffic["zscore"], ZSCORE_THRESH,
                where=traffic["zscore"] > ZSCORE_THRESH, color=ATTACK, alpha=0.4)
for s, e in intervals:
    ax.axvspan(s, e, color=ATTACK, alpha=0.12)
ax.set_title("Residual Z-score (Anomaly Detection)")
ax.set_ylabel("Z-score")
ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

# ── Plot 3: Unique IPs per minute ──
ax = axes[2]
ax.fill_between(traffic.index, traffic["unique_ips"], alpha=0.5, color="#ce93d8")
ax.plot(traffic.index, traffic["unique_ips"], color="#ce93d8", lw=1.2, label="Unique IPs/min")
for s, e in intervals:
    ax.axvspan(s, e, color=ATTACK, alpha=0.2)
ax.set_title("Unique Source IPs per Minute")
ax.set_ylabel("Unique IPs")
ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

# ── Plot 4: Error rate ──
ax = axes[3]
ax.fill_between(traffic.index, traffic["error_rate"] * 100, alpha=0.5, color="#ffb74d")
ax.plot(traffic.index, traffic["error_rate"] * 100, color="#ffb74d", lw=1.2, label="HTTP 4xx/5xx rate (%)")
for s, e in intervals:
    ax.axvspan(s, e, color=ATTACK, alpha=0.2)
ax.set_title("HTTP Error Rate (4xx/5xx) per Minute")
ax.set_ylabel("Error Rate (%)")
ax.set_xlabel("Time (HH:MM)")
ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("ddos_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: ddos_analysis.png")

# ── Residuals distribution ──
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
fig2.suptitle("Regression Residuals Analysis", fontsize=13, color="white", y=1.01, fontweight="bold")
for ax in axes2:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

ax = axes2[0]
normal_res = traffic.loc[~traffic["anomaly"], "residual"]
attack_res = traffic.loc[ traffic["anomaly"], "residual"]
ax.hist(normal_res, bins=40, color=BASELINE, alpha=0.7, label="Normal minutes")
ax.hist(attack_res, bins=20, color=ATTACK,   alpha=0.7, label="Anomalous minutes")
ax.axvline(ZSCORE_THRESH * resid_std + resid_mean, color="white", ls="--", lw=1.5, label="z=3 threshold")
ax.set_title("Residual Distribution", color="white")
ax.set_xlabel("Residual (actual − predicted)", color="#c9d1d9")
ax.set_ylabel("Count", color="#c9d1d9")
ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")

ax = axes2[1]
ax.scatter(traffic.index[~traffic["anomaly"]], traffic.loc[~traffic["anomaly"], "residual"],
           s=6, color=BASELINE, alpha=0.5, label="Normal")
ax.scatter(traffic.index[traffic["anomaly"]], traffic.loc[traffic["anomaly"], "residual"],
           s=20, color=ATTACK, alpha=0.9, label="Anomalous (DDoS)")
ax.axhline(ZSCORE_THRESH * resid_std + resid_mean, color="white", ls="--", lw=1.5, label=f"z={ZSCORE_THRESH} boundary")
ax.set_title("Residuals Over Time", color="white")
ax.set_xlabel("Time", color="#c9d1d9")
ax.set_ylabel("Residual", color="#c9d1d9")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")

plt.tight_layout()
plt.savefig("ddos_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: ddos_residuals.png")

# ── Save summary stats ──
print("\n── Summary Statistics ──")
total      = len(df)
attack_df  = df.loc[intervals[0][0]:intervals[-1][1]] if intervals else pd.DataFrame()
print(f"  Total requests:          {total:,}")
print(f"  Date:                    {traffic.index[0].strftime('%Y-%m-%d')}")
print(f"  Unique IPs (total):      {df['ip'].nunique():,}")
if not attack_df.empty:
    print(f"  Requests during attack:  {len(attack_df):,}  ({len(attack_df)/total*100:.1f}% of total)")
    print(f"  Unique IPs during attack:{attack_df['ip'].nunique():,}")
    peak_row = traffic["requests"].idxmax()
    print(f"  Peak req/min:            {int(traffic['requests'].max())} at {peak_row.strftime('%H:%M')}")

# Save traffic CSV for reference
traffic.to_csv("traffic_per_minute.csv")
print("\nDone. Output files: ddos_analysis.png, ddos_residuals.png, traffic_per_minute.csv")
