import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "h060m_b015m"


def main():
    with (FIXTURE / "micro_metrics.csv").open(newline="") as stream:
        rows = {row[""]: row for row in csv.DictReader(stream)}
    baseline = rows["ETS local"]
    model = rows["Residual XGB ETS LW"]
    print("Saved thesis results: next hour, 15-minute grid")
    print("This command reads aggregate fixtures; it does not retrain a model.\n")
    for metric in ["MAE", "LW-MAE", "HL-MAE"]:
        before, after = float(baseline[metric]), float(model[metric])
        print(f"{metric}: ETS {before:.4f}, residual {after:.4f}, reduction {100 * (before - after) / before:.2f}%")
    with (FIXTURE / "station_bootstrap_delta_ci.csv").open(newline="") as stream:
        row = next(row for row in csv.DictReader(stream) if
                   row["mode"] == "micro" and row["model"] == "Residual XGB ETS LW"
                   and row["baseline"] == "ETS local" and row["metric"] == "LW-MAE")
    print("\nStation-bootstrap record:")
    print(row)


if __name__ == "__main__":
    main()
