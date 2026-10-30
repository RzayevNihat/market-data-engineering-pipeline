"""Generate a reproducible summary of walk-forward research results."""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("reports/walk_forward_30_120_confirm10.csv")
OUTPUT_PATH = Path("reports/walk_forward_summary.csv")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Report not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required = {
        "fold",
        "variant",
        "net_return",
        "max_drawdown",
        "round_trips",
        "exposure",
        "buy_hold_net",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Walk-forward report is empty.")

    if not np.isfinite(
        df[["net_return", "max_drawdown", "buy_hold_net"]]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError("Report contains non-finite metrics.")

    # Buy & Hold should be identical across variants within each fold.
    bh_counts = df.groupby("fold")["buy_hold_net"].nunique()

    if (bh_counts != 1).any():
        raise ValueError(
            "Inconsistent Buy & Hold results within a fold."
        )

    buy_hold = (
        df[["fold", "buy_hold_net"]]
        .drop_duplicates("fold")
        .sort_values("fold")
    )

    buy_hold_compounded = (
        np.prod(1 + buy_hold["buy_hold_net"].to_numpy()) - 1
    )

    rows = []

    for variant, group in df.groupby("variant"):
        group = group.sort_values("fold")

        net_returns = group["net_return"].to_numpy()

        if np.any(net_returns <= -1):
            raise ValueError("Return less than or equal to -100%.")

        rows.append({
            "variant": variant,
            "folds": len(group),
            "compounded_net_return": (
                np.prod(1 + net_returns) - 1
            ),
            "mean_fold_return": net_returns.mean(),
            "worst_fold_return": net_returns.min(),
            "mean_max_drawdown": group["max_drawdown"].mean(),
            "total_round_trips": group["round_trips"].sum(),
            "mean_exposure": group["exposure"].mean(),
        })

    summary = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print("\n--- FINAL RESEARCH SUMMARY ---")
    print(summary.to_string(
        index=False,
        formatters={
            "compounded_net_return": "{:.2%}".format,
            "mean_fold_return": "{:.2%}".format,
            "worst_fold_return": "{:.2%}".format,
            "mean_max_drawdown": "{:.2%}".format,
            "mean_exposure": "{:.2%}".format,
        }
    ))

    print(
        f"\nBuy & Hold compounded return: "
        f"{buy_hold_compounded:.2%}"
    )

    pivot = df.pivot(
        index="fold",
        columns="variant",
        values="net_return"
    )

    if {"no_filter", "q50_train_only"}.issubset(pivot.columns):
        print("\n--- FILTER IMPROVEMENT BY FOLD ---")

        for fold in pivot.index:
            difference = (
                pivot.loc[fold, "q50_train_only"]
                - pivot.loc[fold, "no_filter"]
            )
            print(f"Fold {fold}: {difference:+.2%}")

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
