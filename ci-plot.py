from pathlib import Path
from typing import Optional

import pandas as pd
import seaborn as sns
import typer
from matplotlib import pyplot as plt

app = typer.Typer()


@app.command()
def build_durations(input: Path = "result.csv", jobs: Optional[str] = None):
    """
    Plots the total durations of CI build times over a time period.
    """
    df = pd.read_csv(input)
    if jobs is not None:
        df = df[df["job"].isin(jobs.split(","))]

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    g = sns.lineplot(data=df, x="timestamp", y="total", hue="job")
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set(ylabel="Duration [s]", ylim=(0, 12000))

    plt.tight_layout()
    plt.savefig("build-durations.png")


AGGREGATED_COLS = "llvm", "rustc-1", "rustc-2", "test-build", "test-run"


@app.command()
def step_durations(input: Path = "result.csv", mode="bootstrap", jobs: Optional[str] = None):
    """
    Plots durations of individual bootstrap steps.

    @param mode: Either "bootstrap" (for displaying individual build stages) or "test" (for displaying test suite
    durations).
    """
    df = pd.read_csv(input)
    if jobs is not None:
        df = df[df["job"].isin(jobs.split(","))]

    if mode == "bootstrap":
        df = df[df.columns[df.columns.isin(["job", *AGGREGATED_COLS])]]
    elif mode == "test":
        df = df[df.columns[~df.columns.isin(AGGREGATED_COLS)]].drop(columns=["timestamp", "total"])
    else:
        assert False

    def fn(data, **kwargs):
        data = pd.melt(data, id_vars=["job"], var_name="section")
        g = sns.barplot(data=data, x="section", y="value")
        g.set_xticklabels(g.get_xticklabels(), rotation=90)

    if jobs is not None and len(jobs) == 1:
        fn(df)
    else:
        grid = sns.FacetGrid(df, col="job", col_wrap=4, sharey=True)
        grid.map_dataframe(fn)
    plt.savefig("step-durations.png", dpi=300)


if __name__ == "__main__":
    app()
