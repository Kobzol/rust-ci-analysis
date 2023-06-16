import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("result.csv")
# df = df[df["job"].isin((
#     "x86_64-mingw-1", "x86_64-mingw-2",
#     "i686-mingw-1", "i686-mingw-2",
#     "x86_64-msvc-1", "x86_64-msvc-2",
#     "i686-msvc-1", "i686-msvc-2",
#     "x86_64-apple-1", "x86_64-apple-2",
#     "dist-x86_64-linux"
# ))]

aggregated_cols = "llvm", "rustc-1", "rustc-2", "test-build", "test-run"

# Bootstrap steps
df = df[df.columns[df.columns.isin(["job", *aggregated_cols])]]

# Test suites
df = df[df.columns[~df.columns.isin(aggregated_cols)]]


def fn(data, **kwargs):
    data = pd.melt(data, id_vars=["job"], var_name="section")
    g = sns.barplot(data=data, x="section", y="value")
    g.set_xticklabels(g.get_xticklabels(), rotation=90)


grid = sns.FacetGrid(df, col="job", col_wrap=4, sharey=True)
grid.map_dataframe(fn)
# plt.show()
plt.savefig("output.png")
