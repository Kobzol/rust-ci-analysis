import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("result.csv")
df = df[df["job"].isin((
    "x86_64-mingw-1", "x86_64-mingw-2",
    "i686-mingw-1", "i686-mingw-2",
    "x86_64-msvc-1", "x86_64-msvc-2",
    "i686-msvc-1", "i686-msvc-2",
    "x86_64-apple-1", "x86_64-apple-2",
    # "dist-x86_64-linux"
))]
df = df.drop(columns=["total"])


def fn(data, **kwargs):
    data = pd.melt(data, id_vars=["job"], var_name="section")
    g = sns.barplot(data=data, x="section", y="value")
    g.set_xticklabels(g.get_xticklabels(), rotation=45)


grid = sns.FacetGrid(df, col="job", col_wrap=4, sharey=True)
grid.map_dataframe(fn)
# plt.show()
plt.savefig("output.png")
