# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns

# df = pd.read_csv("sweep_val_ppl.csv")
# sns.set_theme(style="whitegrid")

# fig, ax = plt.subplots(figsize=(9, 5))
# sns.lineplot(
#     data=df,
#     x="words_seen",
#     y="val_ppl",
#     hue="run",
#     style="schedule",
#     markers=True,
#     dashes=False,
#     ax=ax,
# )
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel("Words seen")
# ax.set_ylabel("Validation perplexity")
# ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
# fig.tight_layout()
# fig.savefig("val_ppl_curves.png", dpi=200)

# final = df.sort_values("step").groupby("run").tail(1)
# print(final[["run", "lr", "grad_accum", "schedule", "val_ppl"]].sort_values("val_ppl"))
# fig, ax = plt.subplots(figsize=(6, 4))
# sns.scatterplot(
#     data=final, x="lr", y="val_ppl", hue="schedule", size="grad_accum", ax=ax
# )
# ax.set_xscale("log")
# ax.set_yscale("log")
# fig.tight_layout()
# fig.savefig("final_ppl_vs_lr.png", dpi=200)

import pandas as pd

df = pd.concat([pd.read_csv("part1.csv"), pd.read_csv("part2.csv")])
df = df.drop_duplicates(subset=["run", "step"], keep="last")
df.to_csv("sweep_val_ppl.csv", index=False)
print(df.groupby("run").val_ppl.agg(["min", "last"]))
