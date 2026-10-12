import pandas as pd

df = pd.read_csv("results/experiment_tiny.csv")
d = df[(df.snr == 0) & (df.method == "spectral_mild")].sort_values("errors", ascending=False)
d["hyp"] = d["hyp"].astype(str).str[:150]
print(d[["noise", "errors", "words", "hyp"]].head(5).to_string())