import pandas as pd

df = pd.read_csv("data/labels_updated.csv")

df["valence"] = df["valence"].clip(0,1)
df["arousal"] = df["arousal"].clip(0,1)

df.to_csv("data/labels_final.csv", index=False)

print("Fixed values >1")