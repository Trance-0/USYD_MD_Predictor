"""
This script is used to analyze data
"""
import pandas as pd
import matplotlib.pyplot as plt

# read csv data
df = pd.read_csv("./admission_data.csv")

print("raw data:")
print(df.head())

Res_mapping = {k: v for v, k in enumerate(df["Outcome"].unique())}

df["Outcome"] = df["Outcome"].map(Res_mapping)

# data_cleaning, since the data is precious, we don't need to clean and just fill them with empty
df = df[["Section 1", "Section 2", "Section 3", "Outcome"]]

# no need to introduce redundent message
x = df[["Section 1", "Section 2", "Section 3"]]
y = df["Outcome"]

x = x.to_numpy().T
y = y.to_numpy().T

print(x[:5], x.shape)
print(y[:5], y.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(x[0], x[1], x[2], c=y, cmap=plt.get_cmap("viridis"))

ax.set_xlabel("S1")
ax.set_ylabel("S2")
ax.set_zlabel("S3")

plt.show()
