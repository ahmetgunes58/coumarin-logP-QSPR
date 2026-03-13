import pandas as pd
import glob
import numpy as np

# latest predictions file
p = sorted(glob.glob("04_results/stage3_panel_v1/*/predictions_external.csv"))[-1]

df = pd.read_csv(p)

rows = []

for (model, set_role), g in df.groupby(["model", "set_role"], sort=False):

    y = g["y_true"].to_numpy()
    yhat = g["y_pred"].to_numpy()

    rmse = float(np.sqrt(((y - yhat) ** 2).mean()))
    mae = float(np.abs(y - yhat).mean())

    if len(y) > 1:
        r2 = float(1 - (((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
    else:
        r2 = float("nan")

    rows.append([model, set_role, len(g), rmse, mae, r2])

out = pd.DataFrame(
    rows,
    columns=["model", "set_role", "n", "RMSE", "MAE", "R2"]
).sort_values(["set_role", "RMSE"])

print("\nExternal validation summary\n")
print(out.to_string(index=False))