import wandb
import pandas as pd

USER_NAME = "koya-right-stuff-waseda-university"
PROJECT_NAME = "nc_2025"
ENTITY = "koya-right-stuff-waseda-university"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT_NAME}")

print(f"Found {len(runs)} runs")

all_history = []

for run in runs:
    print(f"Downloading history for run: {run.name}")
    
    history = run.history(pandas=True)

    history['run_name'] = run.name
    
    for k, v in run.config.items():
        history[f"config_{k}"] = v
        
    all_history.append(history)

if all_history:
    full_df = pd.concat(all_history)
    full_df.to_csv("all_experiments_history.csv", index=False)
    print("Saved to all_experiments_history.csv")
else:
    print("No data found.")