"""Check latest experiment from DagsHub"""
import mlflow
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri("https://dagshub.com/Julio-analyst/gender-voice-detection.mlflow")
mlflow.set_experiment("gender-voice-detection")

# Get latest run
runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=5)

print("\n=== Latest 5 Experiments ===\n")

for idx, run in runs.iterrows():
    print(f"\n{idx+1}. Run: {run.get('tags.mlflow.runName', 'N/A')}")
    print(f"   Start time: {run['start_time']}")
    
    # Count metrics
    metric_count = 0
    for col in runs.columns:
        if 'metrics.' in col:
            val = run[col]
            if pd.notna(val):
                metric_count += 1
                metric_name = col.replace('metrics.', '')
                print(f"   {metric_name}: {val:.4f}" if isinstance(val, float) else f"   {metric_name}: {val}")
    
    if metric_count == 0:
        print("   ❌ No metrics logged")

