import importlib
import polars as pl
import pandas as pd
import kaggle_evaluation.nfl_inference_server
import os
import numpy as np # Import numpy for calculations

# Import all three modules
nfl_gru = importlib.import_module('nfl_gru')
nfl_gnn = importlib.import_module('nfl_gnn')
nfl_gru_2 = importlib.import_module('nfl_2026')

def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pd.DataFrame:
    

    score_gru = 0.584
    score_gnn = 0.580 
    score_gru2 = 0.562
    
    # 2. Calculate weights based on inverse scores
    inv_gru = 1.0 / score_gru
    inv_gnn = 1.0 / score_gnn
    inv_gru2 = 1.0 / score_gru2
    
    total_inv = inv_gru + inv_gnn + inv_gru2
    
    w_gru = inv_gru / total_inv
    w_gnn = inv_gnn / total_inv
    w_gru2 = inv_gru2 / total_inv
    
    print(f"GRU Weight: {w_gru:.4f}")
    print(f"GNN Weight: {w_gnn:.4f}")
    print(f"GRU2 Weight: {w_gru2:.4f}")
    print(f"Total Weight: {w_gru + w_gnn + w_gru2}")

    # 3. Get predictions from all three models
    pred_gru = nfl_gru.predict(test, test_input)
    pred_gnn = nfl_gnn.predict(test, test_input)
    pred_gru2 = nfl_gru_2.predict(test, test_input)

    # 4. Ensure all are Pandas DataFrames
    if isinstance(pred_gru, pl.DataFrame):
        pred_gru = pred_gru.to_pandas()
    if isinstance(pred_gnn, pl.DataFrame):
        pred_gnn = pred_gnn.to_pandas()
    if isinstance(pred_gru2, pl.DataFrame):
        pred_gru2 = pred_gru2.to_pandas()

    # 5. Get the numpy arrays for calculation
    vals_gru = pred_gru[['x', 'y']].values
    vals_gnn = pred_gnn[['x', 'y']].values
    vals_gru2 = pred_gru2[['x', 'y']].values

    # 6. Apply the weighted average
    pred_ensemble = (w_gru * vals_gru) + (w_gnn * vals_gnn) + (w_gru2 * vals_gru2)
    
    return pd.DataFrame(pred_ensemble, columns=['x', 'y'])

# --- Rest of your server code ---
inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/nfl-big-data-bowl-2026-prediction/',))