import glob
import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import poisson
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
CSV_PATTERNS = [
    os.path.join(BASE_DIR, "results", "duopoly_competition_details*.csv"),
    os.path.join(BASE_DIR, "duopoly_competition_details*.csv"),
]
ARTIFACT_PATH = os.path.join(BASE_DIR, "model_artifacts.pkl")

N_CLUSTERS = 3          # Change to 4 if K=4 also gives clean negative price slopes
CHECKPOINT_DAY = 20     # Number of initial periods used to classify a live season
T_HORIZON = 100         # Total selling periods per season
MAX_CAPACITY = 80       # Starting seat inventory per season
PRICE_GRID = np.linspace(10.0, 100.0, 91)  # Candidate prices ($10 to $100 in $1 steps)


# ==============================================================================
# FAST NUMPY DYNAMIC PROGRAMMING SOLVER
# ==============================================================================
def solve_dp_policy(beta_0, beta_1, beta_2, avg_comp_price, T=T_HORIZON, max_seats=MAX_CAPACITY, price_grid=PRICE_GRID):
    """
    Pre-computes the optimal price table P[t, x] via backward induction.
    t: selling period (1..T)
    x: remaining seats (0..max_seats)
    """
    V = np.zeros((T + 2, max_seats + 1), dtype=np.float64)
    P = np.zeros((T + 1, max_seats + 1), dtype=np.float64)

    lambdas = np.maximum(0.001, beta_0 + beta_1 * price_grid + beta_2 * avg_comp_price)
    k_vals = np.arange(max_seats + 1)
    prob_matrix = poisson.pmf(k_vals[None, :], mu=lambdas[:, None])

    for t in range(T, 0, -1):
        for x in range(1, max_seats + 1):
            sales = k_vals[:x + 1]
            probs = prob_matrix[:, :x + 1].copy()
            probs[:, x] = 1.0 - probs[:, :x].sum(axis=1)

            future_V = V[t + 1, x - sales]
            expected_rev = (probs * (price_grid[:, None] * sales[None, :] + future_V[None, :])).sum(axis=1)

            best_idx = int(np.argmax(expected_rev))
            V[t, x] = expected_rev[best_idx]
            P[t, x] = price_grid[best_idx]

    return V, P


# ==============================================================================
# MAIN OFFLINE TRAINING PIPELINE
# ==============================================================================
def main():
    # 1. Find and concatenate all historical competition CSVs
    csv_files = []
    for pattern in CSV_PATTERNS:
        csv_files.extend(glob.glob(pattern))
    csv_files = sorted(set(csv_files))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {BASE_DIR} or {os.path.join(BASE_DIR, 'results')}")

    df_list = []
    for f in csv_files:
        temp = pd.read_csv(f)
        fname = os.path.basename(f)
        comp_col = temp["competition_id"].astype(str) if "competition_id" in temp.columns else "0"
        seas_col = temp["selling_season"].astype(str) if "selling_season" in temp.columns else "0"
        temp["unique_season_id"] = fname + "_" + comp_col + "_" + seas_col
        df_list.append(temp)

    df = pd.concat(df_list, ignore_index=True)

    # Clean competitor price: ignore sentinel values (>= 500) when competitor is out of stock
    df["clean_comp_price"] = np.where(
        np.isfinite(df["price_competitor"]) & (df["price_competitor"] < 500.0),
        df["price_competitor"],
        np.nan,
    )
    print(f"Loaded {len(csv_files)} CSV files | {df['unique_season_id'].nunique()} unique seasons | {len(df):,} rows\n")

    # 2. Extract first-20-days booking curve features per individual season
    df_init = df[df["selling_period"] <= CHECKPOINT_DAY].copy()
    season_features = (
        df_init.groupby("unique_season_id")
        .agg(
            cum_demand_init=("demand", "sum"),
            mean_price_init=("price", "mean"),
            mean_comp_price_init=("clean_comp_price", "mean"),
            periods_recorded=("selling_period", "nunique"),
        )
        .dropna()
    )
    season_features = season_features[season_features["periods_recorded"] == CHECKPOINT_DAY].copy()
    assert len(season_features) > 0, "Still 0 valid seasons found—check 'selling_period' values in your CSVs."

    feature_cols = ["cum_demand_init", "mean_price_init", "mean_comp_price_init"]

    # 3. Scale features and fit K-Means clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(season_features[feature_cols])

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    season_features["cluster"] = kmeans.fit_predict(X_scaled)

    # Merge cluster labels back onto full 100-period dataset (filtering to active competitor periods for clean OLS)
    df_clustered = df.merge(season_features[["cluster"]], on="unique_season_id", how="inner")
    df_ols = df_clustered.dropna(subset=["demand", "price", "clean_comp_price"]).copy()

    
    # 4. Demean within each season (Fixed-Effects OLS) to remove cross-season demand bias
    for col in ["demand", "price", "clean_comp_price"]:
        df_ols[f"{col}_dm"] = df_ols[col] - df_ols.groupby("unique_season_id")[col].transform("mean")

    summary_rows = []
    cluster_models = {}
    dp_tables = {}

    global_ols = smf.ols("demand_dm ~ price_dm + clean_comp_price_dm - 1", data=df_ols).fit()
    g_b1 = float(global_ols.params["price_dm"])
    g_b2 = float(global_ols.params["clean_comp_price_dm"])
    g_b0 = float(df_ols["demand"].mean() - g_b1 * df_ols["price"].mean() - g_b2 * df_ols["clean_comp_price"].mean())
    g_p_comp = float(df_ols["clean_comp_price"].mean())
    _, P_global = solve_dp_policy(g_b0, g_b1, g_b2, g_p_comp)

    summary_rows.append({
        "Model": "Global (FE-OLS)",
        "Seasons": df_clustered["unique_season_id"].nunique(),
        f"Avg_Sales_Day{CHECKPOINT_DAY}": round(season_features["cum_demand_init"].mean(), 2),
        "Avg_Comp_Price": round(g_p_comp, 2),
        "Intercept (b0)": round(g_b0, 4),
        "Price_Slope (b1)": round(g_b1, 4),
        "Comp_Slope (b2)": round(g_b2, 4),
        "R_squared": round(global_ols.rsquared, 4),
    })

    for c in range(N_CLUSTERS):
        sub_ols = df_ols[df_ols["cluster"] == c]
        sub_feat = season_features[season_features["cluster"] == c]
        ols_c = smf.ols("demand_dm ~ price_dm + clean_comp_price_dm - 1", data=sub_ols).fit()

        b1 = float(ols_c.params["price_dm"])
        b2 = float(ols_c.params["clean_comp_price_dm"])
        b0 = float(sub_ols["demand"].mean() - b1 * sub_ols["price"].mean() - b2 * sub_ols["clean_comp_price"].mean())
        avg_p_comp = float(sub_ols["clean_comp_price"].mean())

        cluster_models[c] = {
            "b0": b0,
            "b1": b1,
            "b2": b2,
            "avg_comp_price": avg_p_comp,
        }

        _, P_c = solve_dp_policy(b0, b1, b2, avg_p_comp)
        dp_tables[c] = P_c

        summary_rows.append({
            "Model": f"Cluster {c}",
            "Seasons": len(sub_feat),
            f"Avg_Sales_Day{CHECKPOINT_DAY}": round(sub_feat["cum_demand_init"].mean(), 2),
            "Avg_Comp_Price": round(avg_p_comp, 2),
            "Intercept (b0)": round(b0, 4),
            "Price_Slope (b1)": round(b1, 4),
            "Comp_Slope (b2)": round(b2, 4),
            "R_squared": round(ols_c.rsquared, 4),
        })

    print(pd.DataFrame(summary_rows).to_string(index=False))

    # 5. Save everything into one single artifact file for duopoly.py
    artifacts = {
        "checkpoint_day": CHECKPOINT_DAY,
        "max_capacity": MAX_CAPACITY,
        "price_grid": PRICE_GRID,
        "scaler": scaler,
        "kmeans": kmeans,
        "feature_cols": feature_cols,
        "cluster_models": cluster_models,
        "dp_tables": dp_tables,
        "P_global": P_global,
    }
    with open(ARTIFACT_PATH, "wb") as fp:
        pickle.dump(artifacts, fp)

    print(f"\nDone! Saved clusters, OLS models, and DP tables to '{ARTIFACT_PATH}'.")


if __name__ == "__main__":
    main()