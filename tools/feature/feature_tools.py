import pandas as pd
import numpy as np
import json
import os
from crewai.tools import tool

PROCESSED_PATH = "artifacts/data/processed_churn.csv"
FEATURES_PATH = "artifacts/data/selected_features.json"
INTERACTION_FEATURES_PATH = "artifacts/data/interaction_features.json"
CLUSTERED_FEATURES_PATH = "artifacts/data/clustered_features.json"
MUTUAL_INFO_PATH = "artifacts/data/mutual_information.json"


@tool("Correlation Analysis")
def correlation_analysis_tool(path: str = PROCESSED_PATH, threshold: float = 0.05) -> str:
    """Compute correlation of each feature with the Churn target and return sorted results."""
    df = pd.read_csv(path)
    if "Churn" not in df.columns:
        return "ERROR: Churn column not found."
    corr = df.corr()["Churn"].drop("Churn").sort_values(key=abs, ascending=False)
    significant = corr[corr.abs() >= threshold]
    return f"Features correlated with Churn (|r| >= {threshold}):\n{significant.to_string()}"


@tool("Feature Importance via Random Forest")
def feature_importance_tool(path: str = PROCESSED_PATH, top_n: int = 15) -> str:
    """Train a quick Random Forest and return top-N feature importances."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path).dropna()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return f"Top {top_n} Feature Importances:\n{importances.head(top_n).to_string()}"


@tool("Variance Threshold Filter")
def variance_threshold_tool(path: str = PROCESSED_PATH, threshold: float = 0.01) -> str:
    """Remove features with variance below the given threshold."""
    from sklearn.feature_selection import VarianceThreshold

    df = pd.read_csv(path).dropna()
    X = df.drop(columns=["Churn"])
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    removed = list(X.columns[~selector.get_support()])
    kept = list(X.columns[selector.get_support()])
    return f"Variance threshold={threshold}.\nRemoved ({len(removed)}): {removed}\nKept ({len(kept)}): {kept}"


@tool("Select Top Features")
def select_top_features_tool(path: str = PROCESSED_PATH, top_n: int = 12) -> str:
    """Select top N features by RF importance and save to selected_features.json."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path).dropna()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected = list(importances.head(top_n).index)
    os.makedirs("artifacts/data", exist_ok=True)
    with open(FEATURES_PATH, "w") as f:
        json.dump(selected, f)
    return f"Selected top {top_n} features saved to {FEATURES_PATH}:\n{selected}"


# ── NEW FEATURE ENGINEERING TOOLS ADDED BELOW ──────────────────────────────────


@tool("Generate Interaction Features")
def generate_interaction_features_tool(
    path: str = PROCESSED_PATH,
    degree: int = 2,
    include_bias: bool = False,
    max_interactions: int = 10
) -> str:
    """
    Automatically creates polynomial or interaction terms (e.g., Tenure * MonthlyCharges).
    Useful for capturing feature relationships that linear models might miss.
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.ensemble import RandomForestClassifier
    
    df = pd.read_csv(path).dropna()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    # Select top features for interaction to avoid explosion
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)
    top_features = pd.Series(rf.feature_importances_, index=X.columns).nlargest(8).index.tolist()
    
    X_top = X[top_features]
    
    # Generate polynomial/interaction features
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=True)
    X_poly = poly.fit_transform(X_top)
    
    # Get feature names
    feature_names = poly.get_feature_names_out(top_features)
    
    # Skip the original features (first len(top_features) columns)
    interaction_names = feature_names[len(top_features):]
    interaction_data = X_poly[:, len(top_features):]
    
    # Limit to max_interactions
    if len(interaction_names) > max_interactions:
        # Select top interactions by correlation with target
        corr_scores = []
        for i, name in enumerate(interaction_names):
            corr = np.corrcoef(interaction_data[:, i], y.values)[0, 1]
            corr_scores.append((name, abs(corr) if not np.isnan(corr) else 0))
        
        corr_scores.sort(key=lambda x: x[1], reverse=True)
        interaction_names = [name for name, _ in corr_scores[:max_interactions]]
        interaction_data = interaction_data[:, [i for i, (name, _) in enumerate(corr_scores[:max_interactions])]]
    
    # Create DataFrame with interaction features
    interaction_df = pd.DataFrame(interaction_data, columns=interaction_names, index=X.index)
    
    # Save interaction feature names
    os.makedirs("artifacts/data", exist_ok=True)
    interaction_info = {
        "original_features": top_features,
        "interaction_features": list(interaction_names),
        "degree": degree,
        "n_interactions_created": len(interaction_names)
    }
    
    with open(INTERACTION_FEATURES_PATH, "w") as f:
        json.dump(interaction_info, f, indent=2)
    
    # Calculate correlation of interactions with target
    interaction_corr = []
    for i, name in enumerate(interaction_names):
        corr = np.corrcoef(interaction_data[:, i], y.values)[0, 1]
        if not np.isnan(corr):
            interaction_corr.append((name, round(corr, 4)))
    
    interaction_corr.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return (
        f"Interaction Features Generated:\n"
        f"Original Features Used: {top_features}\n"
        f"Degree: {degree}, Interaction Only: True\n"
        f"Total Interactions Created: {len(interaction_names)}\n"
        f"Top Interactions by Correlation with Churn:\n"
        + "\n".join([f"  {name}: {corr:+.4f}" for name, corr in interaction_corr[:5]]) +
        f"\nSaved to: {INTERACTION_FEATURES_PATH}\n"
        f"Note: Add these features to your feature set before model training."
    )


@tool("Recursive Feature Elimination RFE")
def recursive_feature_elimination_tool(
    path: str = PROCESSED_PATH,
    n_features_to_select: int = 10,
    step: int = 1
) -> str:
    """
    Iteratively removes least important features to find the optimal subset.
    Uses Recursive Feature Elimination with a Random Forest estimator.
    """
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    
    df = pd.read_csv(path).dropna()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    # Base estimator for RFE
    estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # RFE
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        step=step,
        verbose=0
    )
    
    rfe.fit(X, y)
    
    # Get results
    selected_features = list(X.columns[rfe.support_])
    eliminated_features = list(X.columns[~rfe.support_])
    ranking = pd.Series(rfe.ranking_, index=X.columns).sort_values()
    
    # Calculate performance with selected features
    from sklearn.model_selection import cross_val_score
    X_selected = X[selected_features]
    cv_scores = cross_val_score(estimator, X_selected, y, cv=5, scoring='roc_auc')
    
    result = {
        "n_features_original": len(X.columns),
        "n_features_selected": len(selected_features),
        "selected_features": selected_features,
        "eliminated_features": eliminated_features,
        "feature_ranking": ranking.to_dict(),
        "cv_auc_mean": round(cv_scores.mean(), 4),
        "cv_auc_std": round(cv_scores.std(), 4)
    }
    
    # Save results
    os.makedirs("artifacts/data", exist_ok=True)
    rfe_path = "artifacts/data/rfe_results.json"
    with open(rfe_path, "w") as f:
        json.dump(result, f, indent=2)
    
    return (
        f"Recursive Feature Elimination Complete:\n"
        f"Original Features: {len(X.columns)}\n"
        f"Selected Features: {len(selected_features)}\n"
        f"Eliminated Features: {len(eliminated_features)}\n"
        f"Cross-Validation AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})\n"
        f"Selected Features:\n  {selected_features}\n"
        f"Eliminated Features:\n  {eliminated_features}\n"
        f"Feature Ranking (1=best):\n{ranking.head(15).to_string()}\n"
        f"Results saved to: {rfe_path}"
    )


@tool("Calculate Mutual Information")
def calculate_mutual_information_tool(
    path: str = PROCESSED_PATH,
    top_n: int = 15,
    random_state: int = 42
) -> str:
    """
    Captures non-linear dependencies that correlation misses.
    Mutual Information detects both linear and non-linear relationships.
    """
    from sklearn.feature_selection import mutual_info_classif
    
    if not os.path.exists(path):
        return f"VALIDATION FAILED: Processed data not found at {path}."
    
    df = pd.read_csv(path).dropna()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    # Calculate mutual information (compatible with all sklearn versions)
    # Note: 'discrete' parameter only available in sklearn >= 1.3.0
    # For older versions, sklearn auto-detects based on data type
    try:
        # Try with discrete parameter (sklearn >= 1.3.0)
        mi_scores = mutual_info_classif(X, y, discrete='auto', random_state=random_state)
    except TypeError:
        # Fallback for older sklearn versions (no discrete parameter)
        mi_scores = mutual_info_classif(X, y, random_state=random_state)
    
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    
    # Save results
    os.makedirs("artifacts/data", exist_ok=True)
    mi_results = {
        "mutual_information_scores": mi_series.head(top_n).to_dict(),
        "all_features_mi": mi_series.to_dict(),
        "top_n": top_n,
        "sklearn_version": pd.__version__
    }
    
    mi_path = "artifacts/data/mutual_information.json"
    with open(mi_path, "w") as f:
        json.dump(mi_results, f, indent=2)
    
    # Compare with correlation
    corr = df.corr()["Churn"].drop("Churn")
    comparison = pd.DataFrame({
        "mutual_information": mi_series,
        "correlation": corr.abs()
    }).sort_values("mutual_information", ascending=False)
    
    # Find features where MI >> Correlation (non-linear relationships)
    comparison["mi_corr_ratio"] = comparison["mutual_information"] / (comparison["correlation"] + 0.001)
    nonlinear_features = comparison[comparison["mi_corr_ratio"] > 2.0].head(5)
    
    return (
        f"Mutual Information Analysis Complete:\n"
        f"Top {top_n} Features by Mutual Information:\n{mi_series.head(top_n).to_string()}\n"
        f"\nNon-Linear Relationship Detection:\n"
        f"(Features where MI is 2x+ higher than correlation)\n"
        + (f"{nonlinear_features.to_string()}\n" if len(nonlinear_features) > 0 else "  No strong non-linear relationships detected.\n") +
        f"\nSaved to: {mi_path}\n"
        f"Note: Mutual Information captures non-linear dependencies that correlation misses."
    )

@tool("Cluster Features")
def cluster_features_tool(
    path: str = PROCESSED_PATH,
    correlation_threshold: float = 0.85,
    method: str = "hierarchical"
) -> str:
    """
    Groups highly correlated features to select representatives.
    Reduces multicollinearity by keeping one feature per cluster.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    if not os.path.exists(path):
        return f"VALIDATION FAILED: Processed data not found at {path}."
    
    df = pd.read_csv(path).dropna()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # FIX: Ensure correlation matrix is perfectly symmetric
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix.values, 1.0)
    
    # Convert correlation to distance
    distance_matrix = 1 - corr_matrix
    
    # FIX: Ensure distance matrix is symmetric
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix.values, 0.0)
    
    # Hierarchical clustering
    if method == "hierarchical":
        try:
            # Convert to condensed distance matrix
            condensed_dist = squareform(distance_matrix.values, checks=False)  # Skip symmetry check
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_dist, method='ward')
            
            # Cut dendrogram at threshold
            distance_threshold = 1 - correlation_threshold
            clusters = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
        except Exception as e:
            # Fallback to simple clustering if hierarchical fails
            method = "simple"
            clusters = np.zeros(len(X.columns), dtype=int)
            cluster_id = 1
            clustered = set()
            
            for i, col1 in enumerate(X.columns):
                if col1 in clustered:
                    continue
                
                clusters[i] = cluster_id
                clustered.add(col1)
                
                for j, col2 in enumerate(X.columns):
                    if j <= i or col2 in clustered:
                        continue
                    
                    if corr_matrix.loc[col1, col2] >= correlation_threshold:
                        clusters[j] = cluster_id
                        clustered.add(col2)
                
                cluster_id += 1
    else:
        # Simple greedy clustering
        clusters = np.zeros(len(X.columns), dtype=int)
        cluster_id = 1
        clustered = set()
        
        for i, col1 in enumerate(X.columns):
            if col1 in clustered:
                continue
            
            clusters[i] = cluster_id
            clustered.add(col1)
            
            for j, col2 in enumerate(X.columns):
                if j <= i or col2 in clustered:
                    continue
                
                if corr_matrix.loc[col1, col2] >= correlation_threshold:
                    clusters[j] = cluster_id
                    clustered.add(col2)
            
            cluster_id += 1
    
    # Create cluster assignments DataFrame
    cluster_df = pd.DataFrame({
        "feature": X.columns,
        "cluster": clusters
    })
    
    # Select representative from each cluster (highest correlation with target)
    target_corr = X.corr()["Churn"].abs()
    representatives = []
    
    for cluster_id in np.unique(clusters):
        cluster_features = X.columns[clusters == cluster_id]
        best_feature = cluster_features[np.argmax(target_corr[cluster_features].values)]
        representatives.append(best_feature)
    
    # Calculate multicollinearity reduction
    original_pairs = len(X.columns) * (len(X.columns) - 1) // 2
    reduced_pairs = len(representatives) * (len(representatives) - 1) // 2
    reduction_pct = ((original_pairs - reduced_pairs) / original_pairs) * 100 if original_pairs > 0 else 0
    
    # Save results
    os.makedirs("artifacts/data", exist_ok=True)
    cluster_results = {
        "correlation_threshold": correlation_threshold,
        "method": method,
        "n_original_features": len(X.columns),
        "n_clusters": len(np.unique(clusters)),
        "n_representatives": len(representatives),
        "multicollinearity_reduction_pct": round(reduction_pct, 2),
        "cluster_assignments": cluster_df.to_dict("records"),
        "representative_features": representatives,
        "high_correlation_pairs": []
    }
    
    # Find high correlation pairs
    high_corr_pairs = []
    for i, col1 in enumerate(X.columns):
        for j, col2 in enumerate(X.columns):
            if j > i and corr_matrix.loc[col1, col2] >= correlation_threshold:
                high_corr_pairs.append({
                    "feature1": col1,
                    "feature2": col2,
                    "correlation": round(float(corr_matrix.loc[col1, col2]), 4)
                })
    
    cluster_results["high_correlation_pairs"] = high_corr_pairs
    
    cluster_path = "artifacts/data/clustered_features.json"
    with open(cluster_path, "w") as f:
        json.dump(cluster_results, f, indent=2)
    
    # Format cluster summary
    cluster_summary = []
    for cluster_id in np.unique(clusters):
        cluster_features = list(X.columns[clusters == cluster_id])
        rep = [f for f in representatives if f in cluster_features][0]
        cluster_summary.append(f"  Cluster {cluster_id}: {cluster_features} → Representative: {rep}")
    
    return (
        f"Feature Clustering Complete:\n"
        f"Method: {method}\n"
        f"Correlation Threshold: {correlation_threshold}\n"
        f"Original Features: {len(X.columns)}\n"
        f"Clusters Formed: {len(np.unique(clusters))}\n"
        f"Representative Features: {len(representatives)}\n"
        f"Multicollinearity Reduction: {reduction_pct:.1f}%\n"
        f"\nCluster Assignments:\n"
        + "\n".join(cluster_summary) +
        f"\n\nHigh Correlation Pairs (≥{correlation_threshold}):\n"
        + "\n".join([f"  {p['feature1']} ↔ {p['feature2']}: {p['correlation']}" for p in high_corr_pairs[:10]]) +
        f"\n\nRepresentative Features Selected: {representatives}\n"
        f"Saved to: {cluster_path}\n"
        f"Note: Use representative features to reduce multicollinearity in your model."
    )