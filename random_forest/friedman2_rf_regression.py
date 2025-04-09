import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import time

# Set random seed for reproducibility
np.random.seed(42)

# Define constants
OUTPUT_DIR = "friedman2_dataset"
RF_DIR = "friedman2_rf_regression"
os.makedirs(RF_DIR, exist_ok=True)

def load_data():
    """Load the Friedman #2 dataset created in Part 1"""
    print("Loading Friedman #2 dataset...")
    
    # Load datasets
    train_df = pd.read_csv(os.path.join(OUTPUT_DIR, "train_data.csv"))
    val_df = pd.read_csv(os.path.join(OUTPUT_DIR, "validation_data.csv"))
    test_df = pd.read_csv(os.path.join(OUTPUT_DIR, "test_data.csv"))
    
    # Extract features and targets
    feature_names = ['x1', 'x2', 'x3', 'x4']
    X_train = train_df[feature_names].values
    y_train = train_df['target'].values
    X_val = val_df[feature_names].values
    y_val = val_df['target'].values
    X_test = test_df[feature_names].values
    y_test = test_df['target'].values
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names

def optimize_hyperparameters(X_train, y_train):
    """Find optimal hyperparameters using GridSearchCV"""
    print("Optimizing random forest hyperparameters...")
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create base model
    rf = RandomForestRegressor(random_state=42)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error',
        verbose=1
    )
    
    # Fit GridSearchCV
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"GridSearchCV took {end_time - start_time:.2f} seconds to complete")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {-grid_search.best_score_:.4f}")
    
    # Return the best model
    return grid_search.best_estimator_, grid_search.best_params_

def train_model(X_train, y_train, X_val, y_val, params=None):
    """Train a random forest model with the given parameters"""
    print("Training random forest model...")
    
    if params:
        # Use provided parameters
        rf = RandomForestRegressor(**params, random_state=42, oob_score=True)
    else:
        # Use default parameters
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            random_state=42,
            oob_score=True
        )
    
    # Train the model
    start_time = time.time()
    rf.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"Training took {end_time - start_time:.2f} seconds")
    
    # Calculate OOB score
    oob_score = rf.oob_score_
    print(f"Out-of-Bag Score: {oob_score:.4f}")
    
    # Evaluate on validation set
    y_val_pred = rf.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    print(f"Validation R²: {val_r2:.4f}")
    
    return rf

def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate the model on the test set and generate visualizations"""
    print("Evaluating random forest model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Create a dataframe with predictions and residuals
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'residual': y_test - y_pred,
        'abs_error': np.abs(y_test - y_pred)
    })
    
    # Save results
    results_df.to_csv(os.path.join(RF_DIR, "rf_predictions.csv"), index=False)
    
    # ----- Visualizations -----
    
    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Random Forest: Actual vs Predicted Values')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    plt.savefig(os.path.join(RF_DIR, "rf_actual_vs_predicted.png"))
    plt.close()
    
    # 2. Residual Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred, results_df['residual'], alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Random Forest: Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.savefig(os.path.join(RF_DIR, "rf_residuals.png"))
    plt.close()
    
    # 3. Residual Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['residual'], kde=True)
    plt.title('Random Forest: Residual Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(RF_DIR, "rf_residual_distribution.png"))
    plt.close()
    
    # 4. Feature Importance Plot
    plt.figure(figsize=(12, 8))
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.bar(range(len(feature_names)), importances[indices], align='center')
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices])
    plt.title('Random Forest: Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RF_DIR, "rf_feature_importance.png"))
    plt.close()
    
    # 5. Permutation Importance Plot (more robust than default feature importance)
    plt.figure(figsize=(12, 8))
    
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_importances = perm_importance.importances_mean
    perm_indices = np.argsort(perm_importances)[::-1]
    
    plt.bar(range(len(feature_names)), perm_importances[perm_indices], align='center')
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in perm_indices])
    plt.title('Random Forest: Permutation Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RF_DIR, "rf_permutation_importance.png"))
    plt.close()
    
    # 6. Learning Curve
    plt.figure(figsize=(12, 8))
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_test, y_test, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    # Calculate means and standard deviations
    train_scores_mean = -np.mean(train_scores, axis=1)  # Negative because of neg_rmse
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                     train_scores_mean + train_scores_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Cross-Validation Score')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='green')
    
    plt.title('Random Forest: Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(RF_DIR, "rf_learning_curve.png"))
    plt.close()
    
    # 7. Error Analysis by Feature Range
    # Create a figure with subplots for each feature
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Create a test dataset with features
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['actual'] = y_test
    test_df['predicted'] = y_pred
    test_df['error'] = np.abs(test_df['actual'] - test_df['predicted'])
    
    # Plot error distribution across feature ranges
    for i, feature in enumerate(feature_names):
        ax = axes[i]
        sns.scatterplot(x=feature, y='error', data=test_df, ax=ax, alpha=0.5)
        ax.set_title(f'Error Distribution across {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Absolute Error')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RF_DIR, "rf_error_by_feature.png"))
    plt.close()
    
    return results_df

def main():
    """Main function to orchestrate the workflow"""
    # 1. Load the data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_data()
    
    # 2. Optimize hyperparameters
    print("\nOptimizing hyperparameters (this may take some time)...")
    best_model, best_params = optimize_hyperparameters(X_train, y_train)
    
    # 3. Train the model with best parameters
    print("\nTraining final model with best parameters...")
    rf_model = train_model(X_train, y_train, X_val, y_val, best_params)
    
    # 4. Evaluate the model
    print("\nEvaluating model on test set...")
    results_df = evaluate_model(rf_model, X_test, y_test, feature_names)
    
    # 5. Save the model
    model_file = os.path.join(RF_DIR, "random_forest_model.joblib")
    joblib.dump(rf_model, model_file)
    print(f"\nRandom Forest model saved to {model_file}")
    
    # 6. Save model parameters and metrics
    metrics = {
        'parameters': best_params,
        'oob_score': rf_model.oob_score_,
        'test_rmse': np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test))),
        'test_mae': mean_absolute_error(y_test, rf_model.predict(X_test)),
        'test_r2': r2_score(y_test, rf_model.predict(X_test))
    }
    
    pd.DataFrame([metrics]).to_csv(os.path.join(RF_DIR, "rf_metrics.csv"), index=False)
    print(f"Model metrics saved to {os.path.join(RF_DIR, 'rf_metrics.csv')}")
    
    print("\nRandom Forest regression analysis complete!")

if __name__ == "__main__":
    main() 