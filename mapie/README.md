# MAPIE Conformal Prediction Framework

This project provides a comprehensive framework for running conformal prediction experiments using various methods.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd mapie
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Configuration

Customize your experiment by editing the `config.yaml` file to configure:
- Dataset paths
- Conformal prediction methods and parameters
- Output and visualization options

### Running Experiments

To run an experiment with a single method:

```
python main.py --config config.yaml
```

To run all available conformal prediction methods at once:

```
python main.py --config config.yaml --run-all
```

Alternatively, you can set `run_all_methods: true` in the config.yaml file.

### Available Features

- Support for custom datasets and pre-trained models
- Multiple conformal prediction methods
- Performance metrics and visualization tools
- Comprehensive comparison between different methods

## Available Conformal Methods

The framework supports the following conformal prediction methods:

1. **Naive**: Uses only training data for calibration, simplest method
2. **Split**: Uses a validation set for calibration, theoretically sound
3. **Jackknife**: Leave-one-out cross-validation approach
4. **Jackknife+**: Enhanced jackknife method with stronger guarantees
5. **Jackknife MinMax**: Jackknife with min-max aggregation for better coverage
6. **CV**: Cross-validation-based method
7. **CV+**: Enhanced CV method with stronger guarantees
8. **CV MinMax**: CV with min-max aggregation for better coverage
9. **Jackknife+ AB**: Jackknife+ after bootstrap for improved stability
10. **EnbPI**: Ensemble-based prediction intervals

## File Structure

- `main.py`: Main script to run experiments
- `run_all_methods.py`: Script to run and compare all methods
- `config.yaml`: Configuration file defining experiment parameters
- `methods.py`: Implementation of all conformal prediction methods
- `scoring_functions.py`: Implementation of different conformity scores
- `data_loader.py`: Utilities for loading datasets and models
- `evaluation.py`: Tools for evaluating method performance

## Configuration File

The configuration file (`config.yaml`) allows you to customize all aspects of the experiment:

### Experiment Settings

```yaml
# Conformal Prediction Configuration
experiment_name: "friedman2_absolute_all_methods"

# Paths
data:
  train_path: "/path/to/train_data.csv"
  validation_path: "/path/to/validation_data.csv"
  test_path: "/path/to/test_data.csv"
  model_path: "/path/to/model.joblib"
  
# Conformal Prediction Parameters
conformal_prediction:
  alpha: 0.1  # For 90% confidence level
  scoring_function: "absolute_residual"  # Options: absolute_residual, gamma, residual_normalized
  method: "naive"  # Default method if not running all methods
  
  # Method parameters
  n_folds: 5  # For CV-based methods
  n_bootstrap: 30  # For bootstrap-based methods
  n_estimators: 10  # For ensemble-based methods
  
  # Set this to true to run all available methods
  run_all_methods: true
```

## Output Files

When running all methods, the framework generates the following outputs:

- Method comparison table in CSV format
- Comparison plots for coverage, interval width, and efficiency
- Scatter plot showing the coverage vs. width trade-off
- Individual results and plots for each method

## Example Output

When running all methods, the output will look like:

```
Method Comparison Summary:
Method               Coverage    Width      Efficiency  
--------------------------------------------------
naive                0.9120      1.2345     3.4567    
split                0.9050      1.1234     3.2345    
jackknife            0.9080      1.3456     3.5678    
jackknife_plus       0.9150      1.4567     3.6789    
...

Results saved to: /path/to/results/all_methods_timestamp
```

## License

This code is provided under the MIT License.

## References

- [MAPIE Documentation](https://mapie.readthedocs.io/)
- [Conformal Prediction Tutorial](https://github.com/scikit-learn-contrib/MAPIE/blob/master/examples/notebooks/tutorials/tutorial-getting-started-mapie-regression.ipynb) 