"""
Compare different model versions and track improvements
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class ModelComparison:
    """Track and compare different model versions"""

    def __init__(self, results_file='model_results.json'):
        self.results_file = results_file
        self.results = self.load_results()

    def load_results(self):
        """Load existing results or create new file"""
        if Path(self.results_file).exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {}

    def save_results(self):
        """Save results to JSON"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def add_result(self, name, metrics, config=None):
        """
        Add a model result

        Args:
            name: Model version name (e.g., "baseline", "with_lagged_features")
            metrics: Dict with keys like 'mae_overall', 'rmse_overall', 'r2_overall',
                    'mae_day1', 'mae_day7', etc.
            config: Optional dict with model configuration
        """
        self.results[name] = {
            'metrics': metrics,
            'config': config or {}
        }
        self.save_results()
        print(f"[OK] Added results for '{name}'")

    def compare(self, baseline='baseline', models=None):
        """
        Compare models against baseline

        Args:
            baseline: Name of baseline model
            models: List of model names to compare (None = all models)
        """
        if baseline not in self.results:
            print(f"[ERROR] Baseline '{baseline}' not found!")
            return

        if models is None:
            models = [k for k in self.results.keys() if k != baseline]

        baseline_metrics = self.results[baseline]['metrics']

        print("\n" + "="*80)
        print(f"MODEL COMPARISON (Baseline: {baseline})")
        print("="*80)

        # Compare overall metrics
        print("\n[METRICS] Overall Metrics:")
        print(f"{'Model':<30} {'MAE':<12} {'RMSE':<12} {'R2':<12}")
        print("-"*80)

        # Baseline
        print(f"{baseline:<30} {baseline_metrics['mae_overall']:<12.2f} "
              f"{baseline_metrics['rmse_overall']:<12.2f} "
              f"{baseline_metrics['r2_overall']:<12.4f}")

        # Other models
        for model_name in models:
            if model_name not in self.results:
                continue
            metrics = self.results[model_name]['metrics']

            mae_diff = metrics['mae_overall'] - baseline_metrics['mae_overall']
            rmse_diff = metrics['rmse_overall'] - baseline_metrics['rmse_overall']
            r2_diff = metrics['r2_overall'] - baseline_metrics['r2_overall']

            mae_str = f"{metrics['mae_overall']:.2f} ({mae_diff:+.2f})"
            rmse_str = f"{metrics['rmse_overall']:.2f} ({rmse_diff:+.2f})"
            r2_str = f"{metrics['r2_overall']:.4f} ({r2_diff:+.4f})"

            print(f"{model_name:<30} {mae_str:<12} {rmse_str:<12} {r2_str:<12}")

        # Compare per-day metrics
        print("\n[PER-DAY] Per-Day MAE Comparison:")
        print(f"{'Model':<30} {'Day 1':<10} {'Day 3':<10} {'Day 7':<10}")
        print("-"*80)

        # Baseline
        print(f"{baseline:<30} "
              f"{baseline_metrics.get('mae_day1', 'N/A'):<10} "
              f"{baseline_metrics.get('mae_day3', 'N/A'):<10} "
              f"{baseline_metrics.get('mae_day7', 'N/A'):<10}")

        # Other models
        for model_name in models:
            if model_name not in self.results:
                continue
            metrics = self.results[model_name]['metrics']

            day1 = metrics.get('mae_day1', 0)
            day3 = metrics.get('mae_day3', 0)
            day7 = metrics.get('mae_day7', 0)

            day1_diff = day1 - baseline_metrics.get('mae_day1', day1)
            day3_diff = day3 - baseline_metrics.get('mae_day3', day3)
            day7_diff = day7 - baseline_metrics.get('mae_day7', day7)

            print(f"{model_name:<30} "
                  f"{day1:.2f} ({day1_diff:+.2f})  "
                  f"{day3:.2f} ({day3_diff:+.2f})  "
                  f"{day7:.2f} ({day7_diff:+.2f})")

        print("\n" + "="*80)

    def plot_comparison(self, metric='mae_overall', baseline='baseline'):
        """Plot comparison of models"""
        models = list(self.results.keys())
        values = [self.results[m]['metrics'][metric] for m in models]

        # Sort by value
        sorted_pairs = sorted(zip(models, values), key=lambda x: x[1])
        models, values = zip(*sorted_pairs)

        plt.figure(figsize=(12, 6))
        colors = ['green' if m == baseline else 'steelblue' for m in models]
        plt.barh(models, values, color=colors)
        plt.xlabel(metric.replace('_', ' ').title())
        plt.title(f'Model Comparison: {metric}')
        plt.axvline(x=self.results[baseline]['metrics'][metric],
                    color='red', linestyle='--', label=f'{baseline} baseline')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'comparison_{metric}.png', dpi=300)
        print(f"[PLOT] Saved comparison plot: comparison_{metric}.png")
        plt.show()


# Example usage:
if __name__ == "__main__":
    tracker = ModelComparison()

    # Add Phase 0 baseline results
    baseline_metrics = {
        'mae_overall': 23.29,
        'rmse_overall': 30.60,
        'r2_overall': 0.7520,
        'mae_day1': 14.91,
        'mae_day2': 20.98,
        'mae_day3': 23.79,
        'mae_day4': 25.07,
        'mae_day5': 25.79,
        'mae_day6': 26.13,
        'mae_day7': 26.33,
        'rmse_day1': 20.02,
        'r2_day1': 0.8940,
        'r2_day7': 0.6961
    }

    baseline_config = {
        'features': 12,
        'hidden_dim': 256,
        'num_layers': 2,
        'architecture': 'GRU with attention',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 18
    }

    tracker.add_result('baseline', baseline_metrics, baseline_config)

    print("\n[OK] Baseline saved! Future model results can be added like:")
    print("""
    tracker.add_result('with_lagged_features', {
        'mae_overall': 21.5,  # Example improved value
        'rmse_overall': 28.3,
        'r2_overall': 0.780,
        ...
    })

    tracker.compare(baseline='baseline')
    """)
