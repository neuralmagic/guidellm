import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def process_benchmark_yaml(yaml_file):
    """Process the benchmark YAML file and return a DataFrame with the data."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # Extract concurrency levels from the benchmark configuration
    concurrency_levels = data['benchmarks'][0]['args']['profile']['measured_concurrencies']
    
    # Process metrics for each concurrency level
    processed_data = []
    for i, benchmark in enumerate(data['benchmarks']):
        if 'metrics' in benchmark:
            metrics = benchmark['metrics']
            concurrency = concurrency_levels[i] if i < len(concurrency_levels) else 1.0
            
            # Extract successful metrics
            for metric_name, metric_data in metrics.items():
                if 'successful' in metric_data:
                    successful = metric_data['successful']
                    processed_data.append({
                        'concurrency': concurrency,
                        'metric': metric_name,
                        'count': successful.get('count', 0),
                        'mean': successful.get('mean', 0),
                        'median': successful.get('median', 0),
                        'min': successful.get('min', 0),
                        'max': successful.get('max', 0),
                        'std_dev': successful.get('std_dev', 0),
                        'p95': successful.get('percentiles', {}).get('p95', 0),
                        'p99': successful.get('percentiles', {}).get('p99', 0)
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    return df

def create_visualizations(df):
    """Create visualizations for the benchmark data."""
    # Create plots directory if it doesn't exist
    plot_dir = Path('benchmark_plots')
    plot_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # Sort by concurrency for better visualization
    df = df.sort_values('concurrency')
    
    # Create visualizations for each metric
    metrics_to_plot = [
        'request_latency',
        'time_to_first_token_ms',
        'tokens_per_second',
        'inter_token_latency_ms'
    ]
    
    for metric in metrics_to_plot:
        metric_df = df[df['metric'] == metric]
        if not metric_df.empty:
            # Mean vs Median
            plt.figure(figsize=(12, 6))
            plt.plot(metric_df['concurrency'], metric_df['mean'], 'b-', label='Mean')
            plt.plot(metric_df['concurrency'], metric_df['median'], 'r--', label='Median')
            plt.title(f'{metric.replace("_", " ").title()} vs Concurrency')
            plt.xlabel('Concurrency Level')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_dir / f'{metric}_mean_median.png')
            plt.close()
            
            # Min-Max Range
            plt.figure(figsize=(12, 6))
            plt.fill_between(metric_df['concurrency'], 
                           metric_df['min'], 
                           metric_df['max'], 
                           alpha=0.3, 
                           label='Min-Max Range')
            plt.plot(metric_df['concurrency'], metric_df['mean'], 'b-', label='Mean')
            plt.title(f'{metric.replace("_", " ").title()} Range vs Concurrency')
            plt.xlabel('Concurrency Level')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_dir / f'{metric}_range.png')
            plt.close()
            
            # Percentiles
            plt.figure(figsize=(12, 6))
            plt.plot(metric_df['concurrency'], metric_df['p95'], 'g--', label='95th Percentile')
            plt.plot(metric_df['concurrency'], metric_df['p99'], 'r--', label='99th Percentile')
            plt.plot(metric_df['concurrency'], metric_df['mean'], 'b-', label='Mean')
            plt.title(f'{metric.replace("_", " ").title()} Percentiles vs Concurrency')
            plt.xlabel('Concurrency Level')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_dir / f'{metric}_percentiles.png')
            plt.close()

def main():
    # Process the YAML file
    df = process_benchmark_yaml('llama32-3b.yaml')
    
    # Create visualizations
    create_visualizations(df)
    
    # Print summary statistics by concurrency level
    print("\nSummary Statistics by Concurrency Level:")
    for concurrency in sorted(df['concurrency'].unique()):
        print(f"\nConcurrency Level: {concurrency:.2f}")
        subset = df[df['concurrency'] == concurrency]
        
        for metric in subset['metric'].unique():
            metric_data = subset[subset['metric'] == metric]
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"Count: {metric_data['count'].iloc[0]}")
            print(f"Mean: {metric_data['mean'].iloc[0]:.2f}")
            print(f"Median: {metric_data['median'].iloc[0]:.2f}")
            print(f"Min: {metric_data['min'].iloc[0]:.2f}")
            print(f"Max: {metric_data['max'].iloc[0]:.2f}")
            print(f"Std Dev: {metric_data['std_dev'].iloc[0]:.2f}")
            print(f"95th Percentile: {metric_data['p95'].iloc[0]:.2f}")
            print(f"99th Percentile: {metric_data['p99'].iloc[0]:.2f}")
    
    # Save processed data
    df.to_csv('benchmark_processed_data.csv', index=False)
    print("\nProcessed data saved to benchmark_processed_data.csv")

if __name__ == "__main__":
    main()
