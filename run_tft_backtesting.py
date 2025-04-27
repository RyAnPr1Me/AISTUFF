import argparse
import logging
import pandas as pd

from tft_model import load_model_and_metadata, run_backtesting


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Run time series backtesting on trained TFT model")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--metadata", type=str, required=True, help="Path to the model metadata")
    parser.add_argument("--data", type=str, required=True, help="Path to the input data CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output results")
    parser.add_argument("--n_windows", type=int, default=1, help="Number of windows for backtesting")
    
    args = parser.parse_args()
    
    # Load model and metadata
    model, metadata = load_model_and_metadata(args.model, args.metadata)
    
    # Load data
    logging.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    logging.info(f"Loaded data with shape {df.shape}")
    
    # Run backtesting
    results = run_backtesting(
        model, 
        df, 
        metadata, 
        args.output_dir, 
        n_windows=args.n_windows
    )
    
    # Print summary results
    logging.info("Backtesting Summary:")
    for metric, value in results["summary"].items():
        logging.info(f"  {metric}: {value:.4f}")
    
    logging.info(f"Detailed results saved to {args.output_dir}/backtest_results.json")
    logging.info(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()