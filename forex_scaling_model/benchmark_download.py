
import time
import pandas as pd
from data.sources import DukascopyLoader

def benchmark_dukascopy():
    loader = DukascopyLoader(concurrency=32, request_delay=0.005)
    
    start_time = time.time()
    # Load 3 days of EURUSD (London/NY session only = ~33 files)
    print("Starting benchmark: Loading 3 days of EURUSD...")
    df = loader.load("EURUSD", start="2029-01-01", end="2024-01-04", hours=list(range(7, 18)))
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Benchmark completed in {duration:.2f} seconds")
    print(f"Total ticks: {len(df):,}")
    if not df.empty:
        print(f"Ticks per second: {len(df)/duration:.2f}")

if __name__ == "__main__":
    benchmark_dukascopy()
