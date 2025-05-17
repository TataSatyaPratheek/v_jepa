#!/usr/bin/env python3
"""
Demo script to showcase real-time progress logging features.
This will simulate a training and benchmarking process with rich terminal output.
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import (
    configure_colorful_logger,
    TrainingProgressDisplay,
    BenchmarkProgressDisplay,
    SystemMonitor
)

# Configure logger
logger = configure_colorful_logger("demo")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demo V-JEPA logging capabilities")
    
    # Demo type
    parser.add_argument('--type', type=str, default='training', choices=['training', 'benchmark', 'both'],
                        help='Type of demo to run')
    
    # Training demo settings
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training demo')
    parser.add_argument('--steps', type=int, default=100, help='Steps per epoch for training demo')
    
    # Benchmark demo settings
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations for benchmark demo')
    parser.add_argument('--warmup', type=int, default=20, help='Warmup iterations for benchmark demo')
    
    # Common settings
    parser.add_argument('--delay', type=float, default=0.05, help='Delay between steps (seconds)')
    parser.add_argument('--no-rich', action='store_true', help='Disable Rich library for display')
    
    return parser.parse_args()


def demo_training(args):
    """Run training demo with real-time progress display."""
    logger.info("Starting training demo")
    
    # Create progress display
    display = TrainingProgressDisplay(
        total_epochs=args.epochs,
        steps_per_epoch=args.steps,
        metrics=['loss', 'accuracy', 'memory_mb'],
        use_rich=not args.no_rich
    )
    
    # Start display
    display.start()
    
    # Initialize system monitor
    system_monitor = SystemMonitor()
    
    # Simulate training
    try:
        for epoch in range(1, args.epochs + 1):
            logger.info(f"Starting epoch {epoch}/{args.epochs}")
            
            for step in range(1, args.steps + 1):
                # Simulate step time
                time.sleep(args.delay)
                
                # Generate fake metrics
                metrics = {
                    'loss': max(0.1, 2.0 * (1.0 - (epoch - 1 + step / args.steps) / args.epochs)),
                    'accuracy': min(0.99, 0.5 + 0.5 * (epoch - 1 + step / args.steps) / args.epochs),
                    'memory_mb': 1000.0 + 500.0 * random.random()
                }
                
                # Update display
                display.update_display(epoch, step, metrics)
                
                # Inject random slowdown to make it more realistic
                if random.random() < 0.05:
                    time.sleep(args.delay * 3)
    
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    
    # Stop display
    display.stop()
    
    logger.info("Training demo completed")


def demo_benchmark(args):
    """Run benchmark demo with real-time progress display."""
    logger.info("Starting benchmark demo")
    
    # Create progress display
    display = BenchmarkProgressDisplay(
        total_iterations=args.iterations,
        warmup_iterations=args.warmup,
        use_rich=not args.no_rich
    )
    
    # Start display
    display.start()
    
    # Simulate benchmark
    try:
        # Warmup phase
        for i in range(1, args.warmup + 1):
            # Simulate step time
            time.sleep(args.delay)
            
            # Generate fake metrics
            latency_ms = 20.0 + 10.0 * random.random()
            throughput = 1000.0 / latency_ms
            memory_mb = 800.0 + 200.0 * random.random()
            
            # Update display
            display.update_display(
                iteration=i,
                latency_ms=latency_ms, 
                throughput=throughput, 
                memory_mb=memory_mb
            )
        
        # Benchmark phase
        for i in range(1, args.iterations + 1):
            # Simulate step time
            time.sleep(args.delay)
            
            # Generate fake metrics with improvement trend
            latency_ms = max(10.0, 20.0 + 10.0 * random.random() - 10.0 * i / args.iterations)
            throughput = 1000.0 / latency_ms
            memory_mb = 800.0 + 400.0 * (i / args.iterations) + 100.0 * random.random()
            
            # Update display
            display.update_display(
                iteration=args.warmup + i,
                latency_ms=latency_ms, 
                throughput=throughput, 
                memory_mb=memory_mb
            )
            
            # Inject random slowdown to make it more realistic
            if random.random() < 0.05:
                time.sleep(args.delay * 3)
    
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    
    # Stop display
    display.stop()
    
    # Show final results
    logger.info("Benchmark demo completed")
    logger.info("Final Results:")
    logger.info(f"Average latency: {15.0:.2f} ms")
    logger.info(f"Throughput: {66.7:.2f} it/s")
    logger.info(f"Peak memory: {1200.0:.1f} MB")


def main():
    """Main function."""
    args = parse_args()
    
    # Print welcome message
    print("\n" + "=" * 80)
    print(" V-JEPA Real-time Logging Demo ".center(80, "="))
    print("=" * 80 + "\n")
    
    # Check if rich is available
    try:
        from rich.console import Console
        rich_available = True
    except ImportError:
        rich_available = False
        if not args.no_rich:
            print("Rich library not found. Installing rich will provide enhanced display.")
            print("Run: pip install rich\n")
    
    # Check if colorama is available
    try:
        from colorama import Fore, Style
        colorama_available = True
    except ImportError:
        colorama_available = False
        print("Colorama library not found. Installing colorama will provide colored output in basic mode.")
        print("Run: pip install colorama\n")
    
    # Run demo(s)
    if args.type == 'training' or args.type == 'both':
        demo_training(args)
        if args.type == 'both':
            print("\n" + "-" * 80 + "\n")
            time.sleep(1)
    
    if args.type == 'benchmark' or args.type == 'both':
        demo_benchmark(args)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()