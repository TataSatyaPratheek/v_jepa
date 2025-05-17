import os
import sys
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
import math
from datetime import datetime, timedelta
import psutil
import numpy as np

# Check if rich is available for enhanced terminal output
try:
    from rich.console import Console
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
    from rich.progress import TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Check if colorama is available (fallback if rich is not available)
try:
    from colorama import init, Fore, Back, Style
    init()  # Initialize colorama
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class MetricTracker:
    """Tracks metrics with history and statistics."""
    
    def __init__(self, name: str, window_size: int = 20, fmt: str = '.4f'):
        """
        Initialize metric tracker.
        
        Args:
            name: Metric name
            window_size: Window size for running statistics
            fmt: Format string for display
        """
        self.name = name
        self.window_size = window_size
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0
        self.history = []
        self.running_avg = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def update(self, val: float, n: int = 1):
        """
        Update with new value.
        
        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        # Update history
        self.history.append(val)
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
        
        # Update running average
        self.running_avg = sum(self.history) / len(self.history)
        
        # Update min/max
        self.min_val = min(self.min_val, val)
        self.max_val = max(self.max_val, val)
    
    def get_stats(self) -> Dict[str, float]:
        """Get all statistics as dictionary."""
        return {
            'name': self.name,
            'val': self.val,
            'avg': self.avg,
            'running_avg': self.running_avg,
            'min': self.min_val,
            'max': self.max_val,
            'count': self.count
        }
    
    def __str__(self):
        """String representation."""
        fmtstr = '{name}: {val' + self.fmt + '} (avg: {avg' + self.fmt + '})'
        return fmtstr.format(name=self.name, val=self.val, avg=self.avg)


class ETAEstimator:
    """Estimates time remaining based on progress and history."""
    
    def __init__(self, total_steps: int, window_size: int = 20):
        """
        Initialize ETA estimator.
        
        Args:
            total_steps: Total number of steps
            window_size: Window size for running statistics
        """
        self.total_steps = total_steps
        self.window_size = window_size
        self.step_times = []
        self.start_time = time.time()
        self.last_time = self.start_time
    
    def update(self, current_step: int):
        """
        Update with new step.
        
        Args:
            current_step: Current step number
        """
        current_time = time.time()
        step_time = current_time - self.last_time
        self.last_time = current_time
        
        # Record step time
        self.step_times.append(step_time)
        if len(self.step_times) > self.window_size:
            self.step_times = self.step_times[-self.window_size:]
    
    def get_eta(self, current_step: int) -> Tuple[float, str]:
        """
        Get estimated time remaining.
        
        Args:
            current_step: Current step number
            
        Returns:
            Tuple of (seconds_remaining, formatted_string)
        """
        if len(self.step_times) == 0 or current_step >= self.total_steps:
            return 0, "00:00:00"
        
        # Calculate average step time
        avg_step_time = sum(self.step_times) / len(self.step_times)
        
        # Calculate remaining steps and time
        remaining_steps = self.total_steps - current_step
        remaining_seconds = avg_step_time * remaining_steps
        
        # Format time string
        eta_string = str(timedelta(seconds=int(remaining_seconds)))
        
        return remaining_seconds, eta_string
    
    def get_elapsed(self) -> Tuple[float, str]:
        """
        Get elapsed time.
        
        Returns:
            Tuple of (seconds_elapsed, formatted_string)
        """
        elapsed_seconds = time.time() - self.start_time
        elapsed_string = str(timedelta(seconds=int(elapsed_seconds)))
        
        return elapsed_seconds, elapsed_string
    
    def get_speed(self, current_step: int) -> float:
        """
        Get processing speed in items per second.
        
        Args:
            current_step: Current step number
            
        Returns:
            Items per second
        """
        elapsed_seconds, _ = self.get_elapsed()
        if elapsed_seconds == 0:
            return 0.0
        
        return current_step / elapsed_seconds


class TrainingProgressDisplay:
    """
    Rich, colorful display for training progress.
    
    Uses Rich library if available, otherwise falls back to simpler output.
    """
    
    def __init__(self, 
                total_epochs: int, 
                steps_per_epoch: int,
                metrics: List[str] = None,
                use_rich: bool = True):
        """
        Initialize training progress display.
        
        Args:
            total_epochs: Total number of epochs
            steps_per_epoch: Steps per epoch
            metrics: List of metric names to display
            use_rich: Whether to use Rich for display
        """
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.metrics = metrics or ['loss']
        self.use_rich = use_rich and RICH_AVAILABLE
        
        # Initialize metric trackers
        self.trackers = {name: MetricTracker(name) for name in self.metrics}
        
        # Initialize ETA estimator
        self.eta = ETAEstimator(total_epochs * steps_per_epoch)
        
        # Initialize system monitor
        self.system_monitor = SystemMonitor()
        
        # Initialize rich components if available
        if self.use_rich:
            self.console = Console()
            self.setup_rich_display()
        
        # Initialize fallback display variables
        self.last_print_time = time.time()
        self.print_interval = 0.5  # seconds
    
    def setup_rich_display(self):
        """Set up rich display components."""
        self.layout = Layout()
        
        # Create header section
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=10)
        )
        
        # Set up progress bars
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("[bold yellow]{task.fields[speed]:.2f} it/s"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        
        # Create tasks
        self.epoch_task = self.progress.add_task(
            "[green]Epochs", total=self.total_epochs, speed=0
        )
        self.step_task = self.progress.add_task(
            "[cyan]Steps", total=self.steps_per_epoch, speed=0
        )
        
        # Initialize live display
        self.live = Live(self.layout, console=self.console, refresh_per_second=4)
    
    def generate_header(self) -> Union[Panel, str]:
        """Generate header section."""
        if self.use_rich:
            return Panel(
                Text("V-JEPA Training Progress", style="bold magenta", justify="center"),
                border_style="bright_blue"
            )
        else:
            return "=== V-JEPA Training Progress ==="
    
    def generate_metrics_table(self) -> Union[Table, str]:
        """Generate metrics table."""
        if self.use_rich:
            table = Table(show_header=True, header_style="bold")
            table.add_column("Metric", style="dim")
            table.add_column("Value", justify="right")
            table.add_column("Running Avg", justify="right")
            table.add_column("Min", justify="right")
            table.add_column("Max", justify="right")
            
            # Add metrics
            for name, tracker in self.trackers.items():
                stats = tracker.get_stats()
                table.add_row(
                    name,
                    f"{stats['val']:.4f}",
                    f"{stats['running_avg']:.4f}",
                    f"{stats['min']:.4f}",
                    f"{stats['max']:.4f}"
                )
            
            return table
        else:
            metrics_text = ""
            for name, tracker in self.trackers.items():
                stats = tracker.get_stats()
                metrics_text += f"{name}: {stats['val']:.4f} (avg: {stats['running_avg']:.4f}) "
            
            return metrics_text
    
    def generate_system_info(self) -> Union[Table, str]:
        """Generate system info table."""
        self.system_monitor.update()
        stats = self.system_monitor.get_stats()
        
        if self.use_rich:
            table = Table(show_header=True, header_style="bold yellow")
            table.add_column("Resource", style="dim")
            table.add_column("Usage", justify="right")
            
            table.add_row("CPU", f"{stats['cpu_percent']}%")
            table.add_row("Memory", f"{stats['memory_used_gb']:.2f} GB / {stats['memory_total_gb']:.2f} GB ({stats['memory_percent']}%)")
            if stats['gpu_available']:
                table.add_row("GPU Memory", f"{stats['gpu_memory_used_gb']:.2f} GB / {stats['gpu_memory_total_gb']:.2f} GB ({stats['gpu_memory_percent']}%)")
            
            return table
        else:
            system_text = (
                f"CPU: {stats['cpu_percent']}% | "
                f"Memory: {stats['memory_used_gb']:.2f} GB / {stats['memory_total_gb']:.2f} GB ({stats['memory_percent']}%)"
            )
            if stats['gpu_available']:
                system_text += f" | GPU Memory: {stats['gpu_memory_used_gb']:.2f} GB / {stats['gpu_memory_total_gb']:.2f} GB ({stats['gpu_memory_percent']}%)"
            
            return system_text
    
    def update_display(self, epoch: int, step: int, metrics: Dict[str, float] = None):
        """
        Update display with new progress and metrics.
        
        Args:
            epoch: Current epoch
            step: Current step
            metrics: Dictionary of metrics to update
        """
        # Update metrics
        if metrics:
            for name, value in metrics.items():
                if name in self.trackers:
                    self.trackers[name].update(value)
        
        # Update ETA
        global_step = (epoch - 1) * self.steps_per_epoch + step
        self.eta.update(global_step)
        speed = self.eta.get_speed(global_step)
        
        # Update display based on available libraries
        if self.use_rich:
            self._update_rich_display(epoch, step, speed)
        else:
            self._update_simple_display(epoch, step, speed)
    
    def _update_rich_display(self, epoch: int, step: int, speed: float):
        """Update rich display."""
        # Update progress bars
        self.progress.update(self.epoch_task, completed=epoch-1, speed=speed/self.steps_per_epoch)
        self.progress.update(self.step_task, completed=step, speed=speed)
        
        # Update layout sections
        self.layout["header"].update(self.generate_header())
        
        body_layout = Layout()
        body_layout.split_row(
            Layout(self.progress, name="progress"),
            Layout(self.generate_metrics_table(), name="metrics")
        )
        self.layout["body"].update(body_layout)
        self.layout["footer"].update(self.generate_system_info())
    
    def _update_simple_display(self, epoch: int, step: int, speed: float):
        """Update simple text display."""
        # Only update at intervals to avoid console spam
        current_time = time.time()
        if current_time - self.last_print_time < self.print_interval and step != 1 and step != self.steps_per_epoch:
            return
        
        self.last_print_time = current_time
        
        # Get information
        _, eta_string = self.eta.get_eta(epoch * self.steps_per_epoch + step)
        _, elapsed_string = self.eta.get_elapsed()
        metrics_text = self.generate_metrics_table()
        system_text = self.generate_system_info()
        
        # Create colorful output if colorama is available
        if COLORAMA_AVAILABLE:
            header = f"{Fore.MAGENTA}{Style.BRIGHT}=== V-JEPA Training Progress ==={Style.RESET_ALL}"
            progress = (
                f"{Fore.GREEN}Epoch: {Style.BRIGHT}{epoch}/{self.total_epochs}{Style.RESET_ALL} | "
                f"{Fore.CYAN}Step: {Style.BRIGHT}{step}/{self.steps_per_epoch}{Style.RESET_ALL} | "
                f"{Fore.YELLOW}Speed: {Style.BRIGHT}{speed:.2f} it/s{Style.RESET_ALL}"
            )
            time_info = (
                f"{Fore.BLUE}Elapsed: {Style.BRIGHT}{elapsed_string}{Style.RESET_ALL} | "
                f"{Fore.BLUE}ETA: {Style.BRIGHT}{eta_string}{Style.RESET_ALL}"
            )
            metrics_info = f"{Fore.MAGENTA}{metrics_text}{Style.RESET_ALL}"
            system_info = f"{Fore.YELLOW}{system_text}{Style.RESET_ALL}"
            
            # Print with color
            print(f"\r{header}", flush=True)
            print(f"\r{progress}", flush=True)
            print(f"\r{time_info}", flush=True)
            print(f"\r{metrics_info}", flush=True)
            print(f"\r{system_info}", flush=True)
        else:
            # Print without color
            header = "=== V-JEPA Training Progress ==="
            progress = f"Epoch: {epoch}/{self.total_epochs} | Step: {step}/{self.steps_per_epoch} | Speed: {speed:.2f} it/s"
            time_info = f"Elapsed: {elapsed_string} | ETA: {eta_string}"
            
            print(f"\r{header}", flush=True)
            print(f"\r{progress}", flush=True)
            print(f"\r{time_info}", flush=True)
            print(f"\r{metrics_text}", flush=True)
            print(f"\r{system_text}", flush=True)
        
        # Add separator line
        print("-" * 80, flush=True)
    
    def start(self):
        """Start the display."""
        if self.use_rich:
            self.live.start()
    
    def stop(self):
        """Stop the display."""
        if self.use_rich:
            self.live.stop()


class BenchmarkProgressDisplay:
    """Enhanced display for benchmark progress."""
    
    def __init__(self, 
                total_iterations: int,
                warmup_iterations: int = 0,
                use_rich: bool = True):
        """
        Initialize benchmark progress display.
        
        Args:
            total_iterations: Total number of iterations
            warmup_iterations: Number of warmup iterations
            use_rich: Whether to use Rich for display
        """
        self.total_iterations = total_iterations
        self.warmup_iterations = warmup_iterations
        self.use_rich = use_rich and RICH_AVAILABLE
        
        # Initialize metric trackers
        self.latency_tracker = MetricTracker("Latency (ms)", fmt='.2f')
        self.throughput_tracker = MetricTracker("Throughput (it/s)", fmt='.2f')
        self.memory_tracker = MetricTracker("Memory (MB)", fmt='.1f')
        
        # Initialize ETA estimator
        self.eta = ETAEstimator(total_iterations + warmup_iterations)
        
        # Initialize system monitor
        self.system_monitor = SystemMonitor()
        
        # Initialize rich components if available
        if self.use_rich:
            self.console = Console()
            self.setup_rich_display()
        
        # Initialize fallback display variables
        self.last_print_time = time.time()
        self.print_interval = 0.5  # seconds
    
    def setup_rich_display(self):
        """Set up rich display components."""
        self.layout = Layout()
        
        # Create header section
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=10)
        )
        
        # Set up progress bars
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("[bold yellow]{task.fields[speed]:.2f} it/s"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        
        # Create tasks
        self.iter_task = self.progress.add_task(
            "[green]Progress", total=self.total_iterations + self.warmup_iterations, speed=0
        )
        
        # Initialize live display
        self.live = Live(self.layout, console=self.console, refresh_per_second=4)
    
    def generate_header(self) -> Union[Panel, str]:
        """Generate header section."""
        if self.use_rich:
            return Panel(
                Text("V-JEPA Benchmark Progress", style="bold magenta", justify="center"),
                border_style="bright_blue"
            )
        else:
            return "=== V-JEPA Benchmark Progress ==="
    
    def generate_metrics_table(self) -> Union[Table, str]:
        """Generate metrics table."""
        latency_stats = self.latency_tracker.get_stats()
        throughput_stats = self.throughput_tracker.get_stats()
        memory_stats = self.memory_tracker.get_stats()
        
        if self.use_rich:
            table = Table(show_header=True, header_style="bold")
            table.add_column("Metric", style="dim")
            table.add_column("Value", justify="right")
            table.add_column("Running Avg", justify="right")
            table.add_column("Min", justify="right")
            table.add_column("Max", justify="right")
            
            # Add metrics
            table.add_row(
                "Latency (ms)",
                f"{latency_stats['val']:.2f}",
                f"{latency_stats['running_avg']:.2f}",
                f"{latency_stats['min']:.2f}",
                f"{latency_stats['max']:.2f}"
            )
            table.add_row(
                "Throughput (it/s)",
                f"{throughput_stats['val']:.2f}",
                f"{throughput_stats['running_avg']:.2f}",
                f"{throughput_stats['min']:.2f}",
                f"{throughput_stats['max']:.2f}"
            )
            table.add_row(
                "Memory (MB)",
                f"{memory_stats['val']:.1f}",
                f"{memory_stats['running_avg']:.1f}",
                f"{memory_stats['min']:.1f}",
                f"{memory_stats['max']:.1f}"
            )
            
            return table
        else:
            metrics_text = (
                f"Latency: {latency_stats['val']:.2f} ms (avg: {latency_stats['running_avg']:.2f}) | "
                f"Throughput: {throughput_stats['val']:.2f} it/s (avg: {throughput_stats['running_avg']:.2f}) | "
                f"Memory: {memory_stats['val']:.1f} MB (peak: {memory_stats['max']:.1f})"
            )
            
            return metrics_text
    
    def generate_system_info(self) -> Union[Table, str]:
        """Generate system info table."""
        self.system_monitor.update()
        stats = self.system_monitor.get_stats()
        
        if self.use_rich:
            table = Table(show_header=True, header_style="bold yellow")
            table.add_column("Resource", style="dim")
            table.add_column("Usage", justify="right")
            
            table.add_row("CPU", f"{stats['cpu_percent']}%")
            table.add_row("Memory", f"{stats['memory_used_gb']:.2f} GB / {stats['memory_total_gb']:.2f} GB ({stats['memory_percent']}%)")
            if stats['gpu_available']:
                table.add_row("GPU Memory", f"{stats['gpu_memory_used_gb']:.2f} GB / {stats['gpu_memory_total_gb']:.2f} GB ({stats['gpu_memory_percent']}%)")
            
            return table
        else:
            system_text = (
                f"CPU: {stats['cpu_percent']}% | "
                f"Memory: {stats['memory_used_gb']:.2f} GB / {stats['memory_total_gb']:.2f} GB ({stats['memory_percent']}%)"
            )
            if stats['gpu_available']:
                system_text += f" | GPU Memory: {stats['gpu_memory_used_gb']:.2f} GB / {stats['gpu_memory_total_gb']:.2f} GB ({stats['gpu_memory_percent']}%)"
            
            return system_text
    
    def update_display(self, 
                      iteration: int, 
                      latency_ms: float, 
                      throughput: float, 
                      memory_mb: float):
        """
        Update display with new progress and metrics.
        
        Args:
            iteration: Current iteration
            latency_ms: Current latency in milliseconds
            throughput: Current throughput in items per second
            memory_mb: Current memory usage in MB
        """
        # Update metrics
        self.latency_tracker.update(latency_ms)
        self.throughput_tracker.update(throughput)
        self.memory_tracker.update(memory_mb)
        
        # Update ETA
        self.eta.update(iteration)
        
        # Get progress state
        phase = "Warmup" if iteration <= self.warmup_iterations else "Benchmark"
        progress_iter = iteration
        phase_iter = iteration if iteration <= self.warmup_iterations else iteration - self.warmup_iterations
        phase_total = self.warmup_iterations if iteration <= self.warmup_iterations else self.total_iterations
        
        # Update display based on available libraries
        if self.use_rich:
            self._update_rich_display(phase, progress_iter, phase_iter, phase_total, throughput)
        else:
            self._update_simple_display(phase, progress_iter, phase_iter, phase_total, throughput)
    
    def _update_rich_display(self, phase, progress_iter, phase_iter, phase_total, throughput):
        """Update rich display."""
        # Update progress bars
        self.progress.update(self.iter_task, 
                            completed=progress_iter, 
                            speed=throughput,
                            description=f"[green]{phase} Progress")
        
        # Update layout sections
        self.layout["header"].update(self.generate_header())
        
        # Update body with metrics
        phase_text = f"[bold yellow]{phase} Phase: {phase_iter}/{phase_total}"
        
        body_layout = Layout()
        body_layout.split(
            Layout(self.progress, name="progress", size=3),
            Layout(Text(phase_text), name="phase", size=1),
            Layout(self.generate_metrics_table(), name="metrics")
        )
        self.layout["body"].update(body_layout)
        self.layout["footer"].update(self.generate_system_info())
    
    def _update_simple_display(self, phase, progress_iter, phase_iter, phase_total, throughput):
        """Update simple text display."""
        # Only update at intervals to avoid console spam
        current_time = time.time()
        if (current_time - self.last_print_time < self.print_interval 
            and phase_iter != 1 and phase_iter != phase_total):
            return
        
        self.last_print_time = current_time
        
        # Get information
        _, eta_string = self.eta.get_eta(progress_iter)
        _, elapsed_string = self.eta.get_elapsed()
        metrics_text = self.generate_metrics_table()
        system_text = self.generate_system_info()
        
        # Create colorful output if colorama is available
        if COLORAMA_AVAILABLE:
            header = f"{Fore.MAGENTA}{Style.BRIGHT}=== V-JEPA Benchmark Progress ==={Style.RESET_ALL}"
            progress = (
                f"{Fore.GREEN}{phase} Phase: {Style.BRIGHT}{phase_iter}/{phase_total}{Style.RESET_ALL} | "
                f"{Fore.CYAN}Overall: {Style.BRIGHT}{progress_iter}/{self.total_iterations + self.warmup_iterations}{Style.RESET_ALL} | "
                f"{Fore.YELLOW}Speed: {Style.BRIGHT}{throughput:.2f} it/s{Style.RESET_ALL}"
            )
            time_info = (
                f"{Fore.BLUE}Elapsed: {Style.BRIGHT}{elapsed_string}{Style.RESET_ALL} | "
                f"{Fore.BLUE}ETA: {Style.BRIGHT}{eta_string}{Style.RESET_ALL}"
            )
            metrics_info = f"{Fore.MAGENTA}{metrics_text}{Style.RESET_ALL}"
            system_info = f"{Fore.YELLOW}{system_text}{Style.RESET_ALL}"
            
            # Print with color
            print(f"\r{header}", flush=True)
            print(f"\r{progress}", flush=True)
            print(f"\r{time_info}", flush=True)
            print(f"\r{metrics_info}", flush=True)
            print(f"\r{system_info}", flush=True)
        else:
            # Print without color
            header = "=== V-JEPA Benchmark Progress ==="
            progress = f"{phase} Phase: {phase_iter}/{phase_total} | Overall: {progress_iter}/{self.total_iterations + self.warmup_iterations} | Speed: {throughput:.2f} it/s"
            time_info = f"Elapsed: {elapsed_string} | ETA: {eta_string}"
            
            print(f"\r{header}", flush=True)
            print(f"\r{progress}", flush=True)
            print(f"\r{time_info}", flush=True)
            print(f"\r{metrics_text}", flush=True)
            print(f"\r{system_text}", flush=True)
        
        # Add separator line
        print("-" * 80, flush=True)
    
    def start(self):
        """Start the display."""
        if self.use_rich:
            self.live.start()
    
    def stop(self):
        """Stop the display."""
        if self.use_rich:
            self.live.stop()


class SystemMonitor:
    """Monitor system resources."""
    
    def __init__(self, gpu_monitoring: bool = True):
        """
        Initialize system monitor.
        
        Args:
            gpu_monitoring: Whether to monitor GPU
        """
        self.gpu_monitoring = gpu_monitoring
        self.gpu_available = False
        
        # Check for GPU monitoring capabilities
        if gpu_monitoring:
            try:
                import torch
                self.torch_available = True
                self.gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
                self.is_mps = not torch.cuda.is_available() and torch.backends.mps.is_available()
            except ImportError:
                self.torch_available = False
        
        # Initialize stats
        self.stats = {}
        self.update()
    
    def update(self):
        """Update system stats."""
        # CPU usage
        self.stats['cpu_percent'] = psutil.cpu_percent()
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.stats['memory_total_gb'] = memory.total / (1024 ** 3)
        self.stats['memory_used_gb'] = memory.used / (1024 ** 3)
        self.stats['memory_percent'] = memory.percent
        
        # GPU usage if available
        self.stats['gpu_available'] = self.gpu_available
        
        if self.gpu_available and self.torch_available:
            import torch
            if torch.cuda.is_available():
                try:
                    # NVIDIA GPU stats
                    current_device = torch.cuda.current_device()
                    self.stats['gpu_memory_total_gb'] = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)
                    self.stats['gpu_memory_used_gb'] = torch.cuda.memory_allocated(current_device) / (1024 ** 3)
                    self.stats['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved(current_device) / (1024 ** 3)
                    self.stats['gpu_memory_percent'] = (self.stats['gpu_memory_used_gb'] / self.stats['gpu_memory_total_gb']) * 100
                except Exception:
                    self.gpu_available = False
            elif self.is_mps:
                try:
                    # M1/M2 GPU stats
                    self.stats['gpu_memory_total_gb'] = 0  # Not directly available
                    self.stats['gpu_memory_used_gb'] = torch.mps.current_allocated_memory() / (1024 ** 3)
                    self.stats['gpu_memory_reserved_gb'] = 0  # Not directly available
                    self.stats['gpu_memory_percent'] = 0  # Cannot calculate without total
                except Exception:
                    self.gpu_available = False
        
        return self.stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current stats."""
        return self.stats


# Configure colorful logger
def configure_colorful_logger(name: str = "v-jepa", level: int = logging.INFO):
    """
    Configure colorful logger.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    if COLORAMA_AVAILABLE:
        # Colorful formatter
        class ColorfulFormatter(logging.Formatter):
            FORMATS = {
                logging.DEBUG: Fore.CYAN + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
                logging.INFO: Fore.GREEN + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
                logging.WARNING: Fore.YELLOW + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
                logging.ERROR: Fore.RED + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
                logging.CRITICAL: Fore.RED + Back.WHITE + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
            }
            
            def format(self, record):
                log_fmt = self.FORMATS.get(record.levelno)
                formatter = logging.Formatter(log_fmt)
                return formatter.format(record)
        
        formatter = ColorfulFormatter()
    else:
        # Standard formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger