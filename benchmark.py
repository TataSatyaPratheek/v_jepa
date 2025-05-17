import time
from src.utils.memory import MemoryMonitor

def benchmark(model, loader, device="mps"):
    model.to(device)
    monitor = MemoryMonitor()
    
    times = []
    for batch in loader:
        start = time.perf_counter()
        
        with torch.no_grad():
            batch = batch.to(device, non_blocking=True)
            _ = model(batch)
            
        times.append(time.perf_counter() - start)
        monitor.update()
        
    avg_time = sum(times) / len(times)
    print(f"Average inference: {avg_time:.4f}s")
    print(f"Peak memory: {monitor.peak_memory:.2f}MB")
