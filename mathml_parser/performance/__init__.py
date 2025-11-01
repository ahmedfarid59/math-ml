"""
Performance Optimization Engine
==============================

This module provides comprehensive performance optimization capabilities
for the MathML parser including caching, lazy evaluation, memory optimization,
and parallel processing.

Features:
- Intelligent caching mechanisms for parsed expressions
- Lazy evaluation for expensive computations
- Memory usage optimization and monitoring
- Parallel processing for batch operations
- Performance benchmarking and profiling tools
- Adaptive optimization based on usage patterns
"""

import time
import gc
import sys
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import hashlib
import pickle
import weakref
import memory_profiler
import cProfile
import pstats
from abc import ABC, abstractmethod


@dataclass
class PerformanceMetrics:
    """
    Container for performance measurement data.
    
    Attributes:
        operation_name: Name of the operation being measured
        execution_time: Time taken to execute (seconds)
        memory_usage: Memory used during execution (MB)
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        cpu_usage: CPU utilization percentage
        throughput: Operations per second
        timestamp: When the measurement was taken
    """
    operation_name: str
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cpu_usage: float = 0.0
    throughput: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self) -> str:
        return (f"Performance[{self.operation_name}]: "
                f"time={self.execution_time:.3f}s, "
                f"memory={self.memory_usage:.2f}MB, "
                f"cache_ratio={self.cache_hit_ratio:.2%}")
    
    @property
    def cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any) -> None:
        """Store value in cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get current cache size."""
        pass


class LRUCache(CacheStrategy):
    """
    Least Recently Used cache implementation.
    
    Provides O(1) access time with automatic eviction of least recently
    used items when capacity is exceeded.
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize LRU cache.
        
        Args:
            capacity: Maximum number of items to store
        """
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value and mark as recently used."""
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Store value and maintain capacity."""
        if key in self.cache:
            # Update existing
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'size': len(self.cache),
            'capacity': self.capacity
        }


class TTLCache(CacheStrategy):
    """
    Time-To-Live cache implementation.
    
    Automatically expires entries after a specified time period.
    """
    
    def __init__(self, capacity: int = 1000, ttl: float = 3600.0):
        """
        Initialize TTL cache.
        
        Args:
            capacity: Maximum number of items to store
            ttl: Time to live in seconds
        """
        self.capacity = capacity
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.hits = 0
        self.misses = 0
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry has expired."""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value if not expired."""
        self._cleanup_expired()
        
        if key in self.cache and not self._is_expired(key):
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Store value with timestamp."""
        self._cleanup_expired()
        
        # Make space if needed
        while len(self.cache) >= self.capacity:
            # Remove oldest entry
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            self.cache.pop(oldest_key, None)
            self.timestamps.pop(oldest_key, None)
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.timestamps.clear()
        self.hits = 0
        self.misses = 0
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class AdaptiveCache(CacheStrategy):
    """
    Adaptive cache that switches strategies based on usage patterns.
    
    Monitors cache performance and automatically switches between
    LRU and TTL strategies for optimal performance.
    """
    
    def __init__(self, capacity: int = 1000, ttl: float = 3600.0):
        """Initialize adaptive cache with both strategies."""
        self.lru_cache = LRUCache(capacity)
        self.ttl_cache = TTLCache(capacity, ttl)
        self.current_strategy = self.lru_cache
        self.performance_window = []
        self.window_size = 100
        self.last_switch_time = time.time()
        self.min_switch_interval = 60.0  # Minimum seconds between switches
    
    def _evaluate_performance(self) -> None:
        """Evaluate current cache performance and potentially switch strategies."""
        if len(self.performance_window) < self.window_size:
            return
        
        if time.time() - self.last_switch_time < self.min_switch_interval:
            return
        
        # Calculate hit ratios for both strategies
        lru_stats = self.lru_cache.stats()
        ttl_stats = {'hits': self.ttl_cache.hits, 'misses': self.ttl_cache.misses}
        
        lru_hit_ratio = lru_stats['hits'] / (lru_stats['hits'] + lru_stats['misses']) if (lru_stats['hits'] + lru_stats['misses']) > 0 else 0
        ttl_hit_ratio = ttl_stats['hits'] / (ttl_stats['hits'] + ttl_stats['misses']) if (ttl_stats['hits'] + ttl_stats['misses']) > 0 else 0
        
        # Switch to better performing strategy
        if self.current_strategy == self.lru_cache and ttl_hit_ratio > lru_hit_ratio + 0.1:
            self.current_strategy = self.ttl_cache
            self.last_switch_time = time.time()
        elif self.current_strategy == self.ttl_cache and lru_hit_ratio > ttl_hit_ratio + 0.1:
            self.current_strategy = self.lru_cache
            self.last_switch_time = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value using current strategy."""
        result = self.current_strategy.get(key)
        
        # Also try the other strategy if current misses
        if result is None:
            other_strategy = self.ttl_cache if self.current_strategy == self.lru_cache else self.lru_cache
            result = other_strategy.get(key)
            if result is not None:
                # Store in current strategy for future hits
                self.current_strategy.put(key, result)
        
        self._evaluate_performance()
        return result
    
    def put(self, key: str, value: Any) -> None:
        """Store in both strategies."""
        self.lru_cache.put(key, value)
        self.ttl_cache.put(key, value)
    
    def clear(self) -> None:
        """Clear both caches."""
        self.lru_cache.clear()
        self.ttl_cache.clear()
        self.performance_window.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return self.current_strategy.size()


class LazyEvaluator:
    """
    Lazy evaluation system for expensive computations.
    
    Defers computation until the result is actually needed,
    improving performance for complex mathematical expressions.
    """
    
    def __init__(self):
        """Initialize lazy evaluator."""
        self.pending_computations = {}
        self.computed_results = {}
    
    def defer(self, computation_id: str, computation_func: Callable, *args, **kwargs):
        """
        Defer a computation until later.
        
        Args:
            computation_id: Unique identifier for the computation
            computation_func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
        """
        self.pending_computations[computation_id] = {
            'func': computation_func,
            'args': args,
            'kwargs': kwargs,
            'timestamp': time.time()
        }
    
    def evaluate(self, computation_id: str) -> Any:
        """
        Evaluate a deferred computation.
        
        Args:
            computation_id: ID of computation to evaluate
            
        Returns:
            Result of the computation
        """
        if computation_id in self.computed_results:
            return self.computed_results[computation_id]
        
        if computation_id not in self.pending_computations:
            raise ValueError(f"No pending computation with ID: {computation_id}")
        
        computation = self.pending_computations[computation_id]
        
        try:
            result = computation['func'](*computation['args'], **computation['kwargs'])
            self.computed_results[computation_id] = result
            del self.pending_computations[computation_id]
            return result
        except Exception as e:
            # Keep the computation for retry
            raise RuntimeError(f"Failed to evaluate computation {computation_id}: {e}")
    
    def evaluate_all(self) -> Dict[str, Any]:
        """Evaluate all pending computations."""
        results = {}
        
        for computation_id in list(self.pending_computations.keys()):
            try:
                results[computation_id] = self.evaluate(computation_id)
            except Exception as e:
                results[computation_id] = e
        
        return results
    
    def cleanup_old(self, max_age: float = 3600.0) -> None:
        """Remove old pending computations."""
        current_time = time.time()
        old_ids = [
            comp_id for comp_id, comp_data in self.pending_computations.items()
            if current_time - comp_data['timestamp'] > max_age
        ]
        
        for comp_id in old_ids:
            del self.pending_computations[comp_id]


class MemoryOptimizer:
    """
    Memory usage optimization and monitoring.
    
    Provides tools for monitoring memory usage, garbage collection,
    and memory-efficient data structures.
    """
    
    def __init__(self):
        """Initialize memory optimizer."""
        self.memory_snapshots = []
        self.allocation_tracking = defaultdict(int)
        self.weak_references = weakref.WeakSet()
    
    def take_snapshot(self, label: str = "snapshot") -> Dict[str, Any]:
        """
        Take a memory usage snapshot.
        
        Args:
            label: Label for the snapshot
            
        Returns:
            Memory usage information
        """
        gc.collect()  # Force garbage collection
        
        snapshot = {
            'label': label,
            'timestamp': time.time(),
            'memory_usage_mb': self.get_memory_usage(),
            'object_counts': self.get_object_counts(),
            'gc_stats': gc.get_stats()
        }
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback to sys.getsizeof approximation
            return sys.getsizeof(gc.get_objects()) / (1024 * 1024)
    
    def get_object_counts(self) -> Dict[str, int]:
        """Get count of objects by type."""
        object_counts = defaultdict(int)
        
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] += 1
        
        return dict(object_counts)
    
    def analyze_memory_growth(self) -> Dict[str, Any]:
        """Analyze memory growth between snapshots."""
        if len(self.memory_snapshots) < 2:
            return {'error': 'Need at least 2 snapshots for analysis'}
        
        latest = self.memory_snapshots[-1]
        previous = self.memory_snapshots[-2]
        
        memory_growth = latest['memory_usage_mb'] - previous['memory_usage_mb']
        time_diff = latest['timestamp'] - previous['timestamp']
        
        # Analyze object count changes
        object_changes = {}
        for obj_type in set(latest['object_counts'].keys()) | set(previous['object_counts'].keys()):
            current_count = latest['object_counts'].get(obj_type, 0)
            previous_count = previous['object_counts'].get(obj_type, 0)
            change = current_count - previous_count
            if change != 0:
                object_changes[obj_type] = change
        
        return {
            'memory_growth_mb': memory_growth,
            'time_interval': time_diff,
            'growth_rate_mb_per_sec': memory_growth / time_diff if time_diff > 0 else 0,
            'object_changes': object_changes,
            'largest_increases': sorted(
                [(k, v) for k, v in object_changes.items() if v > 0],
                key=lambda x: x[1], reverse=True
            )[:10]
        }
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        collected_counts = [gc.collect(generation) for generation in range(3)]
        
        return {
            'generation_0': collected_counts[0],
            'generation_1': collected_counts[1],
            'generation_2': collected_counts[2],
            'total_collected': sum(collected_counts)
        }
    
    def register_weak_reference(self, obj: Any) -> None:
        """Register object for weak reference tracking."""
        self.weak_references.add(obj)
    
    def cleanup_weak_references(self) -> int:
        """Clean up dead weak references."""
        initial_count = len(self.weak_references)
        # WeakSet automatically removes dead references
        return initial_count - len(self.weak_references)


class ParallelProcessor:
    """
    Parallel processing capabilities for batch operations.
    
    Provides thread-based and process-based parallelization
    for CPU-intensive mathematical computations.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
        """
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.thread_pool = None
        self.process_pool = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()
    
    def process_batch_threaded(self, func: Callable, items: List[Any], 
                             chunk_size: Optional[int] = None) -> List[Any]:
        """
        Process batch of items using thread pool.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            chunk_size: Size of chunks for processing
            
        Returns:
            List of results in same order as input
        """
        if not items:
            return []
        
        if self.thread_pool is None:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.max_workers * 2))
        
        # Process in chunks
        futures = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            future = self.thread_pool.submit(self._process_chunk, func, chunk)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            results.extend(future.result())
        
        return results
    
    def process_batch_multiprocess(self, func: Callable, items: List[Any],
                                 chunk_size: Optional[int] = None) -> List[Any]:
        """
        Process batch of items using process pool.
        
        Args:
            func: Function to apply to each item (must be pickleable)
            items: List of items to process
            chunk_size: Size of chunks for processing
            
        Returns:
            List of results in same order as input
        """
        if not items:
            return []
        
        if self.process_pool is None:
            self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.max_workers * 2))
        
        # Process in chunks
        futures = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            future = self.process_pool.submit(self._process_chunk, func, chunk)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            results.extend(future.result())
        
        return results
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items."""
        return [func(item) for item in chunk]
    
    def map_reduce(self, map_func: Callable, reduce_func: Callable, 
                   items: List[Any], use_processes: bool = False) -> Any:
        """
        Perform map-reduce operation.
        
        Args:
            map_func: Function to map over items
            reduce_func: Function to reduce mapped results
            items: Items to process
            use_processes: Whether to use processes instead of threads
            
        Returns:
            Reduced result
        """
        if use_processes:
            mapped_results = self.process_batch_multiprocess(map_func, items)
        else:
            mapped_results = self.process_batch_threaded(map_func, items)
        
        # Reduce results
        result = mapped_results[0] if mapped_results else None
        for item in mapped_results[1:]:
            result = reduce_func(result, item)
        
        return result
    
    def shutdown(self) -> None:
        """Shutdown thread and process pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None


class PerformanceBenchmark:
    """
    Performance benchmarking and profiling tools.
    
    Provides comprehensive benchmarking capabilities for measuring
    and analyzing parser performance across different scenarios.
    """
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.benchmarks = {}
        self.profiles = {}
        self.baseline_results = {}
    
    def benchmark(self, name: str, iterations: int = 1000):
        """
        Decorator for benchmarking functions.
        
        Args:
            name: Name of the benchmark
            iterations: Number of iterations to run
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.run_benchmark(name, func, iterations, *args, **kwargs)
            return wrapper
        return decorator
    
    def run_benchmark(self, name: str, func: Callable, iterations: int = 1000,
                     *args, **kwargs) -> PerformanceMetrics:
        """
        Run benchmark for a function.
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            iterations: Number of iterations
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Performance metrics
        """
        # Warm-up run
        func(*args, **kwargs)
        
        # Memory before
        memory_before = self._get_memory_usage()
        
        # Timing
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        
        # Memory after
        memory_after = self._get_memory_usage()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / iterations
        throughput = iterations / total_time
        memory_used = memory_after - memory_before
        
        metrics = PerformanceMetrics(
            operation_name=name,
            execution_time=avg_time,
            memory_usage=memory_used,
            throughput=throughput
        )
        
        self.benchmarks[name] = metrics
        return metrics
    
    def profile_function(self, func: Callable, *args, **kwargs) -> pstats.Stats:
        """
        Profile function execution.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Profiling statistics
        """
        profiler = cProfile.Profile()
        
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        return stats
    
    def memory_profile_function(self, func: Callable, *args, **kwargs) -> List[float]:
        """
        Profile memory usage of function.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Memory usage over time
        """
        @memory_profiler.profile
        def wrapper():
            return func(*args, **kwargs)
        
        # This would require line-by-line profiling setup
        # For now, return simple before/after measurement
        before = self._get_memory_usage()
        result = func(*args, **kwargs)
        after = self._get_memory_usage()
        
        return [before, after]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def compare_benchmarks(self, baseline_name: str, 
                          comparison_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compare benchmark results against a baseline.
        
        Args:
            baseline_name: Name of baseline benchmark
            comparison_names: Names of benchmarks to compare
            
        Returns:
            Comparison results
        """
        if baseline_name not in self.benchmarks:
            raise ValueError(f"Baseline benchmark '{baseline_name}' not found")
        
        baseline = self.benchmarks[baseline_name]
        comparisons = {}
        
        for name in comparison_names:
            if name not in self.benchmarks:
                continue
            
            current = self.benchmarks[name]
            
            # Calculate relative performance
            time_ratio = current.execution_time / baseline.execution_time
            memory_ratio = current.memory_usage / baseline.memory_usage if baseline.memory_usage > 0 else 1.0
            throughput_ratio = current.throughput / baseline.throughput if baseline.throughput > 0 else 1.0
            
            comparisons[name] = {
                'time_ratio': time_ratio,
                'time_improvement': (1 - time_ratio) * 100,  # Percentage improvement
                'memory_ratio': memory_ratio,
                'memory_improvement': (1 - memory_ratio) * 100,
                'throughput_ratio': throughput_ratio,
                'throughput_improvement': (throughput_ratio - 1) * 100,
                'overall_score': (throughput_ratio / time_ratio) * (1 / memory_ratio)
            }
        
        return comparisons
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        lines = []
        lines.append("Performance Benchmark Report")
        lines.append("=" * 50)
        lines.append()
        
        if not self.benchmarks:
            lines.append("No benchmarks have been run.")
            return "\n".join(lines)
        
        # Summary table
        lines.append("Benchmark Results:")
        lines.append("-" * 50)
        lines.append(f"{'Name':<20} {'Time (ms)':<12} {'Memory (MB)':<12} {'Throughput':<12}")
        lines.append("-" * 50)
        
        for name, metrics in self.benchmarks.items():
            lines.append(
                f"{name:<20} "
                f"{metrics.execution_time * 1000:<12.3f} "
                f"{metrics.memory_usage:<12.2f} "
                f"{metrics.throughput:<12.1f}"
            )
        
        lines.append()
        
        # Detailed analysis
        lines.append("Detailed Analysis:")
        lines.append("-" * 50)
        
        # Best and worst performers
        if len(self.benchmarks) > 1:
            fastest = min(self.benchmarks.items(), key=lambda x: x[1].execution_time)
            slowest = max(self.benchmarks.items(), key=lambda x: x[1].execution_time)
            
            lines.append(f"Fastest: {fastest[0]} ({fastest[1].execution_time * 1000:.3f}ms)")
            lines.append(f"Slowest: {slowest[0]} ({slowest[1].execution_time * 1000:.3f}ms)")
            
            if fastest[1].execution_time > 0:
                speedup = slowest[1].execution_time / fastest[1].execution_time
                lines.append(f"Speed difference: {speedup:.2f}x")
        
        return "\n".join(lines)


class PerformanceOptimizer:
    """
    Main performance optimization engine.
    
    Coordinates all performance optimization components including
    caching, lazy evaluation, memory optimization, and parallel processing.
    """
    
    def __init__(self, cache_strategy: str = 'adaptive', 
                 enable_lazy_eval: bool = True,
                 enable_parallel: bool = True):
        """
        Initialize performance optimizer.
        
        Args:
            cache_strategy: Cache strategy ('lru', 'ttl', 'adaptive')
            enable_lazy_eval: Whether to enable lazy evaluation
            enable_parallel: Whether to enable parallel processing
        """
        # Initialize components
        self.cache = self._create_cache(cache_strategy)
        self.lazy_evaluator = LazyEvaluator() if enable_lazy_eval else None
        self.memory_optimizer = MemoryOptimizer()
        self.parallel_processor = ParallelProcessor() if enable_parallel else None
        self.benchmark = PerformanceBenchmark()
        
        # Performance monitoring
        self.metrics_history = []
        self.optimization_enabled = True
        
        # Auto-optimization settings
        self.auto_gc_threshold = 100  # MB
        self.auto_cache_cleanup_interval = 3600  # seconds
        self.last_cache_cleanup = time.time()
    
    def _create_cache(self, strategy: str) -> CacheStrategy:
        """Create cache based on strategy."""
        if strategy == 'lru':
            return LRUCache()
        elif strategy == 'ttl':
            return TTLCache()
        elif strategy == 'adaptive':
            return AdaptiveCache()
        else:
            raise ValueError(f"Unknown cache strategy: {strategy}")
    
    def cached_operation(self, cache_key: str = None):
        """
        Decorator for caching expensive operations.
        
        Args:
            cache_key: Custom cache key (if None, uses function name and args)
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if cache_key:
                    key = cache_key
                else:
                    key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                    key = hashlib.md5(key_data.encode()).hexdigest()
                
                # Try to get from cache
                cached_result = self.cache.get(key)
                if cached_result is not None:
                    return cached_result
                
                # Compute and cache result
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                
                # Store in cache
                self.cache.put(key, result)
                
                # Record metrics
                metrics = PerformanceMetrics(
                    operation_name=func.__name__,
                    execution_time=execution_time,
                    cache_misses=1
                )
                self.metrics_history.append(metrics)
                
                return result
            
            return wrapper
        return decorator
    
    def lazy_operation(self, computation_id: str = None):
        """
        Decorator for lazy evaluation of operations.
        
        Args:
            computation_id: Custom computation ID
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.lazy_evaluator is None:
                    return func(*args, **kwargs)
                
                comp_id = computation_id or f"{func.__name__}_{id(args)}"
                
                # Defer computation
                self.lazy_evaluator.defer(comp_id, func, *args, **kwargs)
                
                # Return evaluator function
                return lambda: self.lazy_evaluator.evaluate(comp_id)
            
            return wrapper
        return decorator
    
    def parallel_batch_operation(self, use_processes: bool = False, 
                                chunk_size: Optional[int] = None):
        """
        Decorator for parallel batch operations.
        
        Args:
            use_processes: Whether to use processes instead of threads
            chunk_size: Size of processing chunks
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(items: List[Any], *args, **kwargs):
                if self.parallel_processor is None or len(items) < 10:
                    # Fall back to sequential processing for small batches
                    return [func(item, *args, **kwargs) for item in items]
                
                # Create partial function with additional arguments
                def partial_func(item):
                    return func(item, *args, **kwargs)
                
                if use_processes:
                    return self.parallel_processor.process_batch_multiprocess(
                        partial_func, items, chunk_size
                    )
                else:
                    return self.parallel_processor.process_batch_threaded(
                        partial_func, items, chunk_size
                    )
            
            return wrapper
        return decorator
    
    def monitor_performance(self, operation_name: str):
        """
        Decorator for monitoring operation performance.
        
        Args:
            operation_name: Name of the operation being monitored
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Take memory snapshot before
                memory_before = self.memory_optimizer.get_memory_usage()
                
                # Execute with timing
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                
                # Take memory snapshot after
                memory_after = self.memory_optimizer.get_memory_usage()
                memory_used = memory_after - memory_before
                
                # Record metrics
                metrics = PerformanceMetrics(
                    operation_name=operation_name,
                    execution_time=execution_time,
                    memory_usage=memory_used
                )
                
                self.metrics_history.append(metrics)
                
                # Auto-optimization checks
                self._auto_optimize()
                
                return result
            
            return wrapper
        return decorator
    
    def _auto_optimize(self) -> None:
        """Perform automatic optimizations based on current state."""
        if not self.optimization_enabled:
            return
        
        current_time = time.time()
        current_memory = self.memory_optimizer.get_memory_usage()
        
        # Auto garbage collection if memory usage is high
        if current_memory > self.auto_gc_threshold:
            collected = self.memory_optimizer.force_garbage_collection()
            if collected['total_collected'] > 0:
                print(f"Auto-GC: Collected {collected['total_collected']} objects")
        
        # Auto cache cleanup
        if current_time - self.last_cache_cleanup > self.auto_cache_cleanup_interval:
            if hasattr(self.cache, 'cleanup_expired'):
                self.cache.cleanup_expired()
            
            if self.lazy_evaluator:
                self.lazy_evaluator.cleanup_old()
            
            self.last_cache_cleanup = current_time
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {'error': 'No performance data available'}
        
        # Calculate averages
        total_operations = len(self.metrics_history)
        avg_execution_time = sum(m.execution_time for m in self.metrics_history) / total_operations
        avg_memory_usage = sum(m.memory_usage for m in self.metrics_history) / total_operations
        total_cache_hits = sum(m.cache_hits for m in self.metrics_history)
        total_cache_misses = sum(m.cache_misses for m in self.metrics_history)
        
        # Cache statistics
        cache_stats = {}
        if hasattr(self.cache, 'stats'):
            cache_stats = self.cache.stats()
        
        # Memory analysis
        memory_analysis = self.memory_optimizer.analyze_memory_growth()
        
        return {
            'total_operations': total_operations,
            'average_execution_time': avg_execution_time,
            'average_memory_usage': avg_memory_usage,
            'cache_hit_ratio': total_cache_hits / (total_cache_hits + total_cache_misses) if (total_cache_hits + total_cache_misses) > 0 else 0,
            'cache_statistics': cache_stats,
            'memory_analysis': memory_analysis,
            'optimization_enabled': self.optimization_enabled,
            'components_active': {
                'cache': self.cache is not None,
                'lazy_evaluation': self.lazy_evaluator is not None,
                'parallel_processing': self.parallel_processor is not None,
                'memory_optimization': True
            }
        }
    
    def optimize_for_scenario(self, scenario: str) -> None:
        """
        Optimize settings for specific use case scenarios.
        
        Args:
            scenario: Optimization scenario ('memory_constrained', 'speed_critical', 'balanced')
        """
        if scenario == 'memory_constrained':
            # Prioritize memory usage over speed
            self.cache = LRUCache(capacity=100)  # Smaller cache
            self.auto_gc_threshold = 50  # More aggressive GC
            if self.parallel_processor:
                self.parallel_processor.max_workers = 2  # Fewer workers
        
        elif scenario == 'speed_critical':
            # Prioritize speed over memory usage
            self.cache = AdaptiveCache(capacity=5000)  # Larger cache
            self.auto_gc_threshold = 500  # Less aggressive GC
            if self.parallel_processor:
                self.parallel_processor.max_workers = multiprocessing.cpu_count() * 2
        
        elif scenario == 'balanced':
            # Balanced optimization
            self.cache = AdaptiveCache(capacity=1000)
            self.auto_gc_threshold = 100
            if self.parallel_processor:
                self.parallel_processor.max_workers = multiprocessing.cpu_count()
        
        else:
            raise ValueError(f"Unknown optimization scenario: {scenario}")
    
    def cleanup(self) -> None:
        """Clean up resources and shutdown components."""
        if self.cache:
            self.cache.clear()
        
        if self.lazy_evaluator:
            self.lazy_evaluator.cleanup_old(0)  # Remove all
        
        if self.parallel_processor:
            self.parallel_processor.shutdown()
        
        # Force garbage collection
        self.memory_optimizer.force_garbage_collection()
        
        self.metrics_history.clear()


# Global performance optimizer instance
_global_optimizer = None

def get_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer

def enable_performance_optimization(cache_strategy: str = 'adaptive',
                                  lazy_evaluation: bool = True,
                                  parallel_processing: bool = True) -> PerformanceOptimizer:
    """
    Enable global performance optimization.
    
    Args:
        cache_strategy: Cache strategy to use
        lazy_evaluation: Enable lazy evaluation
        parallel_processing: Enable parallel processing
        
    Returns:
        Configured performance optimizer
    """
    global _global_optimizer
    _global_optimizer = PerformanceOptimizer(
        cache_strategy=cache_strategy,
        enable_lazy_eval=lazy_evaluation,
        enable_parallel=parallel_processing
    )
    return _global_optimizer

# Convenience decorators using global optimizer
def cached(cache_key: str = None):
    """Convenience decorator for caching using global optimizer."""
    return get_optimizer().cached_operation(cache_key)

def lazy(computation_id: str = None):
    """Convenience decorator for lazy evaluation using global optimizer."""
    return get_optimizer().lazy_operation(computation_id)

def parallel_batch(use_processes: bool = False, chunk_size: Optional[int] = None):
    """Convenience decorator for parallel batch processing using global optimizer."""
    return get_optimizer().parallel_batch_operation(use_processes, chunk_size)

def monitor(operation_name: str):
    """Convenience decorator for performance monitoring using global optimizer."""
    return get_optimizer().monitor_performance(operation_name)