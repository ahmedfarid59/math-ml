"""
Performance Benchmarks
======================

Comprehensive benchmark suite for the MathML parser performance
optimization engine. Tests caching, parallel processing, memory
optimization, and overall parsing performance.

Run this module to benchmark all performance features and generate
detailed performance reports.
"""

import time
import random
import math
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from mathml_parser.performance import (
        PerformanceOptimizer, LRUCache, TTLCache, AdaptiveCache,
        LazyEvaluator, MemoryOptimizer, ParallelProcessor, PerformanceBenchmark,
        cached, lazy, parallel_batch, monitor
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the performance module is properly installed.")
    sys.exit(1)


class MathMLBenchmarkSuite:
    """Comprehensive benchmark suite for MathML parser performance."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.optimizer = PerformanceOptimizer()
        self.benchmark = PerformanceBenchmark()
        self.test_expressions = self._generate_test_expressions()
        self.results = {}
    
    def _generate_test_expressions(self) -> List[str]:
        """Generate test mathematical expressions for benchmarking."""
        expressions = []
        
        # Simple expressions
        for i in range(100):
            a, b = random.randint(1, 100), random.randint(1, 100)
            expressions.append(f"{a} + {b}")
            expressions.append(f"{a} * {b}")
            expressions.append(f"{a}^{b % 5 + 1}")
        
        # Complex expressions
        for i in range(50):
            expressions.append(f"sin({random.randint(1, 10)}) + cos({random.randint(1, 10)})")
            expressions.append(f"sqrt({random.randint(1, 100)}) + log({random.randint(1, 100)})")
            expressions.append(f"({random.randint(1, 10)} + {random.randint(1, 10)}) / ({random.randint(1, 10)} + {random.randint(1, 10)})")
        
        # Nested expressions
        for i in range(25):
            expressions.append(f"sin(cos(tan({random.randint(1, 5)})))")
            expressions.append(f"((({random.randint(1, 5)} + {random.randint(1, 5)}) * {random.randint(1, 5)}) + {random.randint(1, 5)})")
        
        # Complex numbers
        for i in range(25):
            real = random.randint(1, 10)
            imag = random.randint(1, 10)
            expressions.append(f"{real} + {imag}i")
            expressions.append(f"({real} + {imag}i) * ({random.randint(1, 5)} - {random.randint(1, 5)}i)")
        
        return expressions
    
    def benchmark_cache_strategies(self) -> Dict[str, Any]:
        """Benchmark different cache strategies."""
        print("🚀 Benchmarking Cache Strategies...")
        
        # Test data
        test_keys = [f"key_{i}" for i in range(1000)]
        test_values = [f"value_{i}" * 100 for i in range(1000)]  # Make values larger
        
        strategies = {
            'LRU': LRUCache(capacity=500),
            'TTL': TTLCache(capacity=500, ttl=300),
            'Adaptive': AdaptiveCache(capacity=500, ttl=300)
        }
        
        results = {}
        
        for name, cache in strategies.items():
            print(f"  Testing {name} cache...")
            
            # Fill cache
            start_time = time.perf_counter()
            for key, value in zip(test_keys, test_values):
                cache.put(key, value)
            fill_time = time.perf_counter() - start_time
            
            # Test retrieval (mix of hits and misses)
            hit_keys = random.sample(test_keys[:500], 200)  # Known keys
            miss_keys = [f"miss_key_{i}" for i in range(100)]  # Unknown keys
            lookup_keys = hit_keys + miss_keys
            random.shuffle(lookup_keys)
            
            start_time = time.perf_counter()
            hits, misses = 0, 0
            for key in lookup_keys:
                result = cache.get(key)
                if result is not None:
                    hits += 1
                else:
                    misses += 1
            lookup_time = time.perf_counter() - start_time
            
            results[name] = {
                'fill_time': fill_time,
                'lookup_time': lookup_time,
                'hits': hits,
                'misses': misses,
                'hit_ratio': hits / (hits + misses),
                'cache_size': cache.size()
            }
        
        self.results['cache_strategies'] = results
        return results
    
    def benchmark_lazy_evaluation(self) -> Dict[str, Any]:
        """Benchmark lazy evaluation performance."""
        print("⏳ Benchmarking Lazy Evaluation...")
        
        evaluator = LazyEvaluator()
        
        # Expensive computation function
        def expensive_computation(n: int) -> float:
            """Simulate expensive mathematical computation."""
            result = 0.0
            for i in range(n * 1000):
                result += math.sin(i) * math.cos(i) * math.sqrt(i + 1)
            return result
        
        # Test immediate vs lazy evaluation
        computations = list(range(10, 51))  # 10-50
        
        # Immediate evaluation
        start_time = time.perf_counter()
        immediate_results = []
        for n in computations:
            result = expensive_computation(n)
            immediate_results.append(result)
        immediate_time = time.perf_counter() - start_time
        
        # Lazy evaluation - defer phase
        start_time = time.perf_counter()
        lazy_computation_ids = []
        for i, n in enumerate(computations):
            comp_id = f"computation_{i}"
            evaluator.defer(comp_id, expensive_computation, n)
            lazy_computation_ids.append(comp_id)
        defer_time = time.perf_counter() - start_time
        
        # Lazy evaluation - evaluation phase (partial)
        start_time = time.perf_counter()
        lazy_results = []
        for comp_id in lazy_computation_ids[:len(lazy_computation_ids)//2]:  # Evaluate only half
            result = evaluator.evaluate(comp_id)
            lazy_results.append(result)
        partial_eval_time = time.perf_counter() - start_time
        
        results = {
            'immediate_evaluation_time': immediate_time,
            'lazy_defer_time': defer_time,
            'lazy_partial_evaluation_time': partial_eval_time,
            'computations_deferred': len(computations),
            'computations_evaluated': len(lazy_results),
            'time_saved_ratio': 1 - (defer_time + partial_eval_time) / immediate_time,
            'memory_efficiency': len(lazy_computation_ids) - len(lazy_results)  # Unevaluated computations
        }
        
        self.results['lazy_evaluation'] = results
        return results
    
    def benchmark_parallel_processing(self) -> Dict[str, Any]:
        """Benchmark parallel processing performance."""
        print("⚡ Benchmarking Parallel Processing...")
        
        # CPU-intensive mathematical function
        def complex_math_operation(x: float) -> float:
            """Complex mathematical operation for parallel testing."""
            result = x
            for _ in range(1000):
                result = math.sin(result) + math.cos(result * 2) + math.sqrt(abs(result) + 1)
            return result
        
        # Test data
        test_data = [random.uniform(0, 10) for _ in range(200)]
        
        with ParallelProcessor() as parallel:
            # Sequential processing
            start_time = time.perf_counter()
            sequential_results = [complex_math_operation(x) for x in test_data]
            sequential_time = time.perf_counter() - start_time
            
            # Threaded parallel processing
            start_time = time.perf_counter()
            threaded_results = parallel.process_batch_threaded(complex_math_operation, test_data)
            threaded_time = time.perf_counter() - start_time
            
            # Process-based parallel processing
            start_time = time.perf_counter()
            process_results = parallel.process_batch_multiprocess(complex_math_operation, test_data)
            process_time = time.perf_counter() - start_time
            
            # Map-reduce operation
            def sum_reduce(a, b):
                return a + b
            
            start_time = time.perf_counter()
            map_reduce_result = parallel.map_reduce(
                complex_math_operation, sum_reduce, test_data[:50], use_processes=False
            )
            map_reduce_time = time.perf_counter() - start_time
        
        results = {
            'data_size': len(test_data),
            'sequential_time': sequential_time,
            'threaded_time': threaded_time,
            'process_time': process_time,
            'map_reduce_time': map_reduce_time,
            'threaded_speedup': sequential_time / threaded_time,
            'process_speedup': sequential_time / process_time,
            'results_consistent': (
                sequential_results == threaded_results == process_results
            )
        }
        
        self.results['parallel_processing'] = results
        return results
    
    def benchmark_memory_optimization(self) -> Dict[str, Any]:
        """Benchmark memory optimization features."""
        print("💾 Benchmarking Memory Optimization...")
        
        memory_optimizer = MemoryOptimizer()
        
        # Take initial snapshot
        initial_snapshot = memory_optimizer.take_snapshot("initial")
        
        # Create memory-intensive data structures
        large_data = []
        for i in range(10000):
            # Create various data structures
            data_item = {
                'id': i,
                'values': [random.random() for _ in range(100)],
                'text': f"test_string_{i}" * 10,
                'nested': {
                    'level1': {
                        'level2': [i, i*2, i*3] * 10
                    }
                }
            }
            large_data.append(data_item)
            
            # Register some for weak reference tracking
            if i % 100 == 0:
                memory_optimizer.register_weak_reference(data_item)
        
        # Take snapshot after data creation
        data_snapshot = memory_optimizer.take_snapshot("after_data_creation")
        
        # Perform garbage collection
        gc_stats = memory_optimizer.force_garbage_collection()
        
        # Take snapshot after GC
        gc_snapshot = memory_optimizer.take_snapshot("after_gc")
        
        # Clean up data and test weak references
        del large_data[::2]  # Delete half the data
        weak_ref_cleaned = memory_optimizer.cleanup_weak_references()
        
        # Final snapshot
        cleanup_snapshot = memory_optimizer.take_snapshot("after_cleanup")
        
        # Analyze memory growth
        growth_analysis = memory_optimizer.analyze_memory_growth()
        
        results = {
            'initial_memory_mb': initial_snapshot['memory_usage_mb'],
            'peak_memory_mb': data_snapshot['memory_usage_mb'],
            'after_gc_memory_mb': gc_snapshot['memory_usage_mb'],
            'final_memory_mb': cleanup_snapshot['memory_usage_mb'],
            'memory_growth_mb': data_snapshot['memory_usage_mb'] - initial_snapshot['memory_usage_mb'],
            'gc_saved_mb': data_snapshot['memory_usage_mb'] - gc_snapshot['memory_usage_mb'],
            'cleanup_saved_mb': gc_snapshot['memory_usage_mb'] - cleanup_snapshot['memory_usage_mb'],
            'gc_objects_collected': gc_stats['total_collected'],
            'weak_references_cleaned': weak_ref_cleaned,
            'growth_analysis': growth_analysis
        }
        
        self.results['memory_optimization'] = results
        return results
    
    def benchmark_integrated_performance(self) -> Dict[str, Any]:
        """Benchmark integrated performance with all optimizations."""
        print("🔧 Benchmarking Integrated Performance...")
        
        # Create optimizers with different configurations
        optimizers = {
            'baseline': PerformanceOptimizer(
                cache_strategy='lru',
                enable_lazy_eval=False,
                enable_parallel=False
            ),
            'cached_only': PerformanceOptimizer(
                cache_strategy='adaptive',
                enable_lazy_eval=False,
                enable_parallel=False
            ),
            'lazy_only': PerformanceOptimizer(
                cache_strategy='lru',
                enable_lazy_eval=True,
                enable_parallel=False
            ),
            'parallel_only': PerformanceOptimizer(
                cache_strategy='lru',
                enable_lazy_eval=False,
                enable_parallel=True
            ),
            'fully_optimized': PerformanceOptimizer(
                cache_strategy='adaptive',
                enable_lazy_eval=True,
                enable_parallel=True
            )
        }
        
        # Mathematical parsing simulation function
        def simulate_mathml_parsing(expression: str) -> Dict[str, Any]:
            """Simulate complex MathML parsing operation."""
            # Simulate parsing complexity based on expression length and complexity
            complexity = len(expression) + expression.count('(') * 2 + expression.count('sin') * 3
            
            # Simulate computation time
            time.sleep(complexity * 0.0001)  # Small delay to simulate work
            
            # Return mock parsed result
            return {
                'expression': expression,
                'complexity': complexity,
                'tokens': len(expression.split()),
                'parsed_tree': f"tree_for_{hash(expression)}",
                'evaluation_ready': True
            }
        
        results = {}
        test_expressions = self.test_expressions[:50]  # Use subset for speed
        
        for name, optimizer in optimizers.items():
            print(f"  Testing {name} configuration...")
            
            # Decorate function with current optimizer's features
            if 'cached' in name or 'fully' in name:
                parsing_func = optimizer.cached_operation()(simulate_mathml_parsing)
            else:
                parsing_func = simulate_mathml_parsing
            
            if 'parallel' in name or 'fully' in name:
                # Use parallel batch processing
                parsing_func = optimizer.parallel_batch_operation()(lambda expr: parsing_func(expr))
                
                start_time = time.perf_counter()
                batch_results = parsing_func(test_expressions)
                total_time = time.perf_counter() - start_time
                
                successful_parses = len([r for r in batch_results if r.get('evaluation_ready')])
            else:
                # Sequential processing
                start_time = time.perf_counter()
                batch_results = []
                for expr in test_expressions:
                    result = parsing_func(expr)
                    batch_results.append(result)
                total_time = time.perf_counter() - start_time
                
                successful_parses = len([r for r in batch_results if r.get('evaluation_ready')])
            
            # Get performance summary
            perf_summary = optimizer.get_performance_summary()
            
            results[name] = {
                'total_time': total_time,
                'expressions_processed': len(test_expressions),
                'successful_parses': successful_parses,
                'average_time_per_expression': total_time / len(test_expressions),
                'throughput_expr_per_sec': len(test_expressions) / total_time,
                'performance_summary': perf_summary
            }
        
        # Calculate relative improvements
        baseline_time = results['baseline']['total_time']
        for name in results:
            if name != 'baseline':
                results[name]['speedup_vs_baseline'] = baseline_time / results[name]['total_time']
                results[name]['time_improvement_percent'] = (
                    (baseline_time - results[name]['total_time']) / baseline_time * 100
                )
        
        self.results['integrated_performance'] = results
        return results
    
    def benchmark_scenario_optimization(self) -> Dict[str, Any]:
        """Benchmark performance under different optimization scenarios."""
        print("📊 Benchmarking Scenario Optimization...")
        
        scenarios = ['memory_constrained', 'speed_critical', 'balanced']
        
        # Test function that uses various resources
        def resource_intensive_operation(data_size: int) -> Dict[str, Any]:
            """Simulate resource-intensive mathematical operation."""
            # Memory allocation
            data = [random.random() for _ in range(data_size)]
            
            # CPU-intensive computation
            result = sum(math.sin(x) * math.cos(x) for x in data)
            
            # Return result with metadata
            return {
                'result': result,
                'data_size': data_size,
                'memory_used_estimate': data_size * 8,  # bytes for float
                'computation_complexity': data_size
            }
        
        results = {}
        test_sizes = [1000, 5000, 10000]
        
        for scenario in scenarios:
            print(f"  Testing {scenario} scenario...")
            
            optimizer = PerformanceOptimizer()
            optimizer.optimize_for_scenario(scenario)
            
            # Decorate function with monitoring
            monitored_func = optimizer.monitor_performance(f"operation_{scenario}")(
                resource_intensive_operation
            )
            
            scenario_results = []
            start_time = time.perf_counter()
            
            for size in test_sizes:
                result = monitored_func(size)
                scenario_results.append(result)
            
            total_time = time.perf_counter() - start_time
            
            # Get performance summary
            perf_summary = optimizer.get_performance_summary()
            
            results[scenario] = {
                'total_time': total_time,
                'operations_completed': len(test_sizes),
                'average_time_per_operation': total_time / len(test_sizes),
                'scenario_results': scenario_results,
                'performance_summary': perf_summary
            }
            
            # Cleanup
            optimizer.cleanup()
        
        self.results['scenario_optimization'] = results
        return results
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark suites and return comprehensive results."""
        print("🎯 Running Comprehensive Performance Benchmark Suite")
        print("=" * 60)
        
        start_time = time.perf_counter()
        
        # Run individual benchmark suites
        try:
            cache_results = self.benchmark_cache_strategies()
            lazy_results = self.benchmark_lazy_evaluation()
            parallel_results = self.benchmark_parallel_processing()
            memory_results = self.benchmark_memory_optimization()
            integrated_results = self.benchmark_integrated_performance()
            scenario_results = self.benchmark_scenario_optimization()
            
            total_time = time.perf_counter() - start_time
            
            print(f"\n✅ All benchmarks completed in {total_time:.2f} seconds")
            
            # Compile comprehensive results
            comprehensive_results = {
                'benchmark_metadata': {
                    'total_benchmark_time': total_time,
                    'timestamp': time.time(),
                    'test_expressions_count': len(self.test_expressions),
                    'python_version': sys.version,
                    'system_info': self._get_system_info()
                },
                'cache_strategies': cache_results,
                'lazy_evaluation': lazy_results,
                'parallel_processing': parallel_results,
                'memory_optimization': memory_results,
                'integrated_performance': integrated_results,
                'scenario_optimization': scenario_results
            }
            
            self.results = comprehensive_results
            return comprehensive_results
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        info = {
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        try:
            import psutil
            info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 'unknown'
            })
        except ImportError:
            info['psutil_available'] = False
        
        return info
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return "No benchmark results available. Run benchmarks first."
        
        lines = []
        lines.append("MATHML PARSER PERFORMANCE BENCHMARK REPORT")
        lines.append("=" * 60)
        lines.append()
        
        # Metadata
        if 'benchmark_metadata' in self.results:
            meta = self.results['benchmark_metadata']
            lines.append("Benchmark Metadata:")
            lines.append("-" * 30)
            lines.append(f"Total benchmark time: {meta['total_benchmark_time']:.2f}s")
            lines.append(f"Test expressions: {meta['test_expressions_count']}")
            lines.append(f"Python version: {meta['python_version'].split()[0]}")
            if 'system_info' in meta:
                sys_info = meta['system_info']
                if 'cpu_count' in sys_info:
                    lines.append(f"CPU cores: {sys_info['cpu_count']}")
                if 'memory_total_gb' in sys_info:
                    lines.append(f"Total memory: {sys_info['memory_total_gb']:.1f} GB")
            lines.append()
        
        # Cache strategies
        if 'cache_strategies' in self.results:
            lines.append("Cache Strategy Performance:")
            lines.append("-" * 30)
            cache_results = self.results['cache_strategies']
            
            lines.append(f"{'Strategy':<12} {'Hit Ratio':<12} {'Lookup Time':<15} {'Cache Size':<12}")
            lines.append("-" * 55)
            
            for strategy, data in cache_results.items():
                lines.append(
                    f"{strategy:<12} "
                    f"{data['hit_ratio']:<12.2%} "
                    f"{data['lookup_time']:<15.3f}s "
                    f"{data['cache_size']:<12}"
                )
            
            # Find best strategy
            best_strategy = max(cache_results.items(), key=lambda x: x[1]['hit_ratio'])
            lines.append(f"\nBest cache strategy: {best_strategy[0]} "
                        f"(hit ratio: {best_strategy[1]['hit_ratio']:.2%})")
            lines.append()
        
        # Lazy evaluation
        if 'lazy_evaluation' in self.results:
            lines.append("Lazy Evaluation Performance:")
            lines.append("-" * 30)
            lazy_results = self.results['lazy_evaluation']
            
            lines.append(f"Immediate evaluation time: {lazy_results['immediate_evaluation_time']:.3f}s")
            lines.append(f"Lazy defer time: {lazy_results['lazy_defer_time']:.3f}s")
            lines.append(f"Partial evaluation time: {lazy_results['lazy_partial_evaluation_time']:.3f}s")
            lines.append(f"Time saved ratio: {lazy_results['time_saved_ratio']:.2%}")
            lines.append(f"Memory efficiency: {lazy_results['memory_efficiency']} unevaluated computations")
            lines.append()
        
        # Parallel processing
        if 'parallel_processing' in self.results:
            lines.append("Parallel Processing Performance:")
            lines.append("-" * 30)
            parallel_results = self.results['parallel_processing']
            
            lines.append(f"Data size: {parallel_results['data_size']} items")
            lines.append(f"Sequential time: {parallel_results['sequential_time']:.3f}s")
            lines.append(f"Threaded time: {parallel_results['threaded_time']:.3f}s")
            lines.append(f"Process time: {parallel_results['process_time']:.3f}s")
            lines.append(f"Threaded speedup: {parallel_results['threaded_speedup']:.2f}x")
            lines.append(f"Process speedup: {parallel_results['process_speedup']:.2f}x")
            lines.append(f"Results consistent: {parallel_results['results_consistent']}")
            lines.append()
        
        # Memory optimization
        if 'memory_optimization' in self.results:
            lines.append("Memory Optimization Performance:")
            lines.append("-" * 30)
            memory_results = self.results['memory_optimization']
            
            lines.append(f"Initial memory: {memory_results['initial_memory_mb']:.2f} MB")
            lines.append(f"Peak memory: {memory_results['peak_memory_mb']:.2f} MB")
            lines.append(f"After GC: {memory_results['after_gc_memory_mb']:.2f} MB")
            lines.append(f"Final memory: {memory_results['final_memory_mb']:.2f} MB")
            lines.append(f"Memory growth: {memory_results['memory_growth_mb']:.2f} MB")
            lines.append(f"GC saved: {memory_results['gc_saved_mb']:.2f} MB")
            lines.append(f"Cleanup saved: {memory_results['cleanup_saved_mb']:.2f} MB")
            lines.append(f"Objects collected: {memory_results['gc_objects_collected']}")
            lines.append()
        
        # Integrated performance
        if 'integrated_performance' in self.results:
            lines.append("Integrated Performance Comparison:")
            lines.append("-" * 30)
            integrated_results = self.results['integrated_performance']
            
            lines.append(f"{'Configuration':<20} {'Time (s)':<12} {'Speedup':<10} {'Improvement':<12}")
            lines.append("-" * 60)
            
            for config, data in integrated_results.items():
                speedup = data.get('speedup_vs_baseline', 1.0)
                improvement = data.get('time_improvement_percent', 0.0)
                
                lines.append(
                    f"{config:<20} "
                    f"{data['total_time']:<12.3f} "
                    f"{speedup:<10.2f}x "
                    f"{improvement:<12.1f}%"
                )
            
            # Find best configuration
            best_config = min(
                [(k, v) for k, v in integrated_results.items() if 'speedup_vs_baseline' in v],
                key=lambda x: x[1]['total_time']
            )
            if best_config:
                lines.append(f"\nBest configuration: {best_config[0]} "
                            f"({best_config[1]['speedup_vs_baseline']:.2f}x speedup)")
            lines.append()
        
        # Scenario optimization
        if 'scenario_optimization' in self.results:
            lines.append("Scenario Optimization Performance:")
            lines.append("-" * 30)
            scenario_results = self.results['scenario_optimization']
            
            for scenario, data in scenario_results.items():
                lines.append(f"{scenario.replace('_', ' ').title()}:")
                lines.append(f"  Total time: {data['total_time']:.3f}s")
                lines.append(f"  Avg time per operation: {data['average_time_per_operation']:.3f}s")
            lines.append()
        
        # Recommendations
        lines.append("Performance Recommendations:")
        lines.append("-" * 30)
        
        if 'cache_strategies' in self.results:
            best_cache = max(self.results['cache_strategies'].items(), key=lambda x: x[1]['hit_ratio'])
            lines.append(f"• Use {best_cache[0]} caching strategy for best hit ratio")
        
        if 'parallel_processing' in self.results:
            parallel_data = self.results['parallel_processing']
            if parallel_data['threaded_speedup'] > 1.5:
                lines.append("• Enable threaded parallel processing for CPU-intensive operations")
            if parallel_data['process_speedup'] > 1.5:
                lines.append("• Consider process-based parallelism for very large datasets")
        
        if 'lazy_evaluation' in self.results:
            lazy_data = self.results['lazy_evaluation']
            if lazy_data['time_saved_ratio'] > 0.2:
                lines.append("• Enable lazy evaluation for deferred computations")
        
        if 'integrated_performance' in self.results:
            best_integrated = min(
                [(k, v) for k, v in self.results['integrated_performance'].items() 
                 if 'speedup_vs_baseline' in v],
                key=lambda x: x[1]['total_time']
            )
            if best_integrated and best_integrated[1]['speedup_vs_baseline'] > 1.2:
                lines.append(f"• Use {best_integrated[0]} configuration for optimal performance")
        
        lines.append("\n" + "=" * 60)
        lines.append("BENCHMARK REPORT COMPLETE")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def main():
    """Run the performance benchmark suite."""
    print("Starting MathML Parser Performance Benchmark Suite...")
    print()
    
    try:
        # Create and run benchmark suite
        benchmark_suite = MathMLBenchmarkSuite()
        results = benchmark_suite.run_all_benchmarks()
        
        if 'error' in results:
            print(f"Benchmark failed: {results['error']}")
            return 1
        
        # Generate and display report
        print("\n" + "=" * 60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)
        
        report = benchmark_suite.generate_report()
        print(report)
        
        # Optionally save report to file
        report_filename = f"performance_benchmark_report_{int(time.time())}.txt"
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n📄 Report saved to: {report_filename}")
        except Exception as e:
            print(f"\n⚠️  Could not save report to file: {e}")
        
        print("\n✅ Performance benchmark suite completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Benchmark suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())