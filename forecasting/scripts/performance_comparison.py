# -*- coding: utf-8 -*-
"""
Performance Comparison Script
Compares the original sequential script with the parallel optimized version
"""

import time
import subprocess
import sys
import os
from multiprocessing import cpu_count

def run_script(script_name, description):
    """Run a script and measure execution time"""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            print(f"⏱️  Execution time: {execution_time:.2f} seconds")
            return execution_time, True
        else:
            print(f"❌ {description} failed!")
            print(f"Error: {result.stderr}")
            return execution_time, False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after 5 minutes")
        return 300, False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return 0, False

def main():
    """Compare performance between original and parallel scripts"""
    print("🚀 MULTITARGET PARALLEL PERFORMANCE COMPARISON")
    print("="*60)
    print(f"🖥️  System has {cpu_count()} CPU cores")
    
    # Check if both scripts exist
    original_script = "simple_forecasting_script.py"
    parallel_script = "simple_forecasting_script_parallel.py"
    
    if not os.path.exists(original_script):
        print(f"❌ {original_script} not found!")
        return
    
    if not os.path.exists(parallel_script):
        print(f"❌ {parallel_script} not found!")
        return
    
    # Run original script
    original_time, original_success = run_script(original_script, "Original Sequential Script")
    
    # Run parallel script
    parallel_time, parallel_success = run_script(parallel_script, "Multitarget Parallel Script")
    
    # Compare results
    print(f"\n{'='*60}")
    print("MULTITARGET PARALLEL PERFORMANCE RESULTS")
    print(f"{'='*60}")
    
    if original_success and parallel_success:
        speedup = original_time / parallel_time
        time_saved = original_time - parallel_time
        efficiency = (time_saved / original_time) * 100
        
        print(f"📊 Original Script Time: {original_time:.2f} seconds")
        print(f"📊 Multitarget Parallel Script Time: {parallel_time:.2f} seconds")
        print(f"⚡ Speedup: {speedup:.2f}x faster")
        print(f"⏱️  Time Saved: {time_saved:.2f} seconds")
        print(f"📈 Efficiency Improvement: {efficiency:.1f}%")
        
        # Calculate theoretical maximum speedup
        max_cores = cpu_count()
        theoretical_max = min(max_cores, 11)  # 11 target variables
        
        print(f"\n🔬 Analysis:")
        print(f"   - Available CPU cores: {max_cores}")
        print(f"   - Target variables: 11")
        print(f"   - Theoretical max speedup: {theoretical_max:.1f}x")
        print(f"   - Achieved speedup: {speedup:.1f}x")
        print(f"   - Parallel efficiency: {(speedup/theoretical_max)*100:.1f}%")
        
        if speedup > 2.0:
            print("🎉 Excellent performance improvement achieved!")
        elif speedup > 1.5:
            print("✅ Good performance improvement achieved!")
        elif speedup > 1.1:
            print("⚠️  Moderate performance improvement - room for optimization")
        else:
            print("❌ Limited performance improvement - check system resources")
            
    else:
        print("❌ Could not complete performance comparison due to script failures")
    
    print(f"\n💡 Multitarget Parallel Optimization Features:")
    print(f"   ✅ Chunked processing - multiple targets per core")
    print(f"   ✅ Parallel model training across all cores")
    print(f"   ✅ Parallel EBM training with multitarget chunks")
    print(f"   ✅ Parallel forecasting generation")
    print(f"   ✅ Vectorized data processing")
    print(f"   ✅ Automatic load balancing across cores")
    
    print(f"\n🔧 Configuration Tips:")
    print(f"   - Each core handles multiple targets for better utilization")
    print(f"   - Chunk size automatically calculated based on cores/targets")
    print(f"   - Memory efficient with shared data across processes")
    print(f"   - Optimal for systems with 4+ CPU cores")

if __name__ == "__main__":
    main()
