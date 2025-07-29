#!/usr/bin/env python3
"""
Memory monitoring script for VERL PPO training.
Helps detect memory bottlenecks and potential OOM conditions.

Usage:
    python memory_monitor.py --log-interval 30 --output memory_log.csv
"""

import argparse
import csv
import time
import subprocess
import psutil
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional


def get_gpu_memory_info() -> List[Dict]:
    """Get GPU memory information using nvidia-smi."""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                gpu_info.append({
                    'gpu_id': int(parts[0]),
                    'name': parts[1],
                    'memory_used_mb': int(parts[2]),
                    'memory_total_mb': int(parts[3]),
                    'memory_free_mb': int(parts[4]),
                    'gpu_util_percent': int(parts[5]),
                    'memory_util_percent': round(int(parts[2]) / int(parts[3]) * 100, 1)
                })
        return gpu_info
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Error getting GPU info: {e}")
        return []


def get_system_memory_info() -> Dict:
    """Get system memory information."""
    memory = psutil.virtual_memory()
    return {
        'system_memory_used_gb': round(memory.used / (1024**3), 2),
        'system_memory_total_gb': round(memory.total / (1024**3), 2),
        'system_memory_percent': memory.percent,
        'system_memory_available_gb': round(memory.available / (1024**3), 2)
    }


def get_process_memory_info() -> Dict:
    """Get memory information for current process and its children."""
    current_process = psutil.Process()
    try:
        memory_info = current_process.memory_info()
        children = current_process.children(recursive=True)
        total_rss = memory_info.rss
        total_vms = memory_info.vms
        
        for child in children:
            try:
                child_memory = child.memory_info()
                total_rss += child_memory.rss
                total_vms += child_memory.vms
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return {
            'process_memory_rss_gb': round(total_rss / (1024**3), 2),
            'process_memory_vms_gb': round(total_vms / (1024**3), 2),
            'num_child_processes': len(children)
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {
            'process_memory_rss_gb': 0,
            'process_memory_vms_gb': 0,
            'num_child_processes': 0
        }


def check_oom_risk(gpu_info: List[Dict], system_info: Dict, threshold: float = 90.0) -> List[str]:
    """Check for OOM risk conditions."""
    warnings = []
    
    # Check GPU memory usage
    for gpu in gpu_info:
        if gpu['memory_util_percent'] > threshold:
            warnings.append(f"GPU {gpu['gpu_id']}: {gpu['memory_util_percent']}% memory usage (>{threshold}%)")
    
    # Check system memory usage
    if system_info['system_memory_percent'] > threshold:
        warnings.append(f"System memory: {system_info['system_memory_percent']}% usage (>{threshold}%)")
    
    return warnings


def monitor_memory(log_file: Optional[str] = None, interval: int = 30, oom_threshold: float = 90.0):
    """Main memory monitoring loop."""
    print(f"Starting memory monitoring (interval: {interval}s, OOM threshold: {oom_threshold}%)")
    print("Press Ctrl+C to stop")
    
    # Prepare CSV logging
    csv_writer = None
    csv_file = None
    if log_file:
        csv_file = open(log_file, 'w', newline='')
        fieldnames = [
            'timestamp', 'gpu_id', 'gpu_name', 'gpu_memory_used_mb', 'gpu_memory_total_mb',
            'gpu_memory_util_percent', 'gpu_util_percent', 'system_memory_used_gb',
            'system_memory_total_gb', 'system_memory_percent', 'process_memory_rss_gb',
            'process_memory_vms_gb', 'num_child_processes', 'warnings'
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
    
    try:
        while True:
            timestamp = datetime.now().isoformat()
            gpu_info = get_gpu_memory_info()
            system_info = get_system_memory_info()
            process_info = get_process_memory_info()
            warnings = check_oom_risk(gpu_info, system_info, oom_threshold)
            
            # Print summary
            print(f"\n[{timestamp}]")
            print(f"System Memory: {system_info['system_memory_used_gb']:.1f}/{system_info['system_memory_total_gb']:.1f} GB ({system_info['system_memory_percent']:.1f}%)")
            print(f"Process Memory: RSS={process_info['process_memory_rss_gb']:.1f} GB, VMS={process_info['process_memory_vms_gb']:.1f} GB, Children={process_info['num_child_processes']}")
            
            for gpu in gpu_info:
                print(f"GPU {gpu['gpu_id']} ({gpu['name']}): {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB ({gpu['memory_util_percent']:.1f}%) GPU:{gpu['gpu_util_percent']}%")
            
            if warnings:
                print("⚠️  OOM RISK DETECTED:")
                for warning in warnings:
                    print(f"   {warning}")
            
            # Log to CSV
            if csv_writer:
                for gpu in gpu_info:
                    row = {
                        'timestamp': timestamp,
                        'gpu_id': gpu['gpu_id'],
                        'gpu_name': gpu['name'],
                        'gpu_memory_used_mb': gpu['memory_used_mb'],
                        'gpu_memory_total_mb': gpu['memory_total_mb'],
                        'gpu_memory_util_percent': gpu['memory_util_percent'],
                        'gpu_util_percent': gpu['gpu_util_percent'],
                        'warnings': '; '.join(warnings)
                    }
                    row.update(system_info)
                    row.update(process_info)
                    csv_writer.writerow(row)
                csv_file.flush()
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMemory monitoring stopped.")
    finally:
        if csv_file:
            csv_file.close()
            print(f"Memory log saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Monitor GPU and system memory usage during training")
    parser.add_argument('--log-interval', type=int, default=30, help='Logging interval in seconds (default: 30)')
    parser.add_argument('--output', type=str, help='Output CSV file for logging')
    parser.add_argument('--oom-threshold', type=float, default=90.0, help='Memory usage threshold for OOM warnings (default: 90.0)')
    
    args = parser.parse_args()
    
    monitor_memory(args.output, args.log_interval, args.oom_threshold)


if __name__ == "__main__":
    main() 