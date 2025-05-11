# Script to terminate any running Python processes that may be stuck in an infinite loop or are no longer responding.
# This can be useful during development when debugging long-running or stuck Python scripts.
# The script iterates through all processes and kills any Python process, excluding the current script's process.

import psutil
import os

def kill_all_python_processes():
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'python' in proc.info['name'].lower() and proc.pid != current_pid:
                print(f"Killing PID {proc.pid}: {proc.info['name']}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

if __name__ == "__main__":
    kill_all_python_processes()