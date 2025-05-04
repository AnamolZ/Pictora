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
