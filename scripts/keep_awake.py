"""Keep Windows from idle-sleeping while the pipeline process is alive.

Uses SetThreadExecutionState (the same call media players use); nothing is changed in
the power plan and the request disappears when this process exits.
Usage: python scripts/keep_awake.py <pipeline_pid>
"""
import ctypes
import subprocess
import sys
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in out


if __name__ == "__main__":
    pid = int(sys.argv[1])
    while alive(pid):
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        time.sleep(60)
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
