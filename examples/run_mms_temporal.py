"""Temporal convergence study — convenience wrapper (see run_mms_spatial.py).

The temporal study is embedded at the bottom of run_mms_spatial.py.
Run that script for the full convergence table.
"""
import subprocess, sys
subprocess.run([sys.executable,
                __file__.replace('run_mms_temporal.py', 'run_mms_spatial.py')],
               check=True)
