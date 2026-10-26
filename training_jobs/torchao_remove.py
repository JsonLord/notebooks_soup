"""Remove torchao from the T4 session (incompatible with trl 1.16)."""
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True, text=True)
print(r.stdout[-200:] if r.stdout else "")
print(r.stderr[-200:] if r.stderr else "")
print("TORCHAO_REMOVAL_DONE rc=%d" % r.returncode)
