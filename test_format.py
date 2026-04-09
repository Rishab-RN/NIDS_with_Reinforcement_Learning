"""Capture and validate the exact inference.py output format."""
import subprocess, sys, os

result = subprocess.run(
    [sys.executable, "inference.py", "--url", "http://localhost:8000"],
    capture_output=True, text=True, timeout=300,
    env={**os.environ, "HF_TOKEN": "dummy", "PYTHONPATH": "."}
)

lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
print(f"Total output lines: {len(lines)}")
print()

starts = [l for l in lines if l.startswith("[START]")]
steps  = [l for l in lines if l.startswith("[STEP]")]
ends   = [l for l in lines if l.startswith("[END]")]

print(f"[START] lines: {len(starts)}")
print(f"[STEP]  lines: {len(steps)}")
print(f"[END]   lines: {len(ends)}")
print()

# Show first START, first STEP, first END
print("--- First [START] ---")
print(starts[0] if starts else "NONE")
print()
print("--- First [STEP] ---")
print(steps[0] if steps else "NONE")
print()
print("--- First [END] ---")
print(ends[0] if ends else "NONE")
print()
print("--- Last [END] ---")
print(ends[-1] if ends else "NONE")
print()

# Validate format
errors = []
for l in starts:
    for key in ["task=", "env=", "model="]:
        if key not in l:
            errors.append(f"START missing {key}: {l}")
for l in steps:
    for key in ["step=", "action=", "reward=", "done=", "error="]:
        if key not in l:
            errors.append(f"STEP missing {key}: {l}")
for l in ends:
    for key in ["success=", "steps=", "score=", "rewards="]:
        if key not in l:
            errors.append(f"END missing {key}: {l}")

if errors:
    print("ERRORS:")
    for e in errors:
        print(f"  {e}")
else:
    print("ALL FORMAT CHECKS PASSED!")
