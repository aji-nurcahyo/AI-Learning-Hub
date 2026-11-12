import glob
import json
import os
import subprocess
import sys
import time

script_dir = os.path.join("content", "level-4-advanced-ai", "chapter-38", "code")
scripts = sorted(glob.glob(os.path.join(script_dir, "*.py")))

print(f"Testing {len(scripts)} scripts in {script_dir}...")
results = {}
total_start = time.time()
failed = False

for s in scripts:
    s_name = os.path.basename(s)
    t0 = time.time()
    res = subprocess.run([sys.executable, s], capture_output=True, text=True)
    dt = round(time.time() - t0, 3)
    passed = (res.returncode == 0)
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {s_name} in {dt}s")
    if not passed:
        print("ERR:", res.stderr[:500])
        failed = True
    results[s_name] = {
        "status": "passed" if passed else "failed",
        "duration_seconds": dt,
        "return_code": res.returncode,
        "stdout_snippet": res.stdout[:500].strip(),
    }

total_duration = round(time.time() - total_start, 2)
print(f"All completed in {total_duration}s. Failed: {failed}")

out_json = os.path.join("content", "level-4-advanced-ai", "chapter-38", "experiments", "results_ch38.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "chapter": 38,
        "title": "Multimodal AI Systems",
        "total_scripts": len(scripts),
        "total_duration_seconds": total_duration,
        "all_passed": not failed,
        "scripts": results,
    }, f, indent=2)
print(f"Saved results to {out_json}")

if failed:
    sys.exit(1)

