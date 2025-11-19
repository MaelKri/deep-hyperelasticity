# submit_jobs.py
import subprocess
from pathlib import Path

inp_dir = Path("C:/Users/aherve2023/Documents/Abaqus/inp_files")
abaqus_cmd = "abaqus"  # adapter selon ton environnement

for inp_file in inp_dir.glob("*.inp"):
    jobname = inp_file.stem
    print(f"Submitting {jobname} ...")
    command = [abaqus_cmd, "job=" + jobname, "input=" + str(inp_file)]
    subprocess.run(command, check=True)
