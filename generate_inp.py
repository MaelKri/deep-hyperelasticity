# generate_inp.py
from pathlib import Path

base_inp = Path("model_base.inp")
out_dir = Path("inp_files")
out_dir.mkdir(exist_ok=True)

gc_values = [0.01, 0.05, 0.1, 0.2, 0.5]  # valeurs à tester

inp_text = base_inp.read_text()

for gc in gc_values:
    jobname = f"job_Gc_{gc}".replace(".", "p")
    new_inp_text = inp_text.replace("<<G_C>>", str(gc))
    new_inp = out_dir / f"{jobname}.inp"
    new_inp.write_text(new_inp_text)
    print(f"Created {new_inp}")
