# param_study_run.py
import subprocess
import shutil
import os
import csv
from pathlib import Path

# Configuration
base_inp = Path("model_base.inp")   # ton .inp contenant token <<G_C>>
out_dir = Path("param_results")
out_dir.mkdir(exist_ok=True)
gc_values = [0.01, 0.05, 0.1, 0.2, 0.5]   # ajuster en unités du modèle
abaqus_cmd = "abaqus"   # commande de lancement (selon ton environnement)

results_csv = out_dir / "results_summary.csv"
with open(results_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Gc", "jobname", "peak_reaction", "disp_at_peak", "notes"])

for gc in gc_values:
    jobname = f"job_Gc_{gc}".replace(".", "p")   # e.g. job_Gc_0p1
    inp_text = base_inp.read_text()
    new_inp_text = inp_text.replace("<<G_C>>", str(gc))
    new_inp = out_dir / f"{jobname}.inp"
    new_inp.write_text(new_inp_text)

    # soumettre le job (Abaqus créera job_Gc_*.odb)
    print(f"Submitting {jobname} (Gc={gc}) ...")
    # appel système: abaqus job=<jobname> input=<path> interactive
    command = [abaqus_cmd, "job=" + jobname, "input=" + str(new_inp)]
    # Si tu veux utiliser plusieurs CPU: add "cpus=4" or similar.
    subprocess.run(command, check=True)

    # Post-traitement : on lance un script Abaqus Python qui ouvrira jobname.odb
    postop_script = out_dir / f"post_{jobname}.py"
    postop_script.write_text(f"""
from odbAccess import openOdb
odb = openOdb(path='{jobname}.odb')
step = odb.steps[odb.steps.keys()[0]]
# Affiche les historyRegion keys pour identifier ce que tu veux extraire
print('Available history regions:', step.historyRegions.keys())
# (Exemple) extraire la reaction au noeud de référence -- ADAPTE le nom:
# hr = step.historyRegions['Node PART-1-1.REF_NODE']
# rf = hr.historyOutputs['RF1'].data
# peak = max(abs(v[1]) for v in rf)
# print('peak reaction =', peak)
odb.close()
""")
    
    
    # Exécute le script via abaqus python
    subprocess.run([abaqus_cmd, "python", str(postop_script)], check=True)

    # NOTE: adapte le script de post-traitement pour extraire tes métriques et les sauvegarder dans CSV
    # Ici on n'écrit rien dans le CSV : remplace par appel à un script plus complet
