# post_process.py
import odbAccess 
from odbAccess import openOdb
import csv
from pathlib import Path

odb_dir = Path(".")  # adapter si  odb sont ailleurs
results_csv = Path("results_summary.csv")

gc_values = []       # pour info dans CSV
peak_loads = []
damaged_elems = []

for odb_file in odb_dir.glob("job_Gc_*.odb"):
    jobname = odb_file.stem
    gc_str = jobname.split("_")[2].replace("p", ".")
    gc = float(gc_str)

    odb = openOdb(path=str(odb_file))
    step = odb.steps[odb.steps.keys()[0]]

    # Exemple : récupérer peak reaction au noeud REF_NODE
    hr_key = [k for k in step.historyRegions.keys() if 'REF_NODE' in k]
    if hr_key:
        hr = step.historyRegions[hr_key[0]]
        rf = hr.historyOutputs['RF1'].data
        peak = max(abs(v[1]) for v in rf)
    else:
        peak = None

    # Exemple : compter éléments endommagés
    damage_count = 0
    try:
        for instName, inst in odb.rootAssembly.instances.items():
            fo = step.frames[-1].fieldOutputs
            if 'SDEG' in fo.keys():
                sdeg = fo['SDEG']
                for v in sdeg.values:
                    if v.data > 0.9:
                        damage_count += 1
    except:
        pass

    odb.close()

    gc_values.append(gc)
    peak_loads.append(peak)
    damaged_elems.append(damage_count)

# Sauvegarde CSV
with open(results_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Gc", "PeakReaction", "DamagedElements"])
    for gc, peak, damage in zip(gc_values, peak_loads, damaged_elems):
        writer.writerow([gc, peak, damage])

print(f"Results saved in {results_csv}")
