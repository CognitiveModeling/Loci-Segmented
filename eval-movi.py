import os
import statistics

def extract_value(movi, seed, pattern, folder_name):
    cmd_output = os.popen(f'cat out/{folder_name}_{movi}_seed_{seed} | grep "{pattern}" | grep -v "Adding" | grep -v "UserWarning"').read()
    lines = cmd_output.split("\n")
    for line in lines:
        if pattern in line:
            return float(line.split("│")[-2].strip())
    raise ValueError(f"Pattern '{pattern}' not found in the output!")

def process_folder(folder_name):
    movies = ["movi_c", "movi_d", "movi_e"]

    results = {}
    for movi in movies:
        ious = []
        aris = []

        for seed in range(1, 6):
            val_mean_reconstruction_IoU = extract_value(movi, seed, "val_mean_reconstruction_IoU", folder_name)
            val_mean_reconstruction_IoU_sum = extract_value(movi, seed, "val_mean_reconstruction_IoU_sum", folder_name)
            val_reconstruction_fg_ari = extract_value(movi, seed, "val_reconstruction_seq_fg_ari", folder_name)
            val_reconstruction_fg_ari_sum = extract_value(movi, seed, "val_seq_ari_sum", folder_name)

            ious.append(val_mean_reconstruction_IoU / val_mean_reconstruction_IoU_sum * 100)
            aris.append(val_reconstruction_fg_ari / val_reconstruction_fg_ari_sum * 100)

        results[movi] = { 
            'IoU': {'mean': statistics.mean(ious), 'std': statistics.stdev(ious)},
            'ARI': {'mean': statistics.mean(aris), 'std': statistics.stdev(aris)},
        }
    return results

# Process for each folder
results = {
    "seg-depth": process_folder("eval_segmentation"),
    "reg-depth": process_folder("eval_regularized"),
    "rnd-depth": process_folder("eval_random"),
    "seg": process_folder("eval_no_depth_segmentation"),
    "reg": process_folder("eval_no_depth_regularized"),
    "rnd": process_folder("eval_no_depth_random"),
}

print("                   |                 IoU                  |                 ARI                 ")
print("       Model       |   MOVi-C   |   MOVi-D   |   MOVi-E   |   MOVi-C   |   MOVi-D   |   MOVi-E")

for mode in ["rnd", "reg", "seg"]:
    print(f"Loci-s-depth ({mode})", end="")
    for movi in ["movi_c", "movi_d", "movi_e"]:
        mean = results[f'{mode}-depth'][movi]['IoU']['mean']
        std = results[f'{mode}-depth'][movi]['IoU']['std']
        print(f" | {mean:.1f} ± {std:.1f}", end = "")
    for movi in ["movi_c", "movi_d", "movi_e"]:
        mean = results[f'{mode}-depth'][movi]['ARI']['mean']
        std = results[f'{mode}-depth'][movi]['ARI']['std']
        print(f" | {mean:.1f} ± {std:.1f}", end = "")
    print()

for mode in ["rnd", "reg", "seg"]:
    print(f"Loci-s       ({mode})", end="")
    for movi in ["movi_c", "movi_d", "movi_e"]:
        mean = results[mode][movi]['IoU']['mean']
        std = results[mode][movi]['IoU']['std']
        print(f" | {mean:.1f} ± {std:.1f}", end = "")
    for movi in ["movi_c", "movi_d", "movi_e"]:
        mean = results[mode][movi]['ARI']['mean']
        std = results[mode][movi]['ARI']['std']
        print(f" | {mean:.1f} ± {std:.1f}", end = "")
    print()

