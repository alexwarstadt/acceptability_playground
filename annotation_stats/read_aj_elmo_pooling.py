import os
from os.path import isfile, join

log_path = "/Users/alexwarstadt/Workspace/acceptability_playground/annotation_stats/aj_elmo_pooling_old_11-21-18/"

table = {}

for log_file in os.listdir(log_path):
    if isfile(join(log_path, log_file)):
        lines = [line for line in open(join(log_path, log_file))]
        last_line = lines[-1]
        experiment_name = lines[4].split(" ")[2].strip()
        mcc, acc = last_line.split(" ")[2], last_line.split(" ")[4]
        entry = {"MCC": mcc, "Accuracy": acc}
        table[experiment_name] = entry

for out_file in os.listdir(join(log_path, "outputs")):
    predictions = [1 if float(line) > 0.5 else 0 for line in open(join(log_path, "outputs", out_file))]
    exp_name = log_file[:-4]
    table[exp_name]["Outputs"] = predictions

pass
