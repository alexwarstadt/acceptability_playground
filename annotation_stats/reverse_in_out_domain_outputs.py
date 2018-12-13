import os
from os.path import join


out_dir = "/Users/alexwarstadt/Workspace/acceptability_playground/annotation_stats/aj_elmo_pooling_old_11-21-18/outputs"

for out_file in os.listdir(out_dir):
    f = open(join(out_dir, out_file))
    lines = [l for l in f]
    f.close()
    out_lines = lines[:516]
    in_lines = lines[516:]
    write_file = open(join(out_dir, out_file), "w")
    for l in in_lines:
        write_file.write(l)
    for l in out_lines:
        write_file.write(l)
    write_file.close()
