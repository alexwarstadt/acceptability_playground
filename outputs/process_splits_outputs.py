import os
from scipy.stats.stats import pearsonr
import numpy as np




split_0 = [int(line.split("\t")[1]) for line in open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/splits/split_0/test.tsv")]
split_1 = [int(line.split("\t")[1]) for line in open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/splits/split_1/test.tsv")]
split_2 = [int(line.split("\t")[1]) for line in open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/splits/split_2/test.tsv")]
split_3 = [int(line.split("\t")[1]) for line in open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/splits/split_3/test.tsv")]
split_4 = [int(line.split("\t")[1]) for line in open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/splits/split_4/test.tsv")]
split_5 = [int(line.split("\t")[1]) for line in open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/splits/split_5/test.tsv")]
split_6 = [int(line.split("\t")[1]) for line in open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/splits/split_6/test.tsv")]
split_7 = [int(line.split("\t")[1]) for line in open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/splits/split_7/test.tsv")]
split_8 = [int(line.split("\t")[1]) for line in open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/splits/split_8/test.tsv")]
split_9 = [int(line.split("\t")[1]) for line in open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/splits/split_9/test.tsv")]
split_orig = [int(line.split("\t")[1]) for line in open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/tokenized/mixed_test.tsv")]

split_0_point = 955
split_1_point = 1690
split_2_point = 913
split_3_point = 900
split_4_point = 879

split_5_point = 568
split_6_point = 417
split_7_point = 609
split_8_point = 363
split_9_point = 830
split_orig_point = 531

outputs_path = "/Users/alexwarstadt/Workspace/acceptability_playground/outputs/splits/"

split_0_in_mccs = []
split_0_out_mccs = []
split_0_overall_mccs = []
split_0_in_accs = []
split_0_out_accs = []
split_0_overall_accs = []
for x in os.listdir(outputs_path + "split_0/outputs/"):
    outs = [0 if float(o) < 0.5 else 1 for o in open(outputs_path + "split_0/outputs/" + x)]
    split_0_in_mccs.append(pearsonr(outs[:split_0_point], split_0[:split_0_point])[0])
    split_0_out_mccs.append(pearsonr(outs[split_0_point:], split_0[split_0_point:])[0])
    split_0_overall_mccs.append(pearsonr(outs, split_0)[0])

    split_0_in_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[:split_0_point], split_0[:split_0_point])]))
    split_0_out_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[split_0_point:], split_0[split_0_point:])]))
    split_0_overall_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs, split_0)]))


print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (np.average(split_0_in_mccs),
                                                          np.std(split_0_in_mccs),
                                                          np.average(split_0_in_accs),
                                                          np.std(split_0_in_accs),
                                                          np.average(split_0_out_mccs),
                                                          np.std(split_0_out_mccs),
                                                          np.average(split_0_out_accs),
                                                          np.std(split_0_out_accs),
                                                          np.average(split_0_overall_mccs),
                                                          np.std(split_0_overall_mccs),
                                                          np.average(split_0_overall_accs),
                                                          np.std(split_0_overall_accs),
                                                          ))





split_1_in_mccs = []
split_1_out_mccs = []
split_1_overall_mccs = []
split_1_in_accs = []
split_1_out_accs = []
split_1_overall_accs = []
for x in os.listdir(outputs_path + "split_1/outputs/"):
    outs = [0 if float(o) < 0.5 else 1 for o in open(outputs_path + "split_1/outputs/" + x)]
    split_1_in_mccs.append(pearsonr(outs[:split_1_point], split_1[:split_1_point])[0])
    split_1_out_mccs.append(pearsonr(outs[split_1_point:], split_1[split_1_point:])[0])
    split_1_overall_mccs.append(pearsonr(outs, split_1)[0])

    split_1_in_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[:split_1_point], split_1[:split_1_point])]))
    split_1_out_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[split_1_point:], split_1[split_1_point:])]))
    split_1_overall_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs, split_1)]))


print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (np.average(split_1_in_mccs),
                                                          np.std(split_1_in_mccs),
                                                          np.average(split_1_in_accs),
                                                          np.std(split_1_in_accs),
                                                          np.average(split_1_out_mccs),
                                                          np.std(split_1_out_mccs),
                                                          np.average(split_1_out_accs),
                                                          np.std(split_1_out_accs),
                                                          np.average(split_1_overall_mccs),
                                                          np.std(split_1_overall_mccs),
                                                          np.average(split_1_overall_accs),
                                                          np.std(split_1_overall_accs),
                                                          ))


split_2_in_mccs = []
split_2_out_mccs = []
split_2_overall_mccs = []
split_2_in_accs = []
split_2_out_accs = []
split_2_overall_accs = []
for x in os.listdir(outputs_path + "split_2/outputs/"):
    outs = [0 if float(o) < 0.5 else 1 for o in open(outputs_path + "split_2/outputs/" + x)]
    split_2_in_mccs.append(pearsonr(outs[:split_2_point], split_2[:split_2_point])[0])
    split_2_out_mccs.append(pearsonr(outs[split_2_point:], split_2[split_2_point:])[0])
    split_2_overall_mccs.append(pearsonr(outs, split_2)[0])

    split_2_in_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[:split_2_point], split_2[:split_2_point])]))
    split_2_out_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[split_2_point:], split_2[split_2_point:])]))
    split_2_overall_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs, split_2)]))


print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (np.average(split_2_in_mccs),
                                                          np.std(split_2_in_mccs),
                                                          np.average(split_2_in_accs),
                                                          np.std(split_2_in_accs),
                                                          np.average(split_2_out_mccs),
                                                          np.std(split_2_out_mccs),
                                                          np.average(split_2_out_accs),
                                                          np.std(split_2_out_accs),
                                                          np.average(split_2_overall_mccs),
                                                          np.std(split_2_overall_mccs),
                                                          np.average(split_2_overall_accs),
                                                          np.std(split_2_overall_accs),
                                                          ))



split_3_in_mccs = []
split_3_out_mccs = []
split_3_overall_mccs = []
split_3_in_accs = []
split_3_out_accs = []
split_3_overall_accs = []
for x in os.listdir(outputs_path + "split_3/outputs/"):
    outs = [0 if float(o) < 0.5 else 1 for o in open(outputs_path + "split_3/outputs/" + x)]
    split_3_in_mccs.append(pearsonr(outs[:split_3_point], split_3[:split_3_point])[0])
    split_3_out_mccs.append(pearsonr(outs[split_3_point:], split_3[split_3_point:])[0])
    split_3_overall_mccs.append(pearsonr(outs, split_3)[0])

    split_3_in_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[:split_3_point], split_3[:split_3_point])]))
    split_3_out_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[split_3_point:], split_3[split_3_point:])]))
    split_3_overall_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs, split_3)]))


print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (np.average(split_3_in_mccs),
                                                          np.std(split_3_in_mccs),
                                                          np.average(split_3_in_accs),
                                                          np.std(split_3_in_accs),
                                                          np.average(split_3_out_mccs),
                                                          np.std(split_3_out_mccs),
                                                          np.average(split_3_out_accs),
                                                          np.std(split_3_out_accs),
                                                          np.average(split_3_overall_mccs),
                                                          np.std(split_3_overall_mccs),
                                                          np.average(split_3_overall_accs),
                                                          np.std(split_3_overall_accs),
                                                          ))



split_4_in_mccs = []
split_4_out_mccs = []
split_4_overall_mccs = []
split_4_in_accs = []
split_4_out_accs = []
split_4_overall_accs = []
for x in os.listdir(outputs_path + "split_4/outputs/"):
    outs = [0 if float(o) < 0.5 else 1 for o in open(outputs_path + "split_4/outputs/" + x)]
    split_4_in_mccs.append(pearsonr(outs[:split_4_point], split_4[:split_4_point])[0])
    split_4_out_mccs.append(pearsonr(outs[split_4_point:], split_4[split_4_point:])[0])
    split_4_overall_mccs.append(pearsonr(outs, split_4)[0])

    split_4_in_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[:split_4_point], split_4[:split_4_point])]))
    split_4_out_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[split_4_point:], split_4[split_4_point:])]))
    split_4_overall_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs, split_4)]))


print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (np.average(split_4_in_mccs),
                                                          np.std(split_4_in_mccs),
                                                          np.average(split_4_in_accs),
                                                          np.std(split_4_in_accs),
                                                          np.average(split_4_out_mccs),
                                                          np.std(split_4_out_mccs),
                                                          np.average(split_4_out_accs),
                                                          np.std(split_4_out_accs),
                                                          np.average(split_4_overall_mccs),
                                                          np.std(split_4_overall_mccs),
                                                          np.average(split_4_overall_accs),
                                                          np.std(split_4_overall_accs),
                                                          ))



split_orig_in_mccs = []
split_orig_out_mccs = []
split_orig_overall_mccs = []
split_orig_in_accs = []
split_orig_out_accs = []
split_orig_overall_accs = []
for x in os.listdir(outputs_path + "split_orig/outputs/"):
    outs = [0 if float(o) < 0.5 else 1 for o in open(outputs_path + "split_orig/outputs/" + x)]
    split_orig_in_mccs.append(pearsonr(outs[:split_orig_point], split_orig[:split_orig_point])[0])
    split_orig_out_mccs.append(pearsonr(outs[split_orig_point:], split_orig[split_orig_point:])[0])
    split_orig_overall_mccs.append(pearsonr(outs, split_orig)[0])

    split_orig_in_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[:split_orig_point], split_orig[:split_orig_point])]))
    split_orig_out_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[split_orig_point:], split_orig[split_orig_point:])]))
    split_orig_overall_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs, split_orig)]))


print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (np.average(split_orig_in_mccs),
                                                          np.std(split_orig_in_mccs),
                                                          np.average(split_orig_in_accs),
                                                          np.std(split_orig_in_accs),
                                                          np.average(split_orig_out_mccs),
                                                          np.std(split_orig_out_mccs),
                                                          np.average(split_orig_out_accs),
                                                          np.std(split_orig_out_accs),
                                                          np.average(split_orig_overall_mccs),
                                                          np.std(split_orig_overall_mccs),
                                                          np.average(split_orig_overall_accs),
                                                          np.std(split_orig_overall_accs),
                                                          ))


split_5_in_mccs = []
split_5_out_mccs = []
split_5_overall_mccs = []
split_5_in_accs = []
split_5_out_accs = []
split_5_overall_accs = []
for x in os.listdir(outputs_path + "split_5/"):
    outs = [0 if float(o) < 0.5 else 1 for o in open(outputs_path + "split_5/" + x)]
    split_5_in_mccs.append(pearsonr(outs[:split_5_point], split_5[:split_5_point])[0])
    split_5_out_mccs.append(pearsonr(outs[split_5_point:], split_5[split_5_point:])[0])
    split_5_overall_mccs.append(pearsonr(outs, split_5)[0])

    split_5_in_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[:split_5_point], split_5[:split_5_point])]))
    split_5_out_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[split_5_point:], split_5[split_5_point:])]))
    split_5_overall_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs, split_5)]))


print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (np.average(split_5_in_mccs),
                                                          np.std(split_5_in_mccs),
                                                          np.average(split_5_in_accs),
                                                          np.std(split_5_in_accs),
                                                          np.average(split_5_out_mccs),
                                                          np.std(split_5_out_mccs),
                                                          np.average(split_5_out_accs),
                                                          np.std(split_5_out_accs),
                                                          np.average(split_5_overall_mccs),
                                                          np.std(split_5_overall_mccs),
                                                          np.average(split_5_overall_accs),
                                                          np.std(split_5_overall_accs),
                                                          ))

split_6_in_mccs = []
split_6_out_mccs = []
split_6_overall_mccs = []
split_6_in_accs = []
split_6_out_accs = []
split_6_overall_accs = []
for x in os.listdir(outputs_path + "split_6/"):
    outs = [0 if float(o) < 0.5 else 1 for o in open(outputs_path + "split_6/" + x)]
    split_6_in_mccs.append(pearsonr(outs[:split_6_point], split_6[:split_6_point])[0])
    split_6_out_mccs.append(pearsonr(outs[split_6_point:], split_6[split_6_point:])[0])
    split_6_overall_mccs.append(pearsonr(outs, split_6)[0])

    split_6_in_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[:split_6_point], split_6[:split_6_point])]))
    split_6_out_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[split_6_point:], split_6[split_6_point:])]))
    split_6_overall_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs, split_6)]))


print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (np.average(split_6_in_mccs),
                                                          np.std(split_6_in_mccs),
                                                          np.average(split_6_in_accs),
                                                          np.std(split_6_in_accs),
                                                          np.average(split_6_out_mccs),
                                                          np.std(split_6_out_mccs),
                                                          np.average(split_6_out_accs),
                                                          np.std(split_6_out_accs),
                                                          np.average(split_6_overall_mccs),
                                                          np.std(split_6_overall_mccs),
                                                          np.average(split_6_overall_accs),
                                                          np.std(split_6_overall_accs),
                                                          ))


split_7_in_mccs = []
split_7_out_mccs = []
split_7_overall_mccs = []
split_7_in_accs = []
split_7_out_accs = []
split_7_overall_accs = []
for x in os.listdir(outputs_path + "split_7/"):
    outs = [0 if float(o) < 0.5 else 1 for o in open(outputs_path + "split_7/" + x)]
    split_7_in_mccs.append(pearsonr(outs[:split_7_point], split_7[:split_7_point])[0])
    split_7_out_mccs.append(pearsonr(outs[split_7_point:], split_7[split_7_point:])[0])
    split_7_overall_mccs.append(pearsonr(outs, split_7)[0])

    split_7_in_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[:split_7_point], split_7[:split_7_point])]))
    split_7_out_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[split_7_point:], split_7[split_7_point:])]))
    split_7_overall_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs, split_7)]))


print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (np.average(split_7_in_mccs),
                                                          np.std(split_7_in_mccs),
                                                          np.average(split_7_in_accs),
                                                          np.std(split_7_in_accs),
                                                          np.average(split_7_out_mccs),
                                                          np.std(split_7_out_mccs),
                                                          np.average(split_7_out_accs),
                                                          np.std(split_7_out_accs),
                                                          np.average(split_7_overall_mccs),
                                                          np.std(split_7_overall_mccs),
                                                          np.average(split_7_overall_accs),
                                                          np.std(split_7_overall_accs),
                                                          ))


split_8_in_mccs = []
split_8_out_mccs = []
split_8_overall_mccs = []
split_8_in_accs = []
split_8_out_accs = []
split_8_overall_accs = []
for x in os.listdir(outputs_path + "split_8/"):
    outs = [0 if float(o) < 0.5 else 1 for o in open(outputs_path + "split_8/" + x)]
    split_8_in_mccs.append(pearsonr(outs[:split_8_point], split_8[:split_8_point])[0])
    split_8_out_mccs.append(pearsonr(outs[split_8_point:], split_8[split_8_point:])[0])
    split_8_overall_mccs.append(pearsonr(outs, split_8)[0])

    split_8_in_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[:split_8_point], split_8[:split_8_point])]))
    split_8_out_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[split_8_point:], split_8[split_8_point:])]))
    split_8_overall_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs, split_8)]))


print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (np.average(split_8_in_mccs),
                                                          np.std(split_8_in_mccs),
                                                          np.average(split_8_in_accs),
                                                          np.std(split_8_in_accs),
                                                          np.average(split_8_out_mccs),
                                                          np.std(split_8_out_mccs),
                                                          np.average(split_8_out_accs),
                                                          np.std(split_8_out_accs),
                                                          np.average(split_8_overall_mccs),
                                                          np.std(split_8_overall_mccs),
                                                          np.average(split_8_overall_accs),
                                                          np.std(split_8_overall_accs),
                                                          ))


split_9_in_mccs = []
split_9_out_mccs = []
split_9_overall_mccs = []
split_9_in_accs = []
split_9_out_accs = []
split_9_overall_accs = []
for x in os.listdir(outputs_path + "split_9/"):
    outs = [0 if float(o) < 0.5 else 1 for o in open(outputs_path + "split_9/" + x)]
    split_9_in_mccs.append(pearsonr(outs[:split_9_point], split_9[:split_9_point])[0])
    split_9_out_mccs.append(pearsonr(outs[split_9_point:], split_9[split_9_point:])[0])
    split_9_overall_mccs.append(pearsonr(outs, split_9)[0])

    split_9_in_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[:split_9_point], split_9[:split_9_point])]))
    split_9_out_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs[split_9_point:], split_9[split_9_point:])]))
    split_9_overall_accs.append(np.average([1 if x == y else 0 for x, y in zip(outs, split_9)]))


print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (np.average(split_9_in_mccs),
                                                          np.std(split_9_in_mccs),
                                                          np.average(split_9_in_accs),
                                                          np.std(split_9_in_accs),
                                                          np.average(split_9_out_mccs),
                                                          np.std(split_9_out_mccs),
                                                          np.average(split_9_out_accs),
                                                          np.std(split_9_out_accs),
                                                          np.average(split_9_overall_mccs),
                                                          np.std(split_9_overall_mccs),
                                                          np.average(split_9_overall_accs),
                                                          np.std(split_9_overall_accs),
                                                          ))