import numpy as np
import os
from os.path import isfile, join
from scipy.stats.stats import pearsonr
from math import isnan
from scipy import stats
# import matplotlib.pyplot as plt
import pylab as plt
import random
from matplotlib.pyplot import figure
from nltk import tokenize



# def process_outputs(outputs, metadata, floats=False):


def read_table():
    table_file = open("/Users/alexwarstadt/Workspace/acceptability_playground/annotation_stats/annotated_tables/annotated_simple.tsv")
    header = table_file.readline().split("\t")[4:]
    header = [x.strip() for x in header]
    table = []
    metadata = []
    for line in table_file:
        vals = line.split("\t")
        md = vals[0:4]
        md[2] = int(md[2])
        md_dict = {}
        for m, k in zip(md, ["source", "domain", "label", "sentence"]):
            md_dict[k] = m
        data = [1 if v.strip() is "1" else 0 for v in vals[4:]]
        if len(data) != 63:
            print(vals[3])
        metadata.append(md_dict)
        table.append(data)
    return np.array(table).transpose(), header, metadata


def read_outputs(out_files, floats=False, two_col=False):
    outputs_table = []
    for f in out_files:
        if two_col:
            outputs = [x.split("\t")[1] for x in open(f)]
        else:
            outputs = [x for x in open(f)]
        if floats:
            outputs = [float(x) for x in outputs]
        else:
            outputs = [1 if float(x) > 0.5 else 0 for x in outputs]
        outputs_table.append(outputs)
    average_guess = [1 if x > 0.5 else 0 for x in np.average(outputs_table, 0)]
    outputs_table.insert(0, average_guess)
    return np.array(outputs_table)


def read_outputs_stilts(out_files, floats=False, two_col=False):
    outputs_table = []
    for f in out_files:
        f = open(f)
        f.readline()
        if floats:
            outputs = [float(x.split("\t")[1]) for x in f]
        else:
            outputs = [float(x.split("\t")[1]) for x in f]
            outputs = [1 if o > 0.5 else 0 for o in outputs]
        outputs_table.append(outputs)
    average_guess = [1 if x > 0.5 else 0 for x in np.average(outputs_table, 0)]
    outputs_table.insert(0, average_guess)
    return np.array(outputs_table)


def process_outputs(outputs_table, annotations_table, metadata, floats=False):
    # TODO: implement float option
    gold_labels = [md["label"] for md in metadata]
    overall_performance = [pearsonr(gold_labels, outs) for outs in outputs_table]
    processed_table = []
    for out_col in outputs_table:
        processed_col = []
        for annotation_col in annotations_table:
            filtered_out = filter_column(out_col, annotation_col)
            filtered_gold = filter_column(gold_labels, annotation_col)
            mcc = pearsonr(filtered_out, filtered_gold)
            if np.isnan(mcc[0]):
                mcc = [0, 1]
            processed_col.append(mcc)
        processed_table.append(processed_col)
    return np.array(processed_table).transpose(), overall_performance     # [n_classifiers x n_categories]




def filter_column(out_col, annotation_col):
    filtered = []
    for o, a in zip(out_col, annotation_col):
        if a == 1:
            filtered.append(o)
    return np.array(filtered)



open_ai_fancy = "/Users/alexwarstadt/Workspace/acceptability_playground/annotation_stats/annotated_tables/open_ai_fancy.tsv"
open_ai_fancy = "/Users/alexwarstadt/Workspace/acceptability_playground/annotation_stats/annotated_tables/open_ai_fancy.tsv"
open_ai_fancy = "/Users/alexwarstadt/Workspace/acceptability_playground/annotation_stats/annotated_tables/open_ai_fancy.tsv"
open_ai_fancy_cat = "/Users/alexwarstadt/Workspace/acceptability_playground/annotation_stats/annotated_tables/open_ai_fancy.tsv"
BERT_fancy_cat = "/Users/alexwarstadt/Workspace/acceptability_playground/annotation_stats/annotated_tables/BERT_fancy.tsv"
elmo_fancy_cat = "/Users/alexwarstadt/Workspace/acceptability_playground/annotation_stats/annotated_tables/elmo_fancy.tsv"

def openAI_table():
    openAI_path = "/Users/alexwarstadt/Workspace/acceptability_playground/annotation_stats/open_ai"
    openAI_files = [join(openAI_path, f, "cola_val.tsv") for f in os.listdir(openAI_path) if not f.startswith(".")]
    return read_outputs_stilts(openAI_files, False)

def BERT_table():
    BERT_path = "/Users/alexwarstadt/Workspace/acceptability_playground/annotation_stats/BERT_success"
    BERT_files = [join(BERT_path, f, "cola", "val_results.tsv") for f in os.listdir(BERT_path) if not f.startswith(".")]
    return read_outputs(BERT_files, False, True)

def elmo_table():
    elmo_path = "/Users/alexwarstadt/Workspace/acceptability_playground/annotation_stats/aj_elmo_pooling_old_11-21-18/outputs/"
    elmo_files = [join(elmo_path, f) for f in os.listdir(elmo_path)]
    return read_outputs(elmo_files, False)

def make_fancy_table(model_outputs_table, annotations_table, metadata, header, out_path):
    processed_table, overall_performance = process_outputs(model_outputs_table, annotations_table, metadata)
    processed_table = np.array(processed_table).transpose()
    fancy_table = open(out_path, "w")
    topline = "\t".join(["%f\t%f" % mcc_p for mcc_p in overall_performance])
    fancy_table.write("\tmean\tsd\t%s\n" % topline)
    avgs = []
    for mccs, ps, h in zip(processed_table[0], processed_table[1], header):
        avg = np.average(mccs)
        avgs.append(avg)
        std = np.std(mccs)
        fancy_table.write("%s\t%f\t%f\t" % (h, avg, std))
        fancy_table.write("\t".join(["%f\t%f" % (mcc, p) for mcc, p in zip(mccs, ps)]) + "\n")
    fancy_table.close()


# def get_avgs_errors(processed_table):
#     avgs = []
#     errors = []
#     for mccs, ps in zip(processed_table[0], processed_table[1]):
#         avg = np.average(mccs)
#         avgs.append(avg)
#         std = np.std(mccs)
#         # interval = stats.norm.interval(0.95, loc=avg, scale=std / (len(mccs) ** 0.5))
#         # errors.append([avg - interval[0], interval[1] - avg])
#         errors.append(std)
#     return avgs, errors

def get_avgs_errors(processed_table):
    avgs = []
    errors = []
    for mccs, ps in zip(processed_table[0], processed_table[1]):
        avg = max(np.average(mccs), 0)
        avgs.append(avg)
        std = np.std(mccs)
        mini = max(0, avg-std)
        errors.append((avg - mini, std))
        # interval = stats.norm.interval(0.95, loc=avg, scale=std / (len(mccs) ** 0.5))
        # mini = max(0, min(mccs))
        # maxi = max(0, max(mccs))
        # errors.append((avg - mini, maxi - avg))
    return avgs, errors


# def make_plot(avgs, errors):
#     fig, ax = plt.subplots()
#     ind = np.arange(len(avgs))
#     width = 0.9
#     plt.bar(ind, avgs, width=width, yerr=np.array(errors).transpose())
#     ax.set_xticks(ind + width / 2 - 0.5)
#     ax.set_xticklabels(header, rotation="vertical", va='top', ha='center')
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(6)
#     plt.tight_layout()
#     plt.show()


def make_triple_plot(avgs, errors, labels):
    fig, ax = plt.subplots()
    ind = np.arange(len(avgs[0]))
    width = 0.25

    # ax.set_xlabel("Sentence Length")

    # figure(num=None, figsize=(8, 2), dpi=80, facecolor='w', edgecolor='k')

    x = np.linspace(0, len(avgs[0]))
    l1 = ax.plot(x, [0.320 for x in range(50)], color='#ffcc00', linestyle="--")
    l2 = ax.plot(x, [0.528 for x in range(50)], color='m', linestyle="--")
    l3 = ax.plot(x, [0.582 for x in range(50)], color='c', linestyle="--")
    l4 = ax.plot(x, [0.697 for x in range(50)], color='k', linestyle="--")

    p1 = ax.bar(ind, np.array(avgs[0]), width=width, yerr=np.array(errors[0]).transpose(), color='#ffcc00')
    p2 = ax.bar(ind + width, np.array(avgs[1]), width=width, yerr=np.array(errors[1]).transpose(), color='m')
    p3 = ax.bar(ind + 2*width, np.array(avgs[2]), width=width, yerr=np.array(errors[2]).transpose(), color='c')

    ax.legend((p1[0], p2[0], p3[0], l1[0], l2[0], l3[0], l4[0]), ('CoLA Baseline', 'OpenAI', "BERT", 'CoLA mean', 'OpenAI mean', 'BERT mean', 'Human perf.'), loc=1)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(labels, rotation="30", va='top', ha='right', rotation_mode='anchor')
    ax.set_ylim([0, 1])
    plt.ylabel('MCC', fontsize=16)
    # plt.xlabel('', fontsize=16)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    plt.tight_layout()
    plt.show()


def make_single_plot(avgs, errors, labels):
    indices = [0, 2, 8, 10, 4, 6, 7, 14]
    avgs = [avgs[i] for i in indices]
    errors = [errors[i] for i in indices]
    labels = [labels[i] for i in indices]
    fig, ax = plt.subplots()
    ind = np.arange(len(avgs))
    width = 0.75

    # ax.set_xlabel("Sentence Length")

    # figure(num=None, figsize=(8, 2), dpi=80, facecolor='w', edgecolor='k')

    x = np.linspace(0, len(avgs))
    l1 = ax.plot(x, [0.320 for x in range(50)], color='c', linestyle="--")
    # l2 = ax.plot(x, [0.528 for x in range(50)], color='m', linestyle="--")
    # l3 = ax.plot(x, [0.582 for x in range(50)], color='c', linestyle="--")
    l4 = ax.plot(x, [0.697 for x in range(50)], color='k', linestyle="--")

    p1 = ax.bar(ind, np.array(avgs), width=width, yerr=np.array(errors).transpose(), color='c')
    # p2 = ax.bar(ind + width, np.array(avgs[1]), width=width, yerr=np.array(errors[1]).transpose(), color='m')
    # p3 = ax.bar(ind + 2*width, np.array(avgs[2]), width=width, yerr=np.array(errors[2]).transpose(), color='c')

    ax.legend((p1[0], l1[0], l4[0]), ('ELMo-Style Pooling Classifier: phenomenon-specific', 'ELMo-Style Pooling Classifier: overall dev', 'Human performance: 20% of dev'), loc=1)
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation="30", va='top', ha='right', rotation_mode='anchor')
    ax.set_ylim([0, 1])
    plt.ylabel('MCC', fontsize=16)
    # plt.xlabel('', fontsize=16)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    plt.tight_layout()
    plt.show()

def make_single_plot_with_freq(avgs, errors, labels, freqs):
    indices = [0, 2, 8, 10, 4, 6, 7, 14]
    avgs = [avgs[i] for i in indices]
    errors = [errors[i] for i in indices]
    labels = [labels[i] for i in indices]
    freqs = [freqs[i] for i in indices]
    fig, ax = plt.subplots()
    ind = np.arange(len(avgs))
    # width = 0.375

    ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.

    width = 0.45


    ax.set_ylabel('MCC', fontsize=16)
    ax2.set_ylabel('Percent of sentences', fontsize=16)

    x = np.linspace(0, len(avgs))
    l1 = ax.plot(x, [0.320 for x in range(50)], color='c', linestyle="--")
    l4 = ax.plot(x, [0.697 for x in range(50)], color='k', linestyle="--")


    p1 = ax.bar(ind, np.array(avgs), width=width, yerr=np.array(errors).transpose(), color='c')
    p2 = ax2.bar(ind + width, np.array(freqs), color='m', width=width)

    ax.legend((p1[0], p2[0], l1[0], l4[0]), ('ELMo-Style Real/Fake Encoder phenomenon-specific', 'Frequency in CoLA', 'ELMo-Style Real/Fake Encoder overall', 'Human perf.'), loc=1)
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation="30", va='top', ha='right', rotation_mode='anchor')
    ax.set_ylim([0, 1])

    # ax2.set_xticks(ind)
    # ax2.set_xticklabels(labels, rotation="30", va='top', ha='right', rotation_mode='anchor')
    ax2.set_ylim([0, 100])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    ax2.tick_params('y', labelsize=14)
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)

    # for tick in ax2.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    plt.tight_layout()
    plt.show()


def make_triple_plot_line(avgs, errors, labels):
    fig, ax = plt.subplots()
    ind = np.arange(len(avgs[0]))
    width = 0.25

    # ax.set_xlabel("Sentence Length")

    # figure(num=None, figsize=(8, 2), dpi=80, facecolor='w', edgecolor='k')

    x = np.linspace(0, len(avgs[0]))
    l1 = ax.plot(x, [0.320 for x in range(50)], color='#ffcc00', linestyle="--")
    l2 = ax.plot(x, [0.528 for x in range(50)], color='m', linestyle="--")
    l3 = ax.plot(x, [0.582 for x in range(50)], color='c', linestyle="--")
    l4 = ax.plot(x, [0.697 for x in range(50)], color='k', linestyle="--")

    p1 = ax.errorbar(ind, np.array(avgs[0]), yerr=np.array(errors[0]).transpose(), color='#ffcc00', ecolor='#ffcc00', capsize=2)
    p2 = ax.errorbar(ind, np.array(avgs[1]), yerr=np.array(errors[1]).transpose(), color='m', ecolor='m', capsize=2)
    p3 = ax.errorbar(ind, np.array(avgs[2]), yerr=np.array(errors[2]).transpose(), color='c', ecolor='c', capsize=2)

    # ax.legend((p1[0], p2[0], p3[0], l1[0], l2[0], l3[0], l4[0]), ('CoLA Baseline', 'OpenAI', "BERT", 'CoLA mean', 'OpenAI mean', 'BERT mean', 'Human perf.'), loc=1)
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation="30", va='top', ha='right', rotation_mode='anchor')
    ax.set_ylim([0, 1])
    plt.ylabel('MCC', fontsize=16)
    plt.xlabel('Sentence Length', fontsize=16)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    plt.tight_layout()
    plt.show()


# def split_plot(avgs, errors, labels, split):
#     fig, ax = plt.subplots()
#     ind = np.arange(len(avgs[0]))
#     width = 0.25
#
#     x = np.linspace(0, len(avgs[0]))
#     l1 = ax.plot(x, [0.320 for x in range(50)], color='#ffcc00', linestyle="--")
#     l2 = ax.plot(x, [0.528 for x in range(50)], color='m', linestyle="--")
#     l3 = ax.plot(x, [0.582 for x in range(50)], color='c', linestyle="--")
#
#     p1 = ax.bar(ind, np.array(avgs[0]), width=width, yerr=np.array(errors[0]).transpose(), color='#ffcc00')
#     p2 = ax.bar(ind + width, np.array(avgs[1]), width=width, yerr=np.array(errors[1]).transpose(), color='m')
#     p3 = ax.bar(ind + 2*width, np.array(avgs[2]), width=width, yerr=np.array(errors[2]).transpose(), color='c')
#
#     ax.legend((p1[0], p2[0], p3[0], l1[0], l2[0], l3[0]), ('CoLA Baseline', 'OpenAI', "BERT", 'CoLA mean', 'OpenAI mean', 'BERT mean'), loc=1)
#     ax.set_xticks(ind + width)
#     # ax.set_xticklabels(header, rotation="vertical", va='top', ha='center')
#     ax.set_xticklabels(labels, rotation="30", va='top', ha='right', rotation_mode='anchor')
#
#
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(6)
#     # plt.tight_layout()
#     # plt.show()


def get_cat_table(annotations_table, cats_vec):
    prev = -1
    cat_table = []
    for cat, annotation_row in zip(cats_vec,annotations_table):
        if prev != cat:
            cat_table.append(annotation_row)
            prev = cat
        else:
            cat_table[-1] = [x or y for x, y in zip(cat_table[-1], annotation_row)]
    return np.array(cat_table)


def pairwise_correlations(table, labels):
    corrs = {}
    for i in range(len(table)):
        for j in range(i+1, len(table)):
            corrs[(labels[i], labels[j])] = pearsonr(table[i], table[j])
    for key, value in sorted(corrs.items(), key=lambda x: x[1][0], reverse=True):
        print("%s\t%s\t%f\t%f" % (key[0], key[1], value[0], value[1]))


major_header = ["Simple", "Predicate", "Adjunct", "Argument Type", "Arg Altern", "Imperative", "Binding", "Question",
                "Comp Clause", "Auxiliary", "to-VP", "N, Adj", "S-Syntax", "Determiner", "Violations"]

major_freqs = [8.3, 24.5, 21.7, 41.0, 40.4, 1.2, 11.6, 21.3, 18.2, 32.6, 16.3, 26.7, 27.4, 17.1, 13.9]

cats_vec = [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9,
            9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14]

def map_bins(l, bins):
    row = [0 for x in range(len(bins))]
    b = min(filter(lambda x: x > l, bins))
    row[bins.index(b)] = 1
    return row


def plot_one(annotations_table, header, metadata):
    openAI_processed, openAI_overall = process_outputs(openAI_table(), annotations_table, metadata)
    BERT_processed, BERT_overall = process_outputs(BERT_table(), annotations_table, metadata)
    elmo_processed, elmo_overall = process_outputs(elmo_table(), annotations_table, metadata)

    openAI_avgs, openAI_errors = get_avgs_errors(openAI_processed)
    BERT_avgs, BERT_errors = get_avgs_errors(BERT_processed)
    elmo_avgs, elmo_errors = get_avgs_errors(elmo_processed)

    make_triple_plot([elmo_avgs, openAI_avgs, BERT_avgs], [elmo_errors, openAI_errors, BERT_errors], header)

def plot_several(annotations_table, header, metadata, split):

    openAI_processed, openAI_overall = process_outputs(openAI_table(), annotations_table, metadata)
    BERT_processed, BERT_overall = process_outputs(BERT_table(), annotations_table, metadata)
    elmo_processed, elmo_overall = process_outputs(elmo_table(), annotations_table, metadata)

    openAI_avgs, openAI_errors = get_avgs_errors(openAI_processed)
    BERT_avgs, BERT_errors = get_avgs_errors(BERT_processed)
    elmo_avgs, elmo_errors = get_avgs_errors(elmo_processed)

    make_triple_plot([x[:split] for x in [elmo_avgs, openAI_avgs, BERT_avgs]],
                     [x[:split] for x in [elmo_errors, openAI_errors, BERT_errors]],
                     header[:split])
    make_triple_plot([x[split:] for x in [elmo_avgs, openAI_avgs, BERT_avgs]],
                     [x[split:] for x in [elmo_errors, openAI_errors, BERT_errors]],
                     header[split:])

    make_triple_plot([elmo_avgs, openAI_avgs, BERT_avgs], [elmo_errors, openAI_errors, BERT_errors], header)


########## MAIN ###########
annotations_table, header, metadata = read_table()
cat_table = get_cat_table(annotations_table, cats_vec)

annotations_table = cat_table
header = major_header


# # Sentence length
# length_bins = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 40]
# length_bins_names = ['1-3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14-16', '17-19', '20-39']
# lengths = np.array([len(tokenize.word_tokenize(m['sentence'])) for m in metadata])
# length_rows = np.array([map_bins(x, length_bins) for x in lengths]).transpose()
# annotations_table = length_rows
# header = length_bins_names


# # Number of Features
# n_features_bins = [2, 3, 4, 5, 6, 7, 8, 9, 10, 18]
# n_features_bins_names = ['0-1', '2', '3', '4', '5', '6', '7', '8', '9', '10-18']
# n_features = np.sum(annotations_table, 0)
# n_features_rows = np.array([map_bins(x, n_features_bins) for x in n_features]).transpose()
# annotations_table = n_features_rows
# header = n_features_bins_names


# Violations
# counts = np.sum(annotations_table[-3:], 1)
# violation_table = np.copy(annotations_table[-3:])

# subtable = np.copy(annotations_table[-3:])
# for i in range(1043):
#     if metadata[i]['label'] == 1:
#         for j in range(3):
#             if random.uniform(0,1) < counts[j] / x:
#                 subtable[j][i] = 1
# violation_table = np.append(violation_table, subtable, axis=0)

# violation_table = np.transpose(np.array(violation_table), [1, 0, 2])
# annotations_table = violation_table
# header = [10000, 10000, 10000,
#           10000, 10000, 10000,
#           10000, 10000, 10000,
#           10000, 10000, 10000,
#           10000, 10000, 10000,
#           ]


openAI_processed, openAI_overall = process_outputs(openAI_table(), annotations_table, metadata)
BERT_processed, BERT_overall = process_outputs(BERT_table(), annotations_table, metadata)
elmo_processed, elmo_overall = process_outputs(elmo_table(), annotations_table, metadata)

openAI_avgs, openAI_errors = get_avgs_errors(openAI_processed)
BERT_avgs, BERT_errors = get_avgs_errors(BERT_processed)
elmo_avgs, elmo_errors = get_avgs_errors(elmo_processed)

# make_triple_plot([elmo_avgs, openAI_avgs, BERT_avgs], [elmo_errors, openAI_errors, BERT_errors], header)
# make_triple_plot_line([elmo_avgs, openAI_avgs, BERT_avgs], [elmo_errors, openAI_errors, BERT_errors], header)

# SMALL PLOT FOR NNAJ paper
# make_single_plot_with_freq(elmo_avgs, elmo_errors, header, major_freqs)
make_single_plot(elmo_avgs, elmo_errors, header)


#
# fig, ax = make_triple_plot([x[:33] for x in [elmo_avgs, openAI_avgs, BERT_avgs]],
#                  [x[:33] for x in [elmo_errors, openAI_errors, BERT_errors]],
#                  header[:33])



# fig2, ax2 = make_triple_plot([x[33:] for x in [elmo_avgs, openAI_avgs, BERT_avgs]],
#                  [x[33:] for x in [elmo_errors, openAI_errors, BERT_errors]],
#                  header[33:])

# fig.add_axes(ax2)

# fig.show()

# outputs_table = read_outputs(outputs_files, False, True)

# make_fancy_table(openAI_table(), cat_table, metadata, major_header, open_ai_fancy_cat)
# make_fancy_table(BERT_table(), cat_table, metadata, major_header, BERT_fancy_cat)
# make_fancy_table(elmo_table(), cat_table, metadata, major_header, elmo_fancy_cat)

