import numpy as np


playground = "/Users/alexwarstadt/Workspace/acceptability_playground/"
all_cola = "CoLA_1.1/all.tsv"
in_domain_test = "CoLA_1.1/in_domain_test.tsv"
in_domain_train = "CoLA_1.1/in_domain_train.tsv"
in_domain_valid = "CoLA_1.1/in_domain_valid.tsv"
out_of_domain_test = "CoLA_1.1/out_of_domain_test.tsv"
out_of_domain_valid = "CoLA_1.1/out_of_domain_valid.tsv"




def read_cola_table(file):
    cola_table = []
    for i, line in enumerate(open(file)):
        line = line.split("\t")
        cola_table.append({"source": line[0],
                           "label": line[1],
                           "accept": line[2],
                           "sentence": line[3].strip(),
                           "id": i
                           })
    return cola_table



### MAIN ###

all_cola_table = read_cola_table(all_cola)

in_domain_test_table = read_cola_table(in_domain_test)


pass