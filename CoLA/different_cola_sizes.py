import numpy as np

train = "CoLA_1.1/in_domain_train.tsv"
lines = np.array([line for line in open(train)])
lines_30 = np.random.choice(lines, 30, False)
lines_100 = np.random.choice(lines, 100, False)
lines_300 = np.random.choice(lines, 300, False)
lines_1000 = np.random.choice(lines, 1000, False)
lines_3000 = np.random.choice(lines, 3000, False)


out_30 = open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/sizes/CoLA_30", "w")
for l in lines_30:
    out_30.write(l)
out_30.close()

out_100 = open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/sizes/CoLA_100", "w")
for l in lines_100:
    out_100.write(l)
out_100.close()

out_300 = open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/sizes/CoLA_300", "w")
for l in lines_300:
    out_300.write(l)
out_300.close()

out_1000 = open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/sizes/CoLA_1000", "w")
for l in lines_1000:
    out_1000.write(l)
out_1000.close()

out_3000 = open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/sizes/CoLA_3000", "w")
for l in lines_3000:
    out_3000.write(l)
out_3000.close()



pass

