# Script to combine dev set data with human judgments

humans = [line.strip().split("\t") for line in open("human_annotation.tsv")]
in_domain_dev = [line.strip().split("\t") for line in open("CoLA_1.1/in_domain_valid.tsv")]
out_of_domain_dev = [line.strip().split("\t") for line in open("CoLA_1.1/out_of_domain_valid.tsv")]

final_output = open("human_annotation_release.tsv", "w")
final_output.write("domain\tsource\tCoLA_label\tsource_annotation\tannotator1\tannotator2\tannotator3\tannotator4\tannotator5\tsentence\n")
for h_line in humans:
    in_domain_line = list(filter(lambda x: x[3] == h_line[0], in_domain_dev))
    out_of_domain_line = list(filter(lambda x: x[3] == h_line[0], out_of_domain_dev))
    if len(in_domain_line) == 1:
        dev_line = in_domain_line[0]
        domain = "in"
    elif len(out_of_domain_line) == 1:
        dev_line = out_of_domain_line[0]
        domain = "out"
    final_vals = [domain] + dev_line[:3] + h_line[1:] + [h_line[0]]
    final_output.write("\t".join(final_vals) + "\n")

final_output.close()
