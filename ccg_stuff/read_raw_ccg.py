import re
from nltk.tree import Tree

file = open("data/raw_ccg.txt")

table = {}

for line in file:
    if line.startswith("ID"):
        id = re.match("ID=(\\d+)", line).group(1)
        id = int(id)
    elif line.startswith("("):
        if id in table.keys():
            table[id].append(line)
        else:
            table[id] = [line]



class CCGBankTree:
    def __init__(self, my_tree, nltk_tree, orig_string):
        self.my_tree = my_tree
        self.nltk_tree = nltk_tree
        self.orig_string = orig_string
        self.sentence = " ".join(nltk_tree.leaves())
        self.pos = nltk_tree.pos()

class MyTree:
    def __init__(self, ccg_cat):
        self.ccg_cat = ccg_cat

    def basic_str(self):
        pass

class Node(MyTree):
    def __init__(self, ccg_cat, head, n_daughters, daughters):
        super(Node, self).__init__(ccg_cat)
        self.head = head
        self.n_daughters = n_daughters
        self.daughters = daughters

    def basic_str(self):
        return "{%s %s}" % (self.ccg_cat, " ".join([d.basic_str() for d in self.daughters]))

class Leaf(MyTree):
    def __init__(self, ccg_cat, mod_pos_tag, orig_pos_tag, word, pred_arg_cat):
        super(Leaf, self).__init__(ccg_cat)
        self.mod_pos_tag = mod_pos_tag
        self.orig_pos_tag = orig_pos_tag
        self.word = word
        self.pred_arg_cat = pred_arg_cat

    def basic_str(self):
        return "{%s %s}" % (self.ccg_cat, self.word)


def parse(tree_string):
    nltk_tree = Tree.fromstring(tree_string, node_pattern="<.*?>", leaf_pattern="<.*?>")
    my_tree = build_my_tree(nltk_tree)
    ccg_bank_tree = CCGBankTree(my_tree, nltk_tree, tree_string)
    return ccg_bank_tree

def build_my_tree(nltk_tree):
    vals = nltk_tree.label()[1:-1].split(" ")
    if vals[0] == "L":
        try:
            my_tree = Leaf(*vals[1:])
        except TypeError:
            print(nltk_tree)
            print(vals)
    elif vals[0] == "T":
        daughters = [build_my_tree(t) for t in nltk_tree if isinstance(t, Tree)]
        my_tree = Node(*vals[1:], daughters)
    else:
        raise Exception("neither a node nor a terminal")
    return my_tree


tree_string = "(<T S[dcl] 1 2> (<L NP PRP PRP We NP>) (<T S[dcl]\\NP 0 2> (<L (S[dcl]\\NP)/S[em] VBP VBP think (S[dcl]\\NP)/S[em]>) (<T S[em] 0 2> (<L S[em]/S[dcl] IN IN that S[em]/S[dcl]>) (<T S[dcl] 1 2> (<T NP 0 1> (<L N NNP NNP Leslie N>) ) (<T S[dcl]\\NP 0 2> (<L (S[dcl]\\NP)/PP VBZ VBZ likes (S[dcl]\\NP)/PP>) (<L PP NN NN us. PP>) ) ) ) ) )"

# t = Tree.fromstring(tree_string, node_pattern="<T(.*?)>", leaf_pattern="<L(.*?)>")
# t.pretty_print()

# my_tree = parse(tree_string)
#
# print(my_tree.basic_str())

def write_table(table, out_path):
    out_file = open(out_path, "w")
    for k in table.keys():
        for v in table[k]:
            out_file.write("ID=%d\n" % k)
            out_file.write(v + "\n")
    out_file.close()

def parse_table(table):
    parsed_table = {}
    for k in table.keys():
        parsed_table[k] = [parse(tree_string) for tree_string in table[k]]
    return parsed_table


def simplify_table(table):
    simple_table = {}
    for k in table.keys():
        simple_table[k] = [tree.basic_str() for tree in table[k]]
    return simple_table

parsed_table = parse_table(table)
simplified_table = simplify_table(parsed_table)
# write_table(simplified_table, "data/simple_ccg.txt")
pass