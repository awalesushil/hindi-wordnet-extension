
import csv
import json

def write_to_file(data, filename):
    path = "data/conve/" # or non-filtered
    with open(path+filename,"w+", encoding='utf-8') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(data)


def read_file(filename):
    with open("data/filtered/"+filename,"r", encoding='utf-8') as f:
        return f.read().split("\n")


train = read_file("train.csv")
valid = read_file("valid.csv")
test = read_file("test.csv")

wordnet = json.load(open("wordnet.json","r"))

def data_prep(data):
    _data = []
    for each in data:
        if len(each.split(",")) > 1:
            e1, r, e2 = each.split(",")
            try:
                _data.append((wordnet[e1]['head_word'], r, wordnet[e2]['head_word']))
            except KeyError:
                print("Synset " + e2 + " not found. Skipping...")
    return _data

write_to_file(data_prep(train), "train.tsv")
write_to_file(data_prep(test), "test.tsv")
write_to_file(data_prep(valid), "valid.tsv")