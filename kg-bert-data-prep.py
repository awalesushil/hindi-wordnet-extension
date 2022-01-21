'''
    Convert WordNet to RDF triples in the form (subject, predicate, object) 
        subject & object - entities
        predicate - relation between entities
    Remove near-duplicate and inverse relations
'''

import json
import csv


from sklearn.model_selection import train_test_split


wordnet = json.load(open("wordnet.json","r"))
word2idx = json.load(open("indices/word2idx.json","r"))
relation_map = relation_map = json.load(open("indices/relations.json","r"))


def write_to_file(data, filename):
    path = "data/kg-bert/" 
    with open(path+filename,"w+", encoding='utf-8') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(data)


def read_file(filename):
    with open(filename,"r", encoding='utf-8') as f:
        return f.read().split("\n")

train = read_file("data/filtered/train.csv")
valid = read_file("data/filtered/valid.csv")
test = read_file("data/filtered/test.csv")

def convert(data):
    _data = []
    for each in data:
        if len(each.split(",")) > 1:
            e1, r, e2 = each.split(",")
            _data.append((e1, r, e2))
    return _data

write_to_file(convert(train), "train.tsv")
write_to_file(convert(valid), "valid.tsv")
write_to_file(convert(test), "test.tsv")

train = read_file("data/kg-bert/train.tsv")
valid = read_file("data/kg-bert/valid.tsv")
test = read_file("data/kg-bert/test.tsv")

# Removing entries whose entity pairs are directly linked to train
train_e_pairs = set()

for each in train:
    if len(each.split("\t")) > 1:
        e1, r, e2 = each.split("\t")
        train_e_pairs.add((e1,e2))

_valid = []
for each in valid:
    if len(each.split("\t")) > 1:
        e1, r, e2 = each.split("\t")
        if (e1, e2) and (e2, e1) not in train_e_pairs:
            _valid.append((e1,r,e2))
write_to_file(_valid, "valid.tsv")

_test = []
for each in test:
    if len(each.split("\t")) > 1:
        e1, r, e2 = each.split("\t")
        if (e1, e2) and (e2, e1) not in train_e_pairs:
            _test.append((e1,r,e2))
write_to_file(_test, "test.tsv")


train = read_file("data/kg-bert/train.tsv")
valid = read_file("data/kg-bert/valid.tsv")
test = read_file("data/kg-bert/test.tsv")

clean = lambda x: x.replace("\n","")

def extract_entity(data):
    entity = []
    for d in data:
        d = d.split("\t")
        if len(d) > 1:
            entity.append(d[0])
            entity.append(d[2])
    return entity

def extract_relation(data):
    relation = []
    for d in data:
        d = d.split("\t")
        if len(d) > 1:
            relation.append(d[1])
    return relation


entity = set(extract_entity(train) + extract_entity(valid) + extract_entity(test))
relation = set(extract_relation(train) + extract_relation(valid) + extract_relation(test))

with open("data/kg-bert/entities.txt","w+") as f:
    writer = csv.writer(f)
    for x in entity:
        writer.writerow([x])

with open("data/kg-bert/relations.txt","w+") as f:
    writer = csv.writer(f)
    for x in relation:
        writer.writerow([x])


with open("data/kg-bert/entity2text.txt","w+") as f:
    writer = csv.writer(f, delimiter="\t")
    for x in entity:
        writer.writerow([x, wordnet[x]['head_word'] + ", " + wordnet[x]['sense']])


with open("data/kg-bert/relation2text.txt","w+") as f:
    writer = csv.writer(f, delimiter="\t")
    for x in relation:
        writer.writerow([x, " ".join(x.split("_")[1:])])
