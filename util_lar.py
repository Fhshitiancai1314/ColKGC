import json
from collections import defaultdict as ddict
import os
from config.config_wn18 import args
import nltk
import random
from tqdm import tqdm

def get_gt(n_rel, path):
    # 读数据
    data = json.load(open(os.path.join(path, "trainP.json"), 'r', encoding='utf-8'))
    gt = ddict(list)
    # 这里是获取 Example
    cnt = len(data)
    for i in range(cnt):
        obj = data[i]
        head = int(obj['headId'])
        rel = int(obj['relId'])
        tail = int(obj['tailId'])
        gt[(head, rel)].append(tail)
        gt[(tail, rel + n_rel)].append(head)

    return gt

def getDict(path, filename,  mode=None):
    id2name = []
    file_name = os.path.join(path, filename)
    with open(file_name, encoding='utf-8') as file:
        lines = file.read().strip('\n').split('\n')
    for i in range(0, len(lines)):
        id, name = lines[i].split('\t')
        if mode == 'rel':
            name = name.split('_')
            name = ' '.join(name)
        id2name.append(name)
    return id2name

def read_file(path, filename, mode='desc'):
    id2name = []
    file_name = os.path.join(path, filename)
    data = json.load(open(file_name, 'r', encoding='utf-8'))
    cnt = len(data)
    for i in range(cnt):
        obj = data[i]
        if mode=='desc':
            name = obj['entity_desc']
        elif mode == 'name':
            name = obj['entity_desc']
        id2name.append(name)
    return id2name

def read_name(path):
    ent_name_file = 'entities.json'
    rel_name_file = 'relations.dict'
    ent_names = read_file(path, ent_name_file, 'name')
    rel_names = getDict(path, rel_name_file, 'rel')
    return ent_names, rel_names

def get_lar_sample_bank(configs, text_dict):
    def process(tokens, stop_words):
        tokens = map(lambda x: x.lower(), tokens)
        tokens = filter(lambda x: x not in stop_words, tokens)
        return list(tokens)

    ent_names = text_dict['ent_names']
    ent_descs = text_dict['ent_descs']
    nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words('english')
    tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
    name_token2ids, desc_token2ids = ddict(list), ddict(list)
    name_id2tokens, desc_id2tokens = dict(), dict()
    desc_stop_words = stop_words

    for i in tqdm(range(configs.n_ent), desc='Processing'):
        name, desc = ent_names[i], ent_descs[i]
        if configs.task == 'wn18rr':
            name = name.split(' , ')[0]
        name_tokens, desc_tokens = tokenizer.tokenize(name), tokenizer.tokenize(desc)
        name_tokens, desc_tokens = process(name_tokens, stop_words), process(desc_tokens, desc_stop_words)
        for token in name_tokens:
            name_token2ids[token].append(i)
        name_id2tokens[i] = list(set(name_tokens))
        for token in desc_tokens:
            desc_token2ids[token].append(i)
        desc_id2tokens[i] = list(set(desc_tokens))
    if configs.max_lar_samples > 0:
        for key, value in name_token2ids.items():
            if len(value) > configs.max_lar_samples:
                name_token2ids[key] = random.sample(value, configs.max_lar_samples)
        for key, value in desc_token2ids.items():
            if len(value) > configs.max_lar_samples:
                desc_token2ids[key] = random.sample(value, configs.max_lar_samples)
    name_lars, desc_lars = {}, {}
    for i in tqdm(range(configs.n_ent), desc='Processing name'):
        lar_list = list(set([ids for token in name_id2tokens[i] for ids in name_token2ids[token]]))
        if configs.max_lar_samples > 0 and len(lar_list) > configs.max_lar_samples:
            lar_list = random.sample(lar_list, configs.max_lar_samples)
        name_lars[i] = lar_list
    for i in tqdm(range(configs.n_ent), desc='Processing desc'):
        lar_list = list(set([ids for token in desc_id2tokens[i] for ids in desc_token2ids[token]]))
        if configs.max_lar_samples > 0 and len(lar_list) > configs.max_lar_samples:
            lar_list = random.sample(lar_list, configs.max_lar_samples)
        desc_lars[i] = lar_list

    lars_dict = {'name_lars': name_lars, 'desc_lars': desc_lars}
    # return token_dict, negs_dict
    return lars_dict


def get_before_get_lar():
    gt = get_gt(args.n_rel, args.path)
    ent_names, rel_names = read_name(args.path)
    ent_descs = read_file(args.path, 'entities.json', 'desc')

    text_dict = {
        'ent_names': ent_names,
        'rel_names': rel_names,
        'ent_descs': ent_descs,
    }

    if args.n_lar > 0:
        lars_dict = get_lar_sample_bank(args, text_dict)

    name_lars = lars_dict['name_lars']
    desc_lars = lars_dict['desc_lars']

    return gt, name_lars, desc_lars

