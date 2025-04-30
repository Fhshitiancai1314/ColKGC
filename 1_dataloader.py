from logger_config import logger
import json
import os
import random

def getList(path, mode=None):
    if mode == 'rel':
        data = json.load(open(os.path.join(path, 'relations.json'), 'r', encoding='utf-8'))
        logger.info('Load {} examples from {}'.format(len(data), path))
        rels = []
        for key, value in data.items():
            rels.append(value)
        return rels
    elif mode == 'ent':
        data = json.load(open(os.path.join(path, 'entities.json'), 'r', encoding='utf-8'))
        logger.info('Load {} examples from {}'.format(len(data), path))
        ents = []
        for ex in data:
            ents.append({ex['entity']: ex['entity_desc']})
        return ents


def rel2Dict(path, rels):
    data = json.load(open(os.path.join(path, 'trainP.json'), 'r', encoding='utf-8'))
    rel2dict = {}
    for rel in rels:
        count = 0
        arr = []
        for ex in data:
            if count <= 5:
                if ex['relation'] == rel:
                    # temp = '({}, {}, {})'.format(ex['head'], ex['relation'], ex['tail'])
                    temp = {}
                    temp['head'] = ex['head']
                    temp['tail'] = ex['tail']
                    arr.append(temp)
                    count += 1
                else:
                    continue
            else:
                break
        rel2dict[rel] = arr

    with open(os.path.join(path, 'rel23.json'), 'w', encoding='utf-8') as f:
        json.dump(rel2dict, f, ensure_ascii=False)
    f.close()

def ent2Dict(path):
    data = json.load(open(os.path.join(path, 'trainP.json'), 'r', encoding='utf-8'))
    ents = json.load(open(os.path.join(path, 'entities.json'), 'r', encoding='utf-8'))
    ent2dict = []
    for ent in ents:
        count = 0
        arr = []
        for ex in data:
            if count <= 3:
                if ex['head'] == ent['entity']:
                    # temp = '({}, {}, {})'.format(ex['head'], ex['relation'], ex['tail'])
                    temp = {
                        'relation': ex['relation'],
                        'tail': ex['tail']
                    }
                    arr.append(temp)
                    count += 1
                else:
                    continue
            else:
                break

        if ent['entity_id']=='/m/01c0cc':
            print()

        ent2dict.append({
            'entity_id': ent['entity_id'],
            'entity': ent['entity'],
            'entity_desc': ent['entity_desc'],
            'ex': arr
        })

    with open(os.path.join(path, 'ent23.json'), 'w', encoding='utf-8') as f:
        json.dump(ent2dict, f, ensure_ascii=False)
    f.close()

def get3(path, num):
    data = json.load(open(os.path.join(path, 'trainP.json'), 'r', encoding='utf-8'))
    lenth = len(data)
    arr = []
    for i in range(num):
        index = random.randint(0, lenth-1)
        ex = data[index]
        arr.append(ex)
    return arr

if __name__ == '__main__':
    path = 'data/FB15k237'
    # rels = getList(path, mode='rel')
    # rel2Dict(path, rels)
    ent2Dict(path)








