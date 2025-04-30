import torch
import json
from dict_hub import get_entity_dict, get_all_triplet_dict
from triplet import EntityDict

def _setup_entity_dict() -> EntityDict:
    # if args.task == 'wiki5m_ind':
    #     return EntityDict(entity_dict_dir=os.path.dirname(args.valid_path),
    #                       inductive_test_path=args.valid_path)
    return get_entity_dict()

entity_dict = _setup_entity_dict()

n=20

ex_list = json.load(open('./llm_rerank/back/ex_list.json', 'r', encoding='utf-8'))
score_list = torch.load('./llm_rerank/back/score_list.pt')
index_list = torch.load('./llm_rerank/back/index_list.pt')
score_list = score_list[:, :n]
index_list = index_list[:, :n]
res_list = []

for i in range(len(ex_list)):
    ex = ex_list[i]
    ex_result = []
    for j in range(index_list.size(1)):
        index = index_list[i,j]
        ent = entity_dict.get_entity_by_idx(index).entity
        score = round(score_list[i,j].item(),4)
        ex_result.append({
            'entity':ent,
            'score':score
        })
    res_list.append({
        'head':ex['head'],
        'relation':ex['relation'],
        'tail':ex['tail'],
        'res':ex_result
    })

with open('./llm_rerank/back/b_temp.json', 'w') as f:
            json.dump(res_list, f)
f.close()


ex_list = json.load(open('./llm_rerank/forward/ex_list.json', 'r', encoding='utf-8'))
score_list = torch.load('./llm_rerank/forward/score_list.pt')
index_list = torch.load('./llm_rerank/forward/index_list.pt')
score_list = score_list[:, :n]
index_list = index_list[:, :n]
res_list = []

for i in range(len(ex_list)):
    ex = ex_list[i]
    ex_result = []
    for j in range(index_list.size(1)):
        index = index_list[i,j]
        ent = entity_dict.get_entity_by_idx(index).entity
        score = round(score_list[i,j].item(),4)
        ex_result.append({
            'entity':ent,
            'score':score
        })
    res_list.append({
        'head':ex['head'],
        'relation':ex['relation'],
        'tail':ex['tail'],
        'res':ex_result
    })

with open('./llm_rerank/forward/f_temp.json', 'w') as f:
            json.dump(res_list, f)
f.close()

