import json
import os

path = './llm_rerank/'
exs = json.load(open('./llm_rerank/forward/f_temp.json', 'r', encoding='utf-8'))
llm_exs = json.load(open('./data/FB15k237/final_f_res.json', 'r', encoding='utf-8'))

for i in range(len(exs)):
    ex = exs[i]
    llm = llm_exs[i]
    llm_rerank = []
    if llm['if_new'] == 0:
        exs[i]['if_new'] = 0
    else:
        llm_res = llm['res']
        res = ex['res']
        for j in range(len(res)):
            for k in range(len(llm_res)):
                if res[j]['entity'] == llm_res[k]['entity']:
                    res[j]['lmm_score'] = llm_res[k]['lmm_score']
            if 'lmm_score' not in res[j].keys():
                res[j]['lmm_score']=0
        exs[i]['res'] = res
        exs[i]['if_new'] = 1

with open('./data/FB15k237/final_f_res.json', 'w') as f:
    json.dump(exs, f, ensure_ascii=False)
f.close()

exs = json.load(open('./llm_rerank/back/b_temp.json', 'r', encoding='utf-8'))
llm_exs = json.load(open('./data/FB15k237/final_b_res.json', 'r', encoding='utf-8'))

for i in range(len(exs)):
    ex = exs[i]
    llm = llm_exs[i]
    llm_rerank = []
    if llm['if_new'] == 0:
        exs[i]['if_new'] = 0
    else:
        llm_res = llm['res']
        res = ex['res']
        for j in range(len(res)):
            for k in range(len(llm_res)):
                if res[j]['entity'] == llm_res[k]['entity']:
                    res[j]['lmm_score'] = llm_res[k]['lmm_score']
            if 'lmm_score' not in res[j].keys():
                res[j]['lmm_score']=0
        exs[i]['res'] = res
        exs[i]['if_new'] = 1

with open('./data/FB15k237/final_b_res.json', 'w') as f:
    json.dump(exs, f, ensure_ascii=False)
f.close()