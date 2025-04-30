import torch
import json


a = 1

task = '' # '' or wn18rr/

def test_final(task):
    if task=='':
        path = 'FB15k237/alh_all'
    else:
        path = 'WN18RR'
    f_ress = json.load(open('/E22301291/SimKGCv2/data/'+path+'/final_f_res.json', 'r', encoding='utf-8'))
    score_list = torch.load('/E22301291/SimKGCv2/llm_rerank/'+task+'forward/score_list.pt')
    index_list = torch.load('/E22301291/SimKGCv2/llm_rerank/'+task+'forward/index_list.pt')

    for i in range(len(f_ress)):
        if f_ress[i]['if_new']==0:
            continue
        else:
            res_index = index_list[i,:20].tolist()
            llm_list = f_ress[i]['res']
            new_score_list = []
            for llm in llm_list:
                new_score_list.append(a*llm['lmm_score']+(1-a)*llm['score'])
#                 new_score_list.append(llm['score'])
        # 测试代码
#             index1=f_ress[i]['y']
#             index0=f_ress[i]['x']
#             temp = res_index[index0]
            new_score_list, res_index = zip(*(sorted(zip(new_score_list, res_index), key=lambda x:x[0], reverse=True)))
#             after = res_index[index1]
#             if (temp!=after):
#                 print('错误！')
            for j in range(len(llm_list)):
                score_list[i,j] = new_score_list[j]
                index_list[i,j] = res_index[j]
#             index_list[i, index1] = temp
#             if (temp!=after):
#                 print('错误！')

    torch.save(score_list, '/E22301291/SimKGCv2/llm_rerank/'+task+'forward/final_score_list.pt')
    torch.save(index_list, '/E22301291/SimKGCv2/llm_rerank/'+task+'forward/final_index_list.pt')

    f_ress = json.load(open('/E22301291/SimKGCv2/data/'+path+'/final_b_res.json', 'r', encoding='utf-8'))
    score_list = torch.load('/E22301291/SimKGCv2/llm_rerank/'+task+'back/score_list.pt')
    index_list = torch.load('/E22301291/SimKGCv2/llm_rerank/'+task+'back/index_list.pt')


    for i in range(len(f_ress)):
        if f_ress[i]['if_new']==0:
            continue
        else:
            res_index = index_list[i,:20].tolist()
            llm_list = f_ress[i]['res']
            new_score_list = []
            for llm in llm_list:
                new_score_list.append(a*llm['lmm_score']+(1-a)*llm['score'])
#                 new_score_list.append(llm['score'])
#             index1=f_ress[i]['y']
#             index0=f_ress[i]['x']
#             temp = res_index[index0]
            new_score_list, res_index = zip(*(sorted(zip(new_score_list, res_index), key=lambda x:x[0], reverse=True)))
            for j in range(len(llm_list)):
                score_list[i,j] = new_score_list[j]
                index_list[i,j] = res_index[j]
#             index_list[i, index1] = temp

    torch.save(score_list, '/E22301291/SimKGCv2/llm_rerank/'+task+'back/final_score_list.pt')
    torch.save(index_list, '/E22301291/SimKGCv2/llm_rerank/'+task+'back/final_index_list.pt')
    
test_final(task)
