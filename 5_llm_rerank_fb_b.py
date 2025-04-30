import json
import os
from zhipuai import ZhipuAI
from logger_config import logger
from util import get_ent2des_fb, get_rel2des_fb, get_rev_rel2des_fb, first_to_rest_ratio

path = 'data/FB15k237'
client = ZhipuAI(api_key="") # 填写您自己的APIKey


def final_b_rerank():
    exs = json.load(open(os.path.join(path, 'b_temp.json'), 'r', encoding='utf-8'))
    if os.path.exists(os.path.join(path, 'b_llm_final.json')):
        llm_rerank = json.load(open(os.path.join(path, 'b_llm_final.json'), 'r', encoding='utf-8'))
    else:
        llm_rerank = []
    logger.info('Start rerank')
    count = len(llm_rerank)
    # 获取字典
    rel2des = get_rev_rel2des_fb()
    ent2des = get_ent2des_fb()


    for i in range(count, len(exs)):
        ex = exs[i]
        h = ex['head']
        r = ex['relation']
        t = ex['tail']
        res = ex['res']
        r_index = ' '.join(r.split(' ')[1:])
        frist_res = res[0]['entity']
        logger.info('now triple: ({}, {}, {}), over {}'.format(h, r, t, i))
        if t == frist_res:
            llm_rerank.append({
                "h": h,
                "r": r,
                "t": t,
                "res": res,
                "llm_res": ''
            })
            if (i + 1) % 100 == 0:
                with open(os.path.join(path, 'f_llm_final.json'), 'w') as f:
                    json.dump(llm_rerank, f, ensure_ascii=False)
            continue
        # all_score_list = []
        # for j in range(len(res)):
        #     all_score_list.append(res[j]['score'])
        # p = first_to_rest_ratio(all_score_list)
        # if p > 1.2:
        #     llm_rerank.append({
        #         "h": h,
        #         "r": r,
        #         "t": t,
        #         "res": res,
        #         "llm_res": ''
        #     })
        #     if (i + 1) % 100 == 0:
        #         with open(os.path.join(path, 'f_llm_final.json'), 'w') as f:
        #             json.dump(llm_rerank, f)
        #     continue
        text = ''
        for re in res:
            text += '{}, '.format(re['entity'])
        text = text.strip().strip(',')
        chat1 = "A complete triplet in a knowledge graph consists of head-entities, relations, and tail-entities. For example: (Zürich, month travel destination monthly climate climate travel destination travel , October)，It means that the month travel destination monthly climate climate travel destination travel of Zürich is October。({}, {}, ?)是一个缺失的三元组，{} is the head-entity, about which the description is:".format(h,r,h)
        chat1_res = ent2des[h]
        chat2 = "{} is the relation of this missing triplet, which means:".format(r)
        chat2_res = rel2des[r_index]
        chat3 = "For missing triples, the entities in the missing part of the candidate set need to be scored to find the correct tail-entity. " \
                "For example, for missing triples :(Zürich, month travel destination monthly climate climate travel destination travel , ?), " \
                "The candidate triples of the missing part are: Central European Time Zone-US, October, Eastern European Time Zone." \
                "The scoring results are as follows：October: 0.82, Central European Time Zone-US: 0.52, Eastern European Time Zone: 0.42, " \
                "According to the scoring results, October has the highest score, so October is missing tail-entity of (Zürich, month travel destination monthly climate climate travel destination travel , ?). " \
                "Then for missing triples ({}, {},?), the candidate set for its missing part is: {},The scoring results are as follows：".format(h, r, text)
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",  # 填写需要调用的模型名称 glm-3-turbo, glm-4, glm-4-airx,
                messages=[
                    {"role": "user", "content": chat1},
                    {"role": "assistant", "content": chat1_res},
                    {"role": "user", "content": chat2},
                    {"role": "assistant", "content": chat2_res},
                    {"role": "user", "content": chat3}
                ],
            )
            llm_res = response.choices[0].message.content
        except Exception as e:
            logger.info('now error: {} !!!!'.format(h, i))
            logger.info(e)
            llm_res = ''

        llm_rerank.append({
            "h": h,
            "r": r,
            "t": t,
            "res": res,
            "llm_res": llm_res
        })
        if (i+1)%100 == 0:
            with open(os.path.join(path, 'b_llm_final.json'), 'w') as f:
                json.dump(llm_rerank, f, ensure_ascii=False)

    with open(os.path.join(path, 'b_llm_final.json'), 'w') as f:
        json.dump(llm_rerank, f, ensure_ascii=False)
    f.close()



# f_rerank()
# final_f_rerank()
# b_rerank()
final_b_rerank()