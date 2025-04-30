import json
import os
from zhipuai import ZhipuAI
from logger_config import logger
from util import get_ent2des, get_rel2des, get_rev_rel2des

path = 'data/WN18RR'
client = ZhipuAI(api_key="") # 填写您自己的APIKey

def final_b_rerank():
    exs = json.load(open(os.path.join(path, 'b_temp.json'), 'r', encoding='utf-8'))
    if os.path.exists(os.path.join(path, 'b_llm_final.json')):
        llm_rerank = json.load(open(os.path.join(path, 'b_llm_final.json'), 'r', encoding='utf-8'))
    else:
        llm_rerank = []
    logger.info('Start rerank')
    count = len(llm_rerank)
    rel2des = get_rev_rel2des()
    ent2des = get_ent2des()

    for i in range(count, len(exs)):
        ex = exs[i]
        h = ex['head']
        r = ex['relation']
        t = ex['tail']
        res = ex['res']
        r_index = ' '.join(r.split(' ')[1:])
        logger.info('now triple: ({}, {}, {}), over {}'.format(h, r, t, i))
        text = ''
        for re in res:
            text += '{}, '.format(re['entity'])
        text = text.strip().strip(',')
        chat1 = "A complete triplet in a knowledge graph consists of head-entities, relations, and tail-entities. For example: (trade_name_NN_1, member of domain usage, metharbital_NN_1), It means that the member of domain usage of trade_name_NN_1 is metharbital_NN_1.({}, {},?) is a missing triplet, {} is the head-entity, about which the description is:".format(h,r,h)
        chat1_res = ent2des[h]
        chat2 = "{} is the relation of this missing triplet, which means:".format(r)
        chat2_res = rel2des[r_index]
        chat3 = "For missing triples, the entities in the missing part of the candidate set need to be scored to find the correct tail-entity. " \
                "For example, for missing triples :(trade_name_NN_1, member of domain usage,?). " \
                "The candidate triples of the missing part are: sinequan_NN_1,nitrostat_NN_2, metharbital_NN_1. " \
                "The scoring results are as follows：metharbital_NN_1:0.7761, sinequan_NN_1:0.652, nitrostat_NN_2:0.6071." \
                "According to the scoring results, metharbital_NN_1 has the highest score, so metharbital_NN_1 is missing tail-entity of (trade_name_NN_1, member of domain usage,?)." \
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

def final_f_rerank():
    exs = json.load(open(os.path.join(path, 'f_temp.json'), 'r', encoding='utf-8'))
    if os.path.exists(os.path.join(path, 'f_llm_final.json')):
        llm_rerank = json.load(open(os.path.join(path, 'f_llm_final.json'), 'r', encoding='utf-8'))
    else:
        llm_rerank = []
    logger.info('Start rerank')
    count = len(llm_rerank)
    rel2des = get_rel2des()
    ent2des = get_ent2des()

    for i in range(count, len(exs)):
        ex = exs[i]
        h = ex['head']
        r = ex['relation']
        t = ex['tail']
        res = ex['res']
        frist_res = res[0]['entity']
        logger.info('now triple: ({}, {}, {}), over {}'.format(h, r, t, i))
        # if t == frist_res:
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
        text = text.strip().strip(',')
        chat1 = "A complete triplet in a knowledge graph consists of head-entities, relations, and tail-entities. For example: (trade_name_NN_1, member of domain usage, metharbital_NN_1), It means that the member of domain usage of trade_name_NN_1 is metharbital_NN_1.({}, {},?) is a missing triplet, {} is the head-entity, about which the description is:".format(
            h, r, h)
        chat1_res = ent2des[h]
        chat2 = "{} is the relation of this missing triplet, which means:".format(r)
        chat2_res = rel2des[r]
        chat3 = "For missing triples, the entities in the missing part of the candidate set need to be scored to find the correct tail-entity. " \
                "For example, for missing triples :(trade_name_NN_1, member of domain usage,?). " \
                "The candidate triples of the missing part are: sinequan_NN_1,nitrostat_NN_2, metharbital_NN_1. " \
                "The scoring results are as follows：metharbital_NN_1:0.7761, sinequan_NN_1:0.652, nitrostat_NN_2:0.6071." \
                "According to the scoring results, metharbital_NN_1 has the highest score, so metharbital_NN_1 is missing tail-entity of (trade_name_NN_1, member of domain usage,?)." \
                "Then for missing triples ({}, {},?), the candidate set for its missing part is: {},The scoring results are as follows：".format(
            h, r, text)
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
            with open(os.path.join(path, 'f_llm_final.json'), 'w') as f:
                json.dump(llm_rerank, f)

    with open(os.path.join(path, 'f_llm_final.json'), 'w') as f:
        json.dump(llm_rerank, f)
    f.close()

def f_rerank():
    exs = json.load(open(os.path.join(path, 'f_temp.json'), 'r', encoding='utf-8'))
    if os.path.exists(os.path.join(path, 'f_llm_temp.json')):
        llm_rerank = json.load(open(os.path.join(path, 'f_llm_temp.json'), 'r', encoding='utf-8'))
    else:
        llm_rerank = []
    logger.info('Start rerank')
    count = len(llm_rerank)
    for i in range(count, len(exs)):
        ex = exs[i]
        h = ex['head']
        r = ex['relation']
        t = ex['tail']
        res = ex['res']
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
                with open(os.path.join(path, 'f_llm_temp.json'), 'w') as f:
                    json.dump(llm_rerank, f)
            continue
        text = ''
        for re in res:
            text += '{}, '.format(re['entity'])
        text = text.strip().strip(',')
        chat1 = "下面我会给定一个缺失三元组并给出缺失部分的候选词，你需要按照缺失部分的可能结果给候选词打分。比如，缺失三元组:（(trade_name_NN_1, member of domain usage, ？) ，它缺少的部分的候选三元组分别是： metharbital_NN_1, sinequan_NN_1  ,nitrostat_NN_2, clozaril_NN_1, protropin_NN_1, maxzide_NN_1, amobarbital_NN_1, zyloprim_NN_1, trimox_NN_1, kaopectate_NN_1, soap_NN_3, lipitor_NN_1,rophy_NN_1 0.4502,tenormin_NN_1 0.4365,ru_486_NN_1 0.4364,bromo-seltzer_NN_1 0.4316,stilboestrol_NN_1 0.4312,super_c_NN_1 0.4311,brioschi_NN_1 0.4266,norepinephrine_NN_1。打分: metharbital_NN_1:0.7761, sinequan_NN_1:0.652, nitrostat_NN_2:0.6071, clozaril_NN_1:0.6038, protropin_NN_1:0.5783, maxzide_NN_1:0.5691, amobarbital_NN_1:0.4779, zyloprim_NN_1:0.471, trimox_NN_1:0.4629, kaopectate_NN_1:0.4599, soap_NN_3:0.4565, lipitor_NN_1:0.4538, rophy_NN_1:0.4502, tenormin_NN_1:0.4365, ru_486_NN_1:0.4364, bromo-seltzer_NN_1:0.4316, stilboestrol_NN_1:0.4312, super_c_NN_1:0.4311, brioschi_NN_1:0.4266, norepinephrine_NN_1:0.4252。那么，对于缺失三元组：({}, {}, ?)，它缺少的部分的候选三元组分别是：{}。打分：".format(h, r, text)
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",  # 填写需要调用的模型名称 glm-3-turbo, glm-4, glm-4-airx,
                messages=[
                    {"role": "user", "content": chat1},
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
            with open(os.path.join(path, 'f_llm_temp.json'), 'w') as f:
                json.dump(llm_rerank, f)

    with open(os.path.join(path, 'f_llm_temp.json'), 'w') as f:
        json.dump(llm_rerank, f)
    f.close()

def b_rerank():
    exs = json.load(open(os.path.join(path, 'b_temp.json'), 'r', encoding='utf-8'))
    if os.path.exists(os.path.join(path, 'b_llm_temp.json')):
        llm_rerank = json.load(open(os.path.join(path, 'b_llm_temp.json'), 'r', encoding='utf-8'))
    else:
        llm_rerank = []
    logger.info('Start rerank')
    count = len(llm_rerank)
    for i in range(count, len(exs)):
        ex = exs[i]
        h = ex['head']
        r = ex['relation']
        t = ex['tail']
        res = ex['res']
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
                with open(os.path.join(path, 'b_llm_temp.json'), 'w') as f:
                    json.dump(llm_rerank, f)
            continue
        text = ''
        for re in res:
            text += '{}, '.format(re['entity'])
        text = text.strip().strip(',')
        chat1 = "下面我会给定一个缺失三元组并给出缺失部分的候选词，你需要按照缺失部分的可能结果给候选词打分。比如，缺失三元组:（(trade_name_NN_1, member of domain usage, ？) ，它缺少的部分的候选三元组分别是： metharbital_NN_1, sinequan_NN_1  ,nitrostat_NN_2, clozaril_NN_1, protropin_NN_1, maxzide_NN_1, amobarbital_NN_1, zyloprim_NN_1, trimox_NN_1, kaopectate_NN_1, soap_NN_3, lipitor_NN_1,rophy_NN_1 0.4502,tenormin_NN_1 0.4365,ru_486_NN_1 0.4364,bromo-seltzer_NN_1 0.4316,stilboestrol_NN_1 0.4312,super_c_NN_1 0.4311,brioschi_NN_1 0.4266,norepinephrine_NN_1。打分: metharbital_NN_1:0.7761, sinequan_NN_1:0.652, nitrostat_NN_2:0.6071, clozaril_NN_1:0.6038, protropin_NN_1:0.5783, maxzide_NN_1:0.5691, amobarbital_NN_1:0.4779, zyloprim_NN_1:0.471, trimox_NN_1:0.4629, kaopectate_NN_1:0.4599, soap_NN_3:0.4565, lipitor_NN_1:0.4538, rophy_NN_1:0.4502, tenormin_NN_1:0.4365, ru_486_NN_1:0.4364, bromo-seltzer_NN_1:0.4316, stilboestrol_NN_1:0.4312, super_c_NN_1:0.4311, brioschi_NN_1:0.4266, norepinephrine_NN_1:0.4252。那么，对于缺失三元组：({}, {}, ?)，它缺少的部分的候选三元组分别是：{}。打分：".format(h, r, text)
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",  # 填写需要调用的模型名称 glm-3-turbo, glm-4, glm-4-airx,
                messages=[
                    {"role": "user", "content": chat1},
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
            with open(os.path.join(path, 'b_llm_temp.json'), 'w') as f:
                json.dump(llm_rerank, f, ensure_ascii=False)

    with open(os.path.join(path, 'b_llm_temp.json'), 'w') as f:
        json.dump(llm_rerank, f, ensure_ascii=False)
    f.close()

# f_rerank()
final_f_rerank()
# b_rerank()
final_b_rerank()