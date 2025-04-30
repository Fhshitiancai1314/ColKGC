from zhipuai import ZhipuAI
import json
import os
from logger_config import logger
from util import get_tokenizer

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import brown



path = 'data/FB15k237'
client = ZhipuAI(api_key="") # 填写您自己的APIKey

def getRel_fb():
    exs = json.load(open(os.path.join(path, 'rel23.json'), 'r', encoding='utf-8'))
    relDict = []
    logger.info('Start generating rels.json')
    count = 0
    for key, val in exs.items():
        if key == 'form of government country location ':
            relDict.append({
                'relation': key,
                'rel_inverse': 'the Countries implementing this government form',
                'rel_desc': 'The definition of the government form of the country should be',
                'inver_rel_desc': 'The country implementing this government form is'
            })
            count += 1
            continue
        logger.info('now relation: {}, over {}'.format(key, count))
        h1, t1 = val[0]['head'], val[0]['tail']
        h2, t2 = val[1]['head'], val[1]['tail']
        h3, t3 = val[2]['head'], val[2]['tail']
        chat1 = "知识图谱中一个完整的三元组由头实体，关系，尾实体构成。现在有一个需求，给定一些关系缺失的不完整三元组，你需要给出缺失的关系，并且给出其通俗的描述。比如：对于缺失三元组(Indonesia, ?, Unitary state), (Equatorial Guinea, ?, Presidential system), (Japan, ?, Parliamentary system), 其缺失的关系应该为(10字以内)：form of government country location。其通俗的描述为（15字以内）：The definition of the government form of the country should be；对于缺失三元组(Unitary state, ?, Indonesia), (Presidential system, ?, Equatorial Guinea), (Parliamentary system, ?, Japan), 其缺失的关系应该为（10字以内）：the Countries implementing this government form。其通俗的描述为（15字以内）：The country implementing this government form is；那么，对于缺失三元组({}, ?, {}), ({}, ?, {}), ({}, ?, {})， 其缺失的关系应该是(10字以内)：".format(h1, t1, h2, t2, h3, t3)
        response = client.chat.completions.create(
            model="glm-4-0520",  # 填写需要调用的模型名称 glm-3-turbo, glm-4
            messages=[
                {"role": "user", "content": chat1},
                {"role": "assistant", "content": key},
                {"role": "user", "content": "其通俗的描述应该为（15字以内）："},
            ],
        )
        desc = response.choices[0].message.content
        chat2 = "对于缺失三元组({}, ?, {}), ({}, ?, {}), ({}, ?, {}), 其缺失的关系应该为（10字以内）：".format(t1, h1, t2, h2, t3, h3)
        response = client.chat.completions.create(
            model="glm-4-0520",  # 填写需要调用的模型名称 glm-3-turbo, glm-4
            messages=[
                {"role": "user", "content": chat1},
                {"role": "assistant", "content": key},
                {"role": "user", "content": "其通俗的描述应该为（15字以内）："},
                {"role": "assistant", "content": desc},
                {"role": "user", "content": chat2},
            ],
        )
        inverse = response.choices[0].message.content
        response = client.chat.completions.create(
            model="glm-4-0520",  # 填写需要调用的模型名称 glm-3-turbo, glm-4
            messages=[
                {"role": "user", "content": chat1},
                {"role": "assistant", "content": key},
                {"role": "user", "content": "其通俗的描述应该为（15字以内）："},
                {"role": "assistant", "content": desc},
                {"role": "user", "content": chat2},
                {"role": "assistant", "content": inverse},
                {"role": "user", "content": '其通俗的描述应该为（15字以内）：'},
            ],
        )
        inv_desc = response.choices[0].message.content
        relDict.append({
            'relation': key,
            'rel_inverse': inverse,
            'rel_desc': desc,
            'inver_rel_desc':inv_desc
        })
        count+=1

    with open(os.path.join(path, 'relv2.json'), 'w') as f:
        json.dump(relDict, f)
    f.close()


def getRel_fbv2():
    exs = json.load(open(os.path.join(path, 'rel23.json'), 'r', encoding='utf-8'))
    relDict = []
    logger.info('Start generating rels.json')
    count = 0
    for key, val in exs.items():
        if key == 'form of government country location ':
            relDict.append({
                'relation': key,
                'rel_desc': 'The definition of the government form of the country should be',
            })
            count += 1
            continue
        logger.info('now relation: {}, over {}'.format(key, count))
        h1, t1 = val[0]['head'], val[0]['tail']
        h2, t2 = val[1]['head'], val[1]['tail']
        h3, t3 = val[2]['head'], val[2]['tail']
        chat1 = "The role of relations in knowledge graphs is to link head and tail entities, and understanding relationships is crucial for understanding triplets. Below, I will provide relation and the triplets it belong to. To understand the function of relation, I need you to provide a description of the relationships, such as:" \
                "\nrelation: form of government country location" \
                "\ntriples: (Dominican Republic, form of government country location , Republic), (Saint Vincent and the Grenadines, form of government country location , Parliamentary system), (Liechtenstein, form of government country location , Parliamentary system)" \
                "\nrelation describe: how a country's government is organized" \
                "\nThen" \
                "\nrelation: {}" \
                "\ntriples: ({}, {}, {}), ({}, {}, {}), ({}, {}, {})" \
                "\nrelation describe:".format(key, h1, key, t1,  h2, key, t2, h3, key, t3)

        response = client.chat.completions.create(
            model="glm-4-flash",  # 填写需要调用的模型名称 glm-3-turbo, glm-4, glm-4-0520
            messages=[
                {"role": "user", "content": chat1},
            ]
        )
        desc = response.choices[0].message.content

        relDict.append({
            'relation': key,
            'rel_desc': desc,
        })
        count+=1

    with open(os.path.join(path, 'relv5.json'), 'w') as f:
        json.dump(relDict, f, ensure_ascii=False)
    f.close()

def getRel_fbv3():
    exs = json.load(open(os.path.join(path, 'relations.json'), 'r', encoding='utf-8'))
    exs2 = json.load(open(os.path.join(path, 'relv4.json'), 'r', encoding='utf-8'))
    relDict = []
    logger.info('Start generating rels.json')
    count = 0
    for key, val in exs.items():
        arr = key.split('/')
        temp = arr.pop(0)
        arr.append(temp)
        arr.reverse()
        rel_inverse = '/'.join(arr)
        ex2 = exs2[count]
        rel_desc = ex2['rel_desc']
        inver_rel_desc = ex2['inver_rel_desc']
        relDict.append({
            'relation': val,
            'ori_relation': key,
            'rel_inverse': rel_inverse,
            'rel_desc': rel_desc,
            'inver_rel_desc': inver_rel_desc
        })
        count += 1

    with open(os.path.join(path, 'relv5.json'), 'w') as f:
        json.dump(relDict, f, ensure_ascii=False)
    f.close()

def getRel_fbv3():
    exs = json.load(open(os.path.join(path, 'relations.json'), 'r', encoding='utf-8'))
    exs2 = json.load(open(os.path.join(path, 'relv5.json'), 'r', encoding='utf-8'))
    relDict = []
    logger.info('Start generating rels.json')
    count = 0
    for key, val in exs.items():
        arr = key.split('/')
        arr.pop(0)
        rel = ' '.join(arr)
        arr.reverse()
        rel_inverse = ' '.join(arr)
        ex2 = exs2[count]
        rel_desc = ex2['rel_desc']
        inver_rel_desc = ex2['inver_rel_desc']
        relDict.append({
            'relation': val,
            'ori_relation': rel,
            'rel_inverse': rel_inverse,
            'rel_desc': rel_desc,
            'inver_rel_desc': inver_rel_desc
        })
        count += 1

    with open(os.path.join(path, 'relv6.json'), 'w') as f:
        json.dump(relDict, f, ensure_ascii=False)
    f.close()

def getRel():
    exs = json.load(open(os.path.join(path, 'rel23.json'), 'r', encoding='utf-8'))
    relDict = []
    logger.info('Start generating rels.json')
    count = 0
    for key, val in exs.items():
        logger.info('now relation: {}, over {}'.format(key, count))
        h1, t1 = val[0]['head'], val[0]['tail']
        h2, t2 = val[1]['head'], val[1]['tail']
        h3, t3 = val[2]['head'], val[2]['tail']
        chat1 = "The triplet in Knowledge Graph is composed of head-entity, relation and tail-entity. than, i will provide some complete triples: ({}, {}, {}), ({}, {}, {}), ({}, {}, {}). {} is the relationship in these triplets, please summarize its meaning (within 10 words).".format(h1, key, t1, h2, key, t2, h3, key, t3, key)
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称 glm-3-turbo, glm-4
            messages=[
                {"role": "user", "content": chat1},
            ],
        )
        desc = response.choices[0].message.content
        chat2 = "Now I will provide some incomplete triples:({}, ?, {}), ({}, ?, {}), ({}, ?, {}). the relation in these triplets are same and should be:".format(t1, h1, t2, h2, t3, h3)
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称 glm-3-turbo, glm-4
            messages=[
                {"role": "user", "content": chat1},
                {"role": "assistant", "content": desc},
                {"role": "user", "content": chat2},
            ],
        )
        inverse = response.choices[0].message.content
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称 glm-3-turbo, glm-4
            messages=[
                {"role": "user", "content": chat1},
                {"role": "assistant", "content": desc},
                {"role": "user", "content": chat2},
                {"role": "assistant", "content": desc},
                {"role": "user", "content": 'please summarize its meaning (within 10 words).'},
            ],
        )
        inv_desc = response.choices[0].message.content
        relDict.append({
            'relation': key,
            'rel_inverse': inverse,
            'rel_desc': desc,
            'inver_rel_desc':inv_desc
        })
        count+=1

    with open(os.path.join(path, 'rels.json'), 'w') as f:
        json.dump(relDict, f)
    f.close()

# 待修改，生成的句子不一定满足长度需求，且生成的句子不一定好
def getEnt_sel():
    exs = json.load(open(os.path.join(path, 'ent23.json'), 'r', encoding='utf-8'))
    brown.categories()

    if os.path.exists(os.path.join(path, 'ents.json')):
        entDict = json.load(open(os.path.join(path, 'ents.json'), 'r', encoding='utf-8'))
    else:
        entDict = []
    logger.info('Start generating ents.json')
    count = len(entDict)
    tokenizer = get_tokenizer()
    for i in range(count, len(exs)):
        ex = exs[i]
        logger.info('now entity: {}, over {}'.format(ex['entity'], i))
        ent = ex['entity']
        ent_id = ex['entity_id']
        ent_desc = ex['entity_desc']
        wl = len(word_tokenize(ent_desc))
        sl = len(sent_tokenize(ent_desc))
        a = wl/sl
        if_new = 0
        if a > 20 or a < 8:
            fors = []
            for trp in ex['ex']:
                str_trp = '({}, {}, {})'.format(ent, trp['relation'], trp['tail'])
                fors.append(str_trp)
            str_ex = ', '.join(fors)
            # chat1 = "In Knowledge Graph, triples are composed of head-etitiy, relation, and tail-etitiy. {} is an entity in the Knowledge Graph, and its associated triplets include: {}. Its description is: {}\nBut this description is not appropriate. Please understand its meaning in the triplet here and combine it with the above description to provide a more reasonable description. The new description is:(给出新的英文描述，且在15字以内)".format(ent, str_ex, ent_desc)
            chat1 = "知识图谱中三元组由头实体，关系，尾实体构成。{}是知识图谱中的一个实体，与它相关的三元组有：{}， 它的描述是：{}. 但是这个描述并不恰当，请根据现有的描述结合你对这个词的了解，给出更合理的描述，新的英文描述是（要求15词以内）：".format(ent,str_ex, ent_desc)
            # chat1 = "{} is an entity in a triplet, and its description is {}. please understand the above description and provide a more appropriate entity description, within 20 words.".format(ent, ent_desc)
            try:
                response = client.chat.completions.create(
                    model="glm-3-turbo",  # 填写需要调用的模型名称 glm-3-turbo, glm-4
                    messages=[
                        {"role": "user", "content": chat1},
                    ],
                )
                llm_desc = response.choices[0].message.content
                if_new = 1
            except Exception as e:
                logger.info('now error: {} !!!!'.format(ex['entity'], i))
                logger.info(e)
                llm_desc = ''
                if_new = 0
        else:
            llm_desc = ''

        entDict.append({
            "entity_id": ent_id,
            "entity": ent,
            "entity_desc": ent_desc,
            "llm_desc": llm_desc,
            "if_new": if_new
        })
        if (i+1)%100 == 0:
            with open(os.path.join(path, 'ents.json'), 'w') as f:
                json.dump(entDict, f)

    with open(os.path.join(path, 'ents.json'), 'w') as f:
        json.dump(entDict, f)
    f.close()

# 待修改，生成的句子不一定满足长度需求，且生成的句子不一定好
def getEnt_all():
    exs = json.load(open(os.path.join(path, 'ent23.json'), 'r', encoding='utf-8'))
    if os.path.exists(os.path.join(path, 'ents_all.json')):
        entDict = json.load(open(os.path.join(path, 'ents_all.json'), 'r', encoding='utf-8'))
    else:
        entDict = []
    logger.info('Start generating ents.json')
    count = len(entDict)
    for i in range(count, len(exs)):
        ex = exs[i]
        logger.info('now entity: {}, over {}'.format(ex['entity'], i))
        ent = ex['entity']
        ent_id = ex['entity_id']
        ent_desc = ex['entity_desc']
        if_new = 0
        fors = []
        for trp in ex['ex']:
            str_trp = '({}, {}, {})'.format(ent, trp['relation'], trp['tail'])
            fors.append(str_trp)
        str_ex = ', '.join(fors)
        # chat1 = "The triplet in Knowledge Graph is composed of head-entity, relationship and tail-entity. Given an example of a triplet: {}. {} is head-entity in these triplets, and its description: {}. Please understand its meaning in triples and provide a more appropriate description based on the above description (within 15 words).".format(str_ex, ent, ent_desc)
        chat1 = "{} is an entity in the triplet, and its description is {}. please understand the above description and provide a more appropriate entity description, within 20 words.".format(ent, ent_desc)
        try:
            response = client.chat.completions.create(
                model="glm-3-turbo",  # 填写需要调用的模型名称 glm-3-turbo, glm-4
                messages=[
                    {"role": "user", "content": chat1},
                ],
            )
            llm_desc = response.choices[0].message.content
            if_new = 1
        except Exception as e:
            logger.info('now error: {} !!!!'.format(ex['entity'], i))
            logger.info(e)
            llm_desc = ''

        entDict.append({
            "entity_id": ent_id,
            "entity": ent,
            "entity_desc": ent_desc,
            "llm_desc": llm_desc,
            "if_new": if_new
        })
        if (i+1)%100 == 0:
            with open(os.path.join(path, 'ents_all.json'), 'w') as f:
                json.dump(entDict, f)

    with open(os.path.join(path, 'ents_all.json'), 'w') as f:
        json.dump(entDict, f)
    f.close()


def getEnt_fbv2():
    exs = json.load(open(os.path.join(path, 'ent23.json'), 'r', encoding='utf-8'))
    if os.path.exists(os.path.join(path, 'entsv4.json')):
        entDict = json.load(open(os.path.join(path, 'entsv4.json'), 'r', encoding='utf-8'))
    else:
        entDict = []
    logger.info('Start generating ents.json')
    count = len(entDict)
    for i in range(count, len(exs)):
        ex = exs[i]
        logger.info('now entity: {}, over {}'.format(ex['entity'], i))
        ent = ex['entity']
        ent_id = ex['entity_id']
        ent_desc = ex['entity_desc']
        if_new = 0

        # chat1 = "The triplet in Knowledge Graph is composed of head-entity, relationship and tail-entity. Given an example of a triplet: {}. {} is head-entity in these triplets, and its description: {}. Please understand its meaning in triples and provide a more appropriate description based on the above description (within 15 words).".format(str_ex, ent, ent_desc)
        chat1 = "{} \nSimplify the above paragraph to retain only the key points".format(ent_desc)
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",  # 填写需要调用的模型名称 glm-3-turbo, glm-4
                messages=[
                    {"role": "user", "content": chat1},
                ],
            )
            llm_desc = response.choices[0].message.content
            if_new = 1
        except Exception as e:
            logger.info('now error: {} !!!!'.format(ex['entity'], i))
            logger.info(e)
            llm_desc = ''

        entDict.append({
            "entity_id": ent_id,
            "entity": ent,
            "entity_desc": ent_desc,
            "llm_desc": llm_desc,
            "if_new": if_new
        })
        if (i+1)%100 == 0:
            with open(os.path.join(path, 'entsv4.json'), 'w') as f:
                json.dump(entDict, f, ensure_ascii=False)

    with open(os.path.join(path, 'entsv4.json'), 'w') as f:
        json.dump(entDict, f, ensure_ascii=False)
    f.close()


def getEnt_fbv3():
    exs = json.load(open(os.path.join(path, 'ent23.json'), 'r', encoding='utf-8'))
    if os.path.exists(os.path.join(path, 'entsv5.json')):
        entDict = json.load(open(os.path.join(path, 'entsv5.json'), 'r', encoding='utf-8'))
    else:
        entDict = []
    logger.info('Start generating ents.json')
    count = len(entDict)
    for i in range(count, len(exs)):
        ex = exs[i]
        logger.info('now entity: {}, over {}'.format(ex['entity'], i))
        ent = ex['entity']
        ent_id = ex['entity_id']
        ent_desc = ex['entity_desc']
        fors = []
        for trp in ex['ex']:
            str_trp = '({}, {}, {})'.format(ent, trp['relation'], trp['tail'])
            fors.append(str_trp)
        str_ex = ', '.join(fors)

        if_new = 0

        # chat1 = "The triplet in Knowledge Graph is composed of head-entity, relationship and tail-entity. Given an example of a triplet: {}. {} is head-entity in these triplets, and its description: {}. Please understand its meaning in triples and provide a more appropriate description based on the above description (within 15 words).".format(str_ex, ent, ent_desc)
        chat1 = "Triplet:(Dominican Republic, form of government country location , Republic), (Dominican Republic, olympics olympic medal honor medals won olympic participating country olympics , 1984 Summer Olympics), (Dominican Republic, currency dated money value measurement unit gdp nominal per capita statistical region location , United States Dollar)\n" \
                "Entity：Dominican Republic\n" \
                "Entity_description:\"The Dominican Republic is a nation on the island of Hispaniola, part of the Greater Antilles archipelago in the Caribbean region. The western three-eighths of the island is occupied by the nation of Haiti, making Hispaniola one of two Caribbean islands, along with Saint Martin, that are shared by two" \
                "Simplify the above paragraph to retain only the key points".format(ent_desc)
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",  # 填写需要调用的模型名称 glm-3-turbo, glm-4
                messages=[
                    {"role": "user", "content": chat1},
                ],
            )
            llm_desc = response.choices[0].message.content
            if_new = 1
        except Exception as e:
            logger.info('now error: {} !!!!'.format(ex['entity'], i))
            logger.info(e)
            llm_desc = ''

        entDict.append({
            "entity_id": ent_id,
            "entity": ent,
            "entity_desc": ent_desc,
            "llm_desc": llm_desc,
            "if_new": if_new
        })
        if (i+1)%100 == 0:
            with open(os.path.join(path, 'entsv4.json'), 'w') as f:
                json.dump(entDict, f, ensure_ascii=False)

    with open(os.path.join(path, 'entsv4.json'), 'w') as f:
        json.dump(entDict, f, ensure_ascii=False)
    f.close()

def getEnt_fb():
    exs = json.load(open(os.path.join(path, 'entities.json'), 'r', encoding='utf-8'))
    entDict = []
    logger.info('Start generating ents.json')
    count = len(entDict)
    for i in range(count, len(exs)):
        ex = exs[i]
        logger.info('now entity: {}, over {}'.format(ex['entity'], i))
        ent = ex['entity']
        ent_id = ex['entity_id']
        ent_desc = ex['entity_desc']
        if_new = 1
        try:
            llm_desc = sent_tokenize(ent_desc)[0]
        except Exception as e:
            llm_desc = ""
            logger.info('now error: {} !!!!'.format(ent))



        entDict.append({
            "entity_id": ent_id,
            "entity": ent,
            "entity_desc": ent_desc,
            "llm_desc": llm_desc,
            "if_new": if_new
        })
        # if (i+1)%100 == 0:
        #     with open(os.path.join(path, 'ents_all.json'), 'w') as f:
        #         json.dump(entDict, f)

    with open(os.path.join(path, 'ents.json'), 'w') as f:
        json.dump(entDict, f)
    f.close()

def getEnt_local():
    exs = json.load(open(os.path.join(path, 'ent23.json'), 'r', encoding='utf-8'))
    if os.path.exists(os.path.join(path, 'ents.json')):
        entDict = json.load(open(os.path.join(path, 'ents.json'), 'r', encoding='utf-8'))
    else:
        entDict = []
    logger.info('Start generating ents.json')
    count = len(entDict)
    tokenizer = get_tokenizer()
    for i in range(count, len(exs)):
        ex = exs[i]
        logger.info('now entity: {}, over {}'.format(ex['entity'], i))
        ent = ex['entity']
        ent_id = ex['entity_id']
        ent_desc = ex['entity_desc']
        length = len(tokenizer.tokenize(ent_desc))
        if_new = 0
        if length > 20 or length < 10:
            fors = []
            for trp in ex['ex']:
                str_trp = '({}, {}, {})'.format(ent, trp['relation'], trp['tail'])
                fors.append(str_trp)
            str_ex = ', '.join(fors)
            # chat1 = "The triplet in Knowledge Graph is composed of head-entity, relationship and tail-entity. Given some example of complete triplets: {}. {} is the head-entity in these triplets, and its description: {}. please understand the above description and provide a more appropriate entity description, within 15 words.".format(str_ex, ent, ent_desc)
            chat1 = "{} is an entity in a triplet, and its description is {}. please understand the above description and provide a more appropriate entity description, within 15 words.".format(ent, ent_desc)
            response = client.chat.completions.create(
                model="glm-4",  # 填写需要调用的模型名称 glm-3-turbo, glm-4
                messages=[
                    {"role": "user", "content": chat1},
                ],
            )
            llm_desc = response.choices[0].message.content
            if_new = 1
        else:
            llm_desc = ''

        entDict.append({
            "entity_id": ent_id,
            "entity": ent,
            "entity_desc": ent_desc,
            "llm_desc":llm_desc,
            "if_new": if_new
        })
        if (i+1)%100 == 0:
            with open(os.path.join(path, 'ents.json'), 'w') as f:
                json.dump(entDict, f)

    with open(os.path.join(path, 'ents.json'), 'w') as f:
        json.dump(entDict, f)
    f.close()


getRel_fbv3()