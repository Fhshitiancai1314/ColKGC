import tqdm
from logger_config import logger
import torch
from time import time
import json
from typing import List, Tuple

def compute_metrics(score_list: torch.tensor,
                    index_list: torch.tensor,
                    target: List[int],
                    k=3, batch_size=256) -> Tuple:
    total = score_list.size(0)
    target = torch.LongTensor(target).unsqueeze(-1).to(score_list.device)
    topk_scores, topk_indices = [], []
    ranks = []

    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0
    sorted_score_list, sorted_indices_lsit = [], []
    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size
        batch_target = target[start:end]

        batch_sorted_score = score_list[start:end, :]
        batch_sorted_indices = index_list[start:end, :]

        target_rank = torch.nonzero(batch_sorted_indices.eq(batch_target).long(), as_tuple=False)

        for idx in range(batch_sorted_score.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx
            cur_rank = idx_rank[1]

            # 0-based -> 1-based
            cur_rank += 1
            mean_rank += cur_rank
            mrr += 1.0 / cur_rank
            hit1 += 1 if cur_rank <= 1 else 0
            hit3 += 1 if cur_rank <= 3 else 0
            hit10 += 1 if cur_rank <= 10 else 0
            ranks.append(cur_rank)

        topk_scores.extend(batch_sorted_score[:, :k].tolist())
        topk_indices.extend(batch_sorted_indices[:, :k].tolist())
    

    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    metrics = {k: round(v / total, 4) for k, v in metrics.items()}
    assert len(topk_scores) == total
    return topk_scores, topk_indices, metrics, ranks


def eval_single_direction(eval_forward=True,
                          batch_size=256,
                         task='') -> dict:
    start_time = time()
    if eval_forward:
        target = json.load(open('/E22301291/SimKGCv2/llm_rerank/'+task+'forward/target.json', 'r', encoding='utf-8'))
        score_list = torch.load('/E22301291/SimKGCv2/llm_rerank/'+task+'forward/final_score_list.pt')
        index_list = torch.load('/E22301291/SimKGCv2/llm_rerank/'+task+'forward/final_index_list.pt')
    else:
        target = json.load(open('/E22301291/SimKGCv2/llm_rerank/'+task+'back/target.json', 'r', encoding='utf-8'))
        score_list = torch.load('/E22301291/SimKGCv2/llm_rerank/'+task+'back/final_score_list.pt')
        index_list = torch.load('/E22301291/SimKGCv2/llm_rerank/'+task+'back/final_index_list.pt')
    logger.info('predict tensor done, compute metrics...')
    _, _, metrics, _ = compute_metrics(score_list=score_list, index_list=index_list,
                                       target=target, batch_size=batch_size)
    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))
    
    logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
    return metrics

task = '' # '' or wn18rr/

forward_metrics = eval_single_direction(eval_forward=True, task=task)
backward_metrics = eval_single_direction(eval_forward=False, task=task)

metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
logger.info('Averaged metrics: {}'.format(metrics))
