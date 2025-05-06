import numpy as np


def compute_metrics(scores, query2targets, k_values=(1, 5, 10)):
    """
    计算Recall@K和mAP@K
    参数：
        scores: (N_queries, N_candidates) 相似度矩阵
        query2targets: list of list, 每个query对应的正确目标索引
        k_values: 需要计算的K值列表
    返回：
        dict: 包含各K值下的Recall和mAP
    """
    metrics = {}
    max_k = max(k_values)
    all_aps = []

    # 遍历每个查询
    for query_idx in range(scores.shape[0]):
        # 获取该查询的候选排序
        ranked_indices = np.argsort(-scores[query_idx])  # 从高到低排序
        relevant = np.array([int(idx in query2targets[query_idx]) for idx in ranked_indices])

        # 计算Recall@K
        for k in k_values:
            hit = np.sum(relevant[:k])
            metrics.setdefault(f"Recall@{k}", []).append(hit / len(query2targets[query_idx]))

        # 计算AP@K
        ap = 0.0
        correct_count = 0
        for i in range(min(max_k, len(ranked_indices))):
            if relevant[i]:
                correct_count += 1
                precision = correct_count / (i + 1)
                ap += precision / min(len(query2targets[query_idx]), max_k)
        all_aps.append(ap)

    # 汇总结果
    final_metrics = {}
    for k in k_values:
        final_metrics[f"Recall@{k}"] = np.mean(metrics[f"Recall@{k}"]) * 100
    final_metrics[f"mAP@{max_k}"] = np.mean(all_aps) * 100

    return final_metrics


def cross_modal_eval(scores_i2t, scores_t2i, txt2img, img2txt, k_values=(1, 5, 10)):
    """跨模态双向评估"""
    # Image->Text 评估
    # print("Evaluating Image->Text...")
    i2t_metrics = compute_metrics(scores_i2t, img2txt, k_values)

    # Text->Image 评估
    # print("\nEvaluating Text->Image...")
    t2i_metrics = compute_metrics(scores_t2i, [[tid] for tid in txt2img], k_values)  # 单目标

    # 综合结果
    return {
        "Image_to_Text": i2t_metrics,
        "Text_to_Image": t2i_metrics,
        "Average": {
            f"Recall@{k}": (i2t_metrics[f"Recall@{k}"] + t2i_metrics[f"Recall@{k}"]) / 2
            for k in k_values
        }
    }


if __name__=="__main__":
    import torch
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from flickr8k_dataloader import Flick8kDataset, dataset_collate
    from models.clip import train_clip_model
    root_dir = r'E:\PythonProject\clip_pytorch\flickr8k'
    val_dataset = Flick8kDataset(root_dir, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=2, pin_memory=False,
                            num_workers=1, collate_fn=dataset_collate)
    model = train_clip_model(
        r'E:\PythonProject\clip_pytorch\models\models_pth\ViT-B-16.pt').to('cuda:0')
    all_i2t = []
    all_t2i = []
    with torch.no_grad():
        model.eval()
        for i, batch in tqdm(enumerate(val_loader), desc="Evaluating",
                             total=len(val_loader)):
            images, texts = batch
            images = images.to('cuda')
            texts = texts.to('cuda')
            logits_per_image, logits_per_text = model(images, texts)

            all_i2t.append(logits_per_image.cpu().numpy())
            all_t2i.append(logits_per_text.cpu().numpy())

    scores_i2t = np.concatenate(all_i2t, axis=0)
    scores_t2i = np.concatenate(all_t2i, axis=0)

    metrics = cross_modal_eval(
        scores_i2t=scores_i2t,
        scores_t2i=scores_t2i,
        txt2img=val_dataset.txt_to_img,
        img2txt=val_dataset.img_to_txt,
        k_values=(1, 5, 10)
    )
    from pprint import pprint
    pprint(metrics)