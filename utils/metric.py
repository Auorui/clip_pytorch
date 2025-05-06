import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from flickr8k_dataloader import Flick8kDataset, dataset_collate
from models.clip import train_clip_model
from models.clip_utils import tokenize

def compute_metrics(scores_i2t, img2txt, scores_t2i, txt2img, k_values=(1, 5, 10)):
    metrics = {}
    total_i2t = scores_i2t.shape[0]
    total_t2i = scores_t2i.shape[0]
    max_k = max(k_values)

    # Image -> Text 评估
    i2t_ranks = []
    i2t_hits = {k: 0 for k in k_values}
    i2t_aps = {k: [] for k in k_values}

    for img_idx in range(total_i2t):
        scores = scores_i2t[img_idx]
        sorted_indices = np.argsort(scores)[::-1]
        gt_indices = img2txt[img_idx]

        # 基础指标计算
        best_rank = float('inf')
        for txt_idx in gt_indices:
            current_rank = np.where(sorted_indices == txt_idx)[0][0]
            if current_rank < best_rank:
                best_rank = current_rank
        best_rank += 1  # 转换为 1-based
        i2t_ranks.append(best_rank)

        # mAP@K 计算
        relevant_positions = []
        for pos, idx in enumerate(sorted_indices[:max_k], 1):  # 1-based 位置
            if idx in gt_indices:
                relevant_positions.append(pos)

        for k in k_values:
            # Recall@k
            if best_rank <= k:
                i2t_hits[k] += 1

            # mAP@k
            rel_in_k = [p for p in relevant_positions if p <= k]
            if not rel_in_k:
                i2t_aps[k].append(0.0)
                continue

            precisions = []
            for i, pos in enumerate(rel_in_k, 1):
                precisions.append(i / pos)
            map_score = sum(precisions) / min(len(gt_indices), k)
            i2t_aps[k].append(map_score)

    # Text -> Image 评估
    t2i_ranks = []
    t2i_hits = {k: 0 for k in k_values}
    t2i_aps = {k: [] for k in k_values}

    for txt_idx in range(total_t2i):
        scores = scores_t2i[txt_idx]
        sorted_indices = np.argsort(scores)[::-1]
        img_idx = txt2img[txt_idx]

        best_rank = np.where(sorted_indices == img_idx)[0][0] + 1
        t2i_ranks.append(best_rank)

        # mAP@K 计算
        relevant_pos = None
        for pos, idx in enumerate(sorted_indices[:max_k], 1):  # 1-based 位置
            if idx == img_idx:
                relevant_pos = pos
                break

        for k in k_values:
            # Recall@k
            if best_rank <= k:
                t2i_hits[k] += 1

            # mAP@k
            if relevant_pos and relevant_pos <= k:
                t2i_aps[k].append(1.0 / relevant_pos)
            else:
                t2i_aps[k].append(0.0)

    # Image->Text
    metrics["i2t"] = {
        **{f"R@{k}": round(i2t_hits[k] / total_i2t * 100, 4) for k in k_values},
        **{f"mAP@{k}": round(np.mean(i2t_aps[k]) * 100, 4) for k in k_values},
        "median_rank": round(np.median(i2t_ranks) * 1, 2),
        "mean_rank": round(np.mean(i2t_ranks) * 1, 2)
    }

    # Text->Image
    metrics["t2i"] = {
        **{f"R@{k}": round(t2i_hits[k] / total_t2i * 100, 4) for k in k_values},
        **{f"mAP@{k}": round(np.mean(t2i_aps[k]) * 100, 4) for k in k_values},
        "median_rank": round(np.median(t2i_ranks) * 1, 2),
        "mean_rank": round(np.mean(t2i_ranks) * 1, 2)
    }

    return metrics


def plot_metrics(metrics):
    import matplotlib.pyplot as plt
    from pyzjr.visualize import matplotlib_patch
    matplotlib_patch()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    k_values = [int(k.split('@')[1]) for k in metrics['i2t'] if k.startswith('R@')]

    # Image-to-Text
    axs[0].plot(k_values, [metrics['i2t'][f'R@{k}'] for k in k_values], label='Recall')
    axs[0].plot(k_values, [metrics['i2t'][f'mAP@{k}'] for k in k_values], label='mAP')
    axs[0].set_title('Image-to-Text Retrieval')

    # Text-to-Image
    axs[1].plot(k_values, [metrics['t2i'][f'R@{k}'] for k in k_values], label='Recall')
    axs[1].plot(k_values, [metrics['t2i'][f'mAP@{k}'] for k in k_values], label='mAP')
    axs[1].set_title('Text-to-Image Retrieval')

    for ax in axs:
        ax.set_xlabel('K')
        ax.set_ylabel('Score (%)')
        ax.legend()
    plt.show()

if __name__=="__main__":
    root_dir = r'E:\PythonProject\clip_pytorch\flickr8k'
    batch_size = 64
    val_dataset = Flick8kDataset(root_dir, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=False,
                            num_workers=1, collate_fn=dataset_collate)
    model = train_clip_model(
        r'E:\PythonProject\clip_pytorch\logs\2025_05_06_16_36_12\weights\best_epoch_weights.pth',
        jit=False).to('cuda:0')
    i_features = []
    t_features = []
    texts = val_dataset.text
    num_text = len(texts)  # 5000
    with torch.no_grad():
        model.eval()
        for i, batch in tqdm(enumerate(val_loader), desc="Evaluating image",
                             total=len(val_loader)):
            images, _ = batch
            images = images.to('cuda:0')
            images_feature = model.encode_image(images)
            i_features.append(images_feature)

        for i in tqdm(range(0, num_text, batch_size), desc="Evaluating text"):
            text = texts[i: min(num_text, i + batch_size)]
            with torch.no_grad():
                texts_feature = model.encode_text(tokenize(text).to('cuda:0'))
                t_features.append(texts_feature)

    i_features = torch.cat(i_features, 0)
    t_features = torch.cat(t_features, 0)
    # print(i_features.shape, t_features.shape)
    i_features = i_features / i_features.norm(dim=-1, keepdim=True)
    t_features = t_features / t_features.norm(dim=-1, keepdim=True)
    logits_per_image = i_features @ t_features.t()
    logits_per_text = logits_per_image.t()

    logits_per_image = logits_per_image.cpu().numpy()
    logits_per_text = logits_per_text.cpu().numpy()
    clip_metric = compute_metrics(logits_per_image, val_dataset.img_to_txt,
                                  logits_per_text, val_dataset.txt_to_img)

    from pprint import pprint
    pprint(clip_metric)

    plot_metrics(clip_metric)
