import torch
import os
import re
import cv2
import json
import random
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from pyzjr.data.datasets import BaseDataset
from pyzjr.data import to_2tuple
from pyzjr.nn import num_worker

from models.clip_utils import tokenize

class Flick8kDataset(BaseDataset):
    def __init__(self, root_dir, target_shape=224, language='en', is_train=True, transform=None):
        """"""
        super(Flick8kDataset, self).__init__()
        self.target_shape = to_2tuple(target_shape)
        self.transform = transform
        self.is_train = is_train
        self.json_path = os.path.join(root_dir,
                            f"{language}_train.json" if is_train else f"{language}_val.json")
        self.json_lines = json.load(open(self.json_path, mode='r', encoding='utf-8'))
        self.text = []   # 存放文本信息
        self.image = []   # 存储图像路径
        self.txt_to_img = {}
        self.img_to_txt = {}
        txt_id = 0
        for img_id, line_info in enumerate(self.json_lines):
            # print(img_id, line_info)
            self.image.append(os.path.join(root_dir, line_info['image']))
            self.img_to_txt[img_id] = []
            for i, caption in enumerate(line_info['caption']):
                self.text.append(self.pre_caption(caption, 77))
                self.img_to_txt[img_id].append(txt_id)
                self.txt_to_img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.json_lines)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        caption = self.text[np.random.choice(self.img_to_txt[idx])]
        if self.transform:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        else:
            image = self.read_image(image_path,
                                    to_rgb=True,
                                    normalize=False)
            image = self.class_augument(image, self.target_shape, self.is_train)
            image = self.hwc2chw(image)
            # image = torch.from_numpy(image).float()
        # caption = tokenize(caption) # 在这里使用 tokenize 会多出一个 dim
        return image, caption

    def class_augument(self, image, target_shape, is_train, prob=.5, hue=.1, sat=0.7, val=0.3):
        h, w = target_shape
        ih, iw = image.shape[:2]
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_image = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
        top = (h - nh) // 2
        left = (w - nw) // 2
        new_image[top:top + nh, left:left + nw] = resized_image
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        if is_train:
            if random.random() > (1 - prob):
                new_image = np.flip(new_image, axis=1)
            r = random.randint(0, 3)
            new_image = np.rot90(new_image, r, (0, 1))
            # 转换到HSV颜色空间
            image_data = np.array(new_image, np.uint8)
            r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
            dtype = image_data.dtype
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
            image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            new_image = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        # 将图像归一化到[0, 1]范围
        new_image = np.array(new_image, dtype='float32')
        new_image = new_image / 255.0
        return new_image

    def pre_caption(self, caption, max_words):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption

def dataset_collate(batch):
    images = []
    captions = []
    for image, caption in batch:
        images.append(image)
        captions.append(caption)
    captions = tokenize(captions)
    images = torch.from_numpy(np.array(images)).float()
    return images, captions

def Flick8kDataLoader(train_dataset, val_dataset, batch_size, collate_fn=dataset_collate):
    num_workers = 0 # num_worker(batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=False,
                              num_workers=num_workers, shuffle=True, drop_last=True,
                              collate_fn=collate_fn)
    # 一般验证集不太多, 所以默认给2, 不计算梯度所以速度也很快
    val_loader = DataLoader(val_dataset, batch_size=2, pin_memory=False,
                            num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader

def flickr8k_transform(n_px=224):
    from PIL import Image
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

if __name__=="__main__":
    root_dir = r'E:\PythonProject\clip_pytorch\flickr8k'
    # 使用 flickr8k_transform 需要重新修改好 collate_fn
    # train_dataset = Flick8kDataset(root_dir, is_train=True, transform=flickr8k_transform(224))
    # val_dataset = Flick8kDataset(root_dir, is_train=False, transform=flickr8k_transform(224))
    # train_loader, val_loader = Flick8kDataLoader(train_dataset, val_dataset, 2, collate_fn= None)
    train_dataset = Flick8kDataset(root_dir, is_train=True)
    val_dataset = Flick8kDataset(root_dir, is_train=False)

    train_loader, val_loader = Flick8kDataLoader(train_dataset, val_dataset, 2)
    for batch in train_loader:
        image, label = batch
        print("Batch images shape:", image.shape)  # (4, 3, 224, 224)
        print("Batch texts shape:", label.shape)  # (4, max_seq_len)
        print("Texts dtype:", label.dtype)