import torch
from PIL import Image
import numpy as np
from models.clip_utils import load, tokenize

def detect_image(image_path, text_language, model_pth):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load(model_pth, device=device)  # 载入模型
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = tokenize(text_language).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        idx = np.argmax(probs, axis=1)
        print("Label probs:", probs)
        for i in range(image.shape[0]):   # batch
            id = idx[i]
            print('{}:\t{}'.format(text_language[id], probs[i, id]))
            print('image {}:\t{}'.format(i, [v for v in zip(text_language, probs[i])]))

if __name__=="__main__":
    model_pth_path = r"E:\PythonProject\clip_pytorch\models\models_pth\ViT-B-16.pt"
    image_path = "./CLIP.png"
    text_language = ["a schematic photo", "a dog", "a black cat", "a handsome man"]
    detect_image(image_path, text_language, model_pth_path)
