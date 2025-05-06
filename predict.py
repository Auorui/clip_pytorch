import cv2
import torch
from PIL import Image
import numpy as np
import torch.nn as nn
from models.clip import train_clip_model
from models.clip_utils import tokenize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode


class Predict(nn.Module):
    def __init__(self, model_pth, target_shape=224):
        super(Predict, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = train_clip_model(model_pth, jit=False).to(self.device)
        self.transform = Compose([
            Resize(target_shape, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(target_shape),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def preprocess(self, image_path):
        pil_image = Image.open(image_path).convert('RGB')
        return self.transform(pil_image).unsqueeze(0)

    def forward(self, image_path, text_language):
        image = self.preprocess(image_path).float().to(self.device)
        text = tokenize(text_language).to(self.device)
        self.model.eval()
        original_image = cv2.imread(image_path)
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            idx = np.argmax(probs, axis=1)
            print("Label probs:", probs)
            for i in range(image.shape[0]):  # batch
                id = idx[i]
                prediction = f"{text_language[id]}: {probs[i, id]:.2f}"

                (text_width, text_height), baseline = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, .8, 2)
                text_x = (original_image.shape[1] - text_width) // 2
                text_y = original_image.shape[0] - 10
                cv2.putText(original_image, prediction, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, .8,
                            (255, 255, 255), 2, cv2.LINE_AA)

                print(prediction)
                print('image {}:\t{}'.format(i, [v for v in zip(text_language, probs[i])]))
                cv2.imshow("predict opencv image", original_image)
                cv2.waitKey(0)

if __name__=="__main__":
    while True:
        model_pth_path = r"E:\PythonProject\clip_pytorch\logs\2025_05_06_16_36_12\weights\best_epoch_weights.pth"
        image_path = input("请输入图像路径：")
        # E:\PythonProject\clip_pytorch\flickr8k\images\35506150_cbdb630f4f.jpg
        # E:\PythonProject\clip_pytorch\flickr8k\images\3682277595_55f8b16975.jpg
        text_language = ["A man in a red jacket is sitting on a bench whilst cooking a meal",
                         "A greyhound walks in the rain through a large puddle",
                         "A group of people ride in a race",
                         "A cyclist is performing a jump near to a railing and a brick wall"]
        model_predict = Predict(model_pth_path)
        model_predict(image_path, text_language)
