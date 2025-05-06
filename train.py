import os
import torch
import argparse
from tqdm import tqdm
from copy import deepcopy
import torch.nn as nn
import numpy as np
from utils import Flick8kDataset, Flick8kDataLoader, cross_modal_eval
from pyzjr import (load_owned_device, release_gpu_memory, AverageMeter, get_lr,
                   SeedEvery, show_config, loss_weights_dirs, get_optimizer,
                   redirect_console, multi_makedirs, LossHistory)
from torch.cuda.amp import GradScaler, autocast
from models import train_clip_model

def parse_args(known=False):
    parser = argparse.ArgumentParser(description='Classification Train')
    # 加载预训练权重，默认None
    parser.add_argument('--resume_training', type=str,
                        default=r"E:\PythonProject\clip_pytorch\models\models_pth\ViT-B-16.pt",
                        help="resume training from last checkpoint")
    # 日志文件存放路径
    parser.add_argument('--log_dir', type=str, default=r'./logs', help='log file path')
    # 数据集路径
    parser.add_argument('--dataset_path', type=str,
                        default=r'E:\PythonProject\clip_pytorch\flickr8k', help='dataset path')
    # 训练轮次epochs次数，默认为100轮
    parser.add_argument('--epochs', type=int, default=100, help='Training rounds')
    # 图片大小
    parser.add_argument('--input_shape', default=224, help='input image shape')
    # batch_size 批量大小 2 4 8,爆内存就改为1试试
    # 详细可看此篇 : https://blog.csdn.net/m0_62919535/article/details/132725967
    # 试过之后还是不行，那就给你的电脑放个假（关机休息）
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    # 初始学习率
    parser.add_argument('--lr', default=2e-5, help='Initial learning rate')
    # 用于优化器的动量参数，控制梯度更新的方向和速度。
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    # 用于优化器的权重衰减参数，用于抑制权重的过度增长，防止过拟合。
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for optimizer')
    # 优化器选择，可选adam、adamw、sgd
    parser.add_argument('--optimizer_type', type=str, default="adamw",
                        help='Optimizer selection, optional adam、adamw and sgd')
    # 训练过程中的保存pth文件频率, 不宜太频繁
    parser.add_argument('--freq', type=int, default=80, help='Save PTH file frequency')
    # 是否开启混合精度训练
    parser.add_argument('--use_amp', type=bool, default=False, help='Enable mixed precision training')
    return parser.parse_known_args()[0] if known else parser.parse_args()


class CLIPTrainEpoch():
    def __init__(
            self,
            model,
            total_epoch,
            loss_function,
            optimizer,
            lr_scheduler,
            use_amp=False,
            device=load_owned_device()
    ):
        self.device = device
        self.model = model.to(self.device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.total_epoch = total_epoch
        release_gpu_memory()
        self.scaler = None
        if use_amp:
            self.scaler = GradScaler()

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        train_losses = AverageMeter()
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{self.total_epoch}',
                  mininterval=0.3) as pbar:
            for batch in train_loader:
                images, texts = batch
                images = images.to(self.device)
                texts = texts.to(self.device)
                self.optimizer.zero_grad()
                with autocast(enabled=self.scaler is not None):
                    # 与作者论文中给出的伪代码相同
                    logits_per_image, logits_per_text = self.model(images, texts)
                    # logits_per_text = logits_per_image.t()
                    labels = torch.arange(len(logits_per_image)).long().to(self.device)
                    loss_logits_per_image = self.loss_function(logits_per_image, labels)
                    loss_logits_per_text = self.loss_function(logits_per_text, labels)
                    loss = (loss_logits_per_image + loss_logits_per_text) / 2

                if self.scaler is not None:
                    loss = torch.nan_to_num(loss)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                with torch.no_grad():
                    import math
                    self.model.logit_scale.clamp_(max=math.log(100))
                self.lr_scheduler.step()
                train_losses.update(loss.item())
                pbar.set_postfix(**{'train_loss': train_losses.avg,
                                    'lr': get_lr(self.optimizer)})
                pbar.update(1)
        return train_losses.avg

    def evaluate(self, val_loader, epoch):
        self.model.eval()
        val_losses = AverageMeter()
        all_i2t = []
        all_t2i = []
        with tqdm(total=len(val_loader), desc=f'Epoch {epoch}/{self.total_epoch}',
                  mininterval=0.3) as pbar:
            for batch in val_loader:
                images, texts = batch
                with torch.no_grad():
                    images = images.to(self.device)
                    texts = texts.to(self.device)
                    logits_per_image, logits_per_text = self.model(images, texts)
                    # logits_per_text = logits_per_image.t()
                    all_i2t.append(logits_per_image.cpu().numpy())
                    all_t2i.append(logits_per_text.cpu().numpy())
                    labels = torch.arange(len(logits_per_image)).long().to(images.device)
                    loss_logits_per_image = self.loss_function(logits_per_image, labels)
                    loss_logits_per_text = self.loss_function(logits_per_text, labels)
                    loss = (loss_logits_per_image + loss_logits_per_text) / 2
                    val_losses.update(loss.item())

                pbar.set_postfix(**{'val_loss': val_losses.avg})
                pbar.update(1)
        scores_i2t = np.concatenate(all_i2t, axis=0)
        scores_t2i = np.concatenate(all_t2i, axis=0)
        metrics = cross_modal_eval(
            scores_i2t=scores_i2t,
            scores_t2i=scores_t2i,
            txt2img=val_loader.dataset.txt_to_img,
            img2txt=val_loader.dataset.img_to_txt,
            k_values=(1, 5, 10)
        )
        from pprint import pprint
        pprint(metrics)
        return val_losses.avg

    def save_models(self, save_dir, epoch, save_period, total_loss, val_loss, loss_history):
        loss_history.append_loss(epoch, total_loss, val_loss)
        print('Epoch:' + str(epoch) + '/' + str(self.total_epoch))
        print('Total Loss: %.5f || Val Loss: %.5f ' % (total_loss, val_loss))
        if epoch % save_period == 0 or epoch == self.total_epoch:
            torch.save(deepcopy(self.model).half().state_dict(), os.path.join(save_dir,
                                f'ep{epoch}_loss{total_loss:.3f}_val_loss{val_loss:.3f}.pth'))

        if len(loss_history.val_loss) <= 1 or val_loss <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(deepcopy(self.model).half().state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(deepcopy(self.model).half().state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


if __name__=="__main__":
    args = parse_args()
    SeedEvery(11)
    loss_log_dirs, save_model_dirs, timelog_dir = loss_weights_dirs(args.log_dir)
    # 记录控制台过程
    redirect_console(os.path.join(timelog_dir, f'out.log'))
    show_config(head="Auorui's custom classification training template", args=args)
    train_dataset = Flick8kDataset(
        args.dataset_path,
        target_shape=args.input_shape,
        is_train=True,
    )
    val_dataset = Flick8kDataset(
        args.dataset_path,
        target_shape=args.input_shape,
        is_train=False,
    )
    model = train_clip_model(args.resume_training).to('cuda:0')
    loss_history = LossHistory(loss_log_dirs, model, input_shape=args.input_shape)

    train_loader, val_loader = Flick8kDataLoader(train_dataset, val_dataset, args.batch_size)
    optimizer = get_optimizer(model,
                              optimizer_type=args.optimizer_type,
                              init_lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    # lr_scheduler = WarmUpLR(optimizer, train_loader, args.epochs)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)
    loss_function = nn.CrossEntropyLoss()

    clip_train = CLIPTrainEpoch(
        model,
        total_epoch=args.epochs,
        loss_function=loss_function,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        use_amp=args.use_amp
    )

    for epoch in range(args.epochs):
        current_epoch = epoch + 1
        total_loss = clip_train.train_one_epoch(train_loader, current_epoch)
        val_loss = clip_train.evaluate(val_loader, current_epoch)

        clip_train.save_models(
            save_dir=save_model_dirs,
            epoch=current_epoch,
            total_loss=total_loss,
            val_loss=val_loss,
            save_period=args.freq,
            loss_history=loss_history
        )

    loss_history.close()



