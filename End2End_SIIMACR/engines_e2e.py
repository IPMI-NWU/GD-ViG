import datetime
import time
import torch
import torchvision
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import nn as nn


class Visualize_train(nn.Module):
    def __init__(self):
        super().__init__()

    def save_image(self, image, tag, epoch, writer):
        if tag == 'img':
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)
            grid = torchvision.utils.make_grid(image, nrow=1, pad_value=1)
        else:
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)
            grid = torchvision.utils.make_grid(image, nrow=1, pad_value=1)
        writer.add_image(tag, grid, epoch)

    def forward(self, img_list, gaze_list, gaze_pred_list,
                epoch, writer):
        self.save_image(img_list.float(), 'img', epoch, writer)
        self.save_image(gaze_list.float(), 'gaze', epoch, writer)
        self.save_image(gaze_pred_list.float(), 'gaze_pred', epoch, writer)


def train_one_epoch_e2e(model, dataloader_train, optimizer, device, epoch, args, writer):
    model.train()
    print_freq = 50
    total_steps = len(dataloader_train)
    start_time = time.time()

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    img_list, gaze_list, gaze_pred_list = [], [], []
    step = 0
    pred_list = []
    label_list = []
    for img, gaze, label, _ in dataloader_train:
        start = time.time()

        img = img.to(device)
        gaze = gaze.to(device)
        label = label.to(device)
        # -------------------------------------------------------
        # pred
        # -------------------------------------------------------
        cls_pred = model(img)
        gaze_pred = model.get_gaze()
        # -------------------------------------------------------
        # loss
        # -------------------------------------------------------
        loss_cls = criterion_cls(cls_pred, label)
        loss_reg = criterion_reg(gaze_pred, gaze)

        l1 = 1.
        l2 = 1.

        loss = l1 * loss_cls + l2 * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # -------------------------------------------------------
        # 可视化
        # -------------------------------------------------------
        if step == 0:
            img_list.append(img[0].detach())
            gaze_list.append(gaze[0].detach())
            gaze_pred_list.append(gaze_pred[0].detach())
        # -------------------------------------------------------
        # 打印loss
        # -------------------------------------------------------
        _, pred = torch.max(cls_pred, 1)
        pred_list.append(pred.cpu().detach().numpy().tolist())
        label_list.append(label.cpu().detach().numpy().tolist())
        if step % print_freq == 0:
            itertime = time.time() - start
            print('    lr: {:.6f}'.format(optimizer.param_groups[0]["lr"]))
            print('    loss: {:.4f}'.format(loss.item()))
            print('    [{} / {}] iter time: {:.4f}'.format(step, total_steps, itertime))
            print('    ------------------------------------')
        step = step + 1
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    pred_list = [b for a in pred_list for b in a]
    label_list = [b for a in label_list for b in a]
    acc = accuracy_score(pred_list, label_list)
    cm = confusion_matrix(pred_list, label_list)
    print(cm)
    print('Epoch: [{}] Total time: {} ({:.4f}  / it )'.format(epoch, total_time_str, total_time / total_steps))

    writer.add_scalar('train_loss', loss.item(), epoch)
    writer.add_scalar('train_acc', acc, epoch)

    visual_train = Visualize_train()
    visual_train(torch.stack(img_list), torch.stack(gaze_list), torch.stack(gaze_pred_list),
                 epoch, writer)
