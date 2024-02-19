import os
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, \
    f1_score
from torch import nn
from torch.utils.data import DataLoader
from End2End_SIIMACR.dataset_e2e import DatasetGaze_e2e
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from End2End_SIIMACR.models.GAViG.GAViG import GAViG


def infer_e2e():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    # -------------------------------------------------------
    size = 224
    device = torch.device('cuda')
    # -------------------------------------------------------
    test_dir = '../Data/SIIM-ACR-Gaze/test/'
    csv_path = '../Data/SIIM-ACR-Gaze/siim_pneumothorax.csv'
    # -------------------------------------------------------
    model = GAViG(num_classes=2)
    base = "output/exp_SIIMACR/"
    model_path = base + "SIIMACR_0.872.pth"
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval().to(device)
    # -------------------------------------------------------
    output_gaze_dir = base + "Gaze map/"
    if not os.path.exists(output_gaze_dir):
        os.makedirs(output_gaze_dir)
    # -------------------------------------------------------
    dataset_test = DatasetGaze_e2e(test_dir, csv_path, 'test', size)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    # -------------------------------------------------------
    pred_list = []
    label_list = []
    pred_score_list = []
    mse = 0.
    for img, gaze, label, img_path in dataloader_test:
        img = img.to(device)
        label = label.to(device)
        # -------------------------------------------------------
        # pred
        # -------------------------------------------------------
        output = model(img)
        gaze_pred = model.get_gaze()
        _, pred = torch.max(output, 1)
        pred_score = torch.nn.Softmax(dim=1)(output)
        # -------------------------------------------------------
        # gaze map
        # -------------------------------------------------------
        gaze_pred = (gaze_pred - gaze_pred.min()) / (gaze_pred.max() - gaze_pred.min())
        gaze_pred = np.array(gaze_pred[0][0].detach().cpu()) * 255
        output_img = Image.fromarray(gaze_pred).convert('L')
        name = img_path[0].split('/')[-1]
        output_img.save(output_gaze_dir + name)
        # -------------------------------------------------------
        # append to list
        # -------------------------------------------------------
        pred_score_list.append(pred_score[0][1].cpu().detach().numpy().tolist())
        pred_list.append(pred.cpu().detach().numpy().tolist())
        label_list.append(label.cpu().detach().numpy().tolist())

    pred_list = [b for a in pred_list for b in a]
    label_list = [b for a in label_list for b in a]
    # -------------------------------------------------------
    # print metrics
    # -------------------------------------------------------
    acc = accuracy_score(label_list, pred_list)
    print('Accuracy   : {:.4f}'.format(acc.item()))
    precision = precision_score(label_list, pred_list, average='weighted')
    print('Precision  : {:.4f}'.format(precision))
    recall = recall_score(label_list, pred_list, average='weighted')
    print('Recall     : {:.4f}'.format(recall))
    f1 = f1_score(label_list, pred_list, average='weighted')
    print('F1 score   : {:.4f}'.format(f1))
    auc = metrics.roc_auc_score(label_list, pred_score_list)
    print('AUC        : {:.4f}'.format(auc))
    cm = confusion_matrix(label_list, pred_list)
    print(cm)


if __name__ == '__main__':
    infer_e2e()
