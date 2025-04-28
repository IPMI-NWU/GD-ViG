# [MICCAI2024] Gaze-directed Vision GNN for Mitigating Shortcut Learning in Medical Image
This project contains the training and testing code for the paper, as well as the model weights trained according to our algorithm. <br>
Paper link: https://link.springer.com/chapter/10.1007/978-3-031-72378-0_48

# Model Weights
The download links and extraction codes for our model weights are as follows：
https://pan.baidu.com/s/1S_pP58kKNSz3F2B3pvLX_w 
7777 

The download link for the model weights on Google Drive is as follows：
https://drive.google.com/drive/folders/1mIUxiqGHdDmnXm2xDH5_8feNEW9LZi6s?usp=sharing

# Dataset
1. The SIIM-ACR dataset can be downloaded from  https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data .
2. The gaze data for SIIM-ACR can be downloaded at https://github.com/HazyResearch/observational .
3. The division of SIIM-ACR we based on https://github.com/MoMarky/Eye-gaze-Guided-Vision-Transformer?tab=readme-ov-file
4. The EGR-CXR dataset can be downloaded from https://physionet.org/content/egd-cxr/1.0.0/ .

```
Data/<br>
     ├──MIMIC_Gaze/ 
          ├── test/ 
               ├── gaze/  
                    ├── xxx.png  
               ├── img/ 
                    ├── xxx.png  
          ├── train/ 
               ├── gaze/ 
                    ├── xxx.png  
               ├── img/ <br>
                    ├── xxx.png  
          ├── mimic_part.csv  

     ├──SIIM-ACR-Gaze/  
          ├── test/  
               ├── gaze/  
                    ├── xxx.png 
               ├── img/ 
                    ├── xxx.png  
          ├── train/ 
               ├── gaze/  
                    ├── xxx.png  
               ├── img/ <br>
                    ├── xxx.png 
          ├── siim_pneumothorax.csv 
          ├── test_list.csv 
          ├── train_list.csv  
```

# Citation
```
@InProceedings{10.1007/978-3-031-72378-0_48,
    author="Wu, Shaoxuan and Zhang, Xiao and Wang, Bin and Jin, Zhuo and Li, Hansheng and Feng, Jun",
    title="Gaze-Directed Vision GNN for Mitigating Shortcut Learning in Medical Image",
    booktitle="Medical Image Computing and Computer Assisted Intervention",
    year="2024",
    publisher="Springer Nature Switzerland",
    pages="514--524",
}
```
