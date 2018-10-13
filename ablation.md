# log_123
1:decoder_edge

2:context

3:decoder_mask

统一：头部训练15个epoch， 学习率1e-3；全部再训练5个epoch， 学习率1e-4.

Baseline: mod, mask rcnn only detection

实验一：context-n2-n5  **c25**

实验二：context-n2-n6  **c26**

实验三：context-n2-n6 + decoder_mask  **c26dm**

实验四：context-n2-n6 + decoder_edge  **c26de** 在损失函数中，只有识别为镜子的roi的edge才有损失。

# 实验结果
### c25
mean_mAP_box_50      0.9283945615197569
 
mean_mAP_box_75      0.8688096524481727 

mean_mAP_box_85      0.7752341076949041
### c26
mean_mAP_box_50      0.9382090762045583
 
mean_mAP_box_75      0.8732667026953097 

mean_mAP_box_85      0.7663650279208901

### c26dm
mean_mAP_box_50      0.9495542949998398
 
mean_mAP_box_75      0.8955744642669738 

mean_mAP_box_85      0.7976094003321994 

mean_mAP_mask_50     0.9377138483917179 

mean_mAP_mask_75     0.8498109130360899 

mean_mAP_mask_85     0.7408607959795617 

mean_mAP_range_mask  0.7828245993307326