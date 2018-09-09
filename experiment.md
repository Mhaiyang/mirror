检测最小置信概率0.7
1. Mask R-CNN原始实现 10小时

    Heads : mean_mAP 0.8920863309352518 mean_mAP_range 0.7190647482014391

    All: mean_mAP 0.9406474820143885 mean_mAP_range 0.7733812949640293

2. 在1的基础上将build_fpn_mask_graph中4个3x3的卷积核都换成了5x5，训练了10小时39分钟，训练损失0.4278，验证损失2.397
    Heads:
    mean_mAP             0.8758992805755396 
    mean mAP range       0.6906474820143886
    All:
    mean_mAP             0.9316546762589928 
    mean mAP range       0.756834532374101
3. mask分支改为conv-->deconv2-->deconv3-->deconv4--.deconv5-->conv, 输入尺寸14x14，输出尺寸224x224，训练12小时2分钟，训练损失0.6067，验证损失2.395
    Heads:
    mean bbox recall     0.9388489208633094 
    mean_mAP             0.8902877697841727 
    mean mAP range       0.5253597122302156
    All:
    mean bbox recall     0.9316546762589928 
    mean_mAP             0.9190647482014388 
    mean mAP range       0.6375899280575538
    Augmented test set ALL:
    mean_mAP_box         0.9064748201438849 
    mean_mAP_mask        0.8896882494004796 
    mean_mAP_range_mask  0.5994604316546769
    

9.1

Augmentation test

**false decoder**:

mean_mAP_box         0.9104763191336236 

mean_mAP_mask        0.8969926166237636 

mean_mAP_range_mask  0.6890442103499147

**mask rcnn**

mean_mAP_box         0.9270889609434917 

mean_mAP_mask        0.9090356564220707 

mean_mAP_range_mask  0.7329078876424174

mean_mAP_box         0.8582973167737625 

mean_mAP_mask        0.8099675851052417 

mean_mAP_range_mask  0.7321110210838009

**correct decoder:**

mean_mAP_box         0.9312083558651557 

mean_mAP_mask        0.912929947800128 

mean_mAP_range_mask  0.6968147848168522

**fusion**

mean_mAP_box         0.9394246353523786 

mean_mAP_mask        0.9232396902763026 

mean_mAP_range_mask  0.7429497569026812

mean_mAP_box         0.8785341257045296 

mean_mAP_mask        0.8257698541490018 

mean_mAP_range_mask  0.7429497569026812

**fusion decoder**

mean_mAP_box         0.9330091842462109 

mean_mAP_mask        0.9204484062870086
 
mean_mAP_range_mask  0.7416283990775417

**fusion context guided decoder false**

mean_mAP_box         0.9365883306535584 

mean_mAP_mask        0.9267513056193604 

mean_mAP_range_mask  0.7601431658718711

**fusion context guided decoder right**

mean_mAP_box         0.9365883306535584 

mean_mAP_mask        0.9293850171266538 

mean_mAP_range_mask  0.7718935710582571

mean_mAP_box         0.8825859895659292 

mean_mAP_mask        0.828741220977743 

mean_mAP_range_mask  0.7718935710582571

**p1**

mean_mAP_box         0.9380740140706124 

mean_mAP_mask        0.927201512715966 

mean_mAP_range_mask  0.7643863677435231