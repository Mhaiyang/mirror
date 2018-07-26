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
    