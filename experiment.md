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
    
实验：

- odel:原始Mask RCNN

- fusion：roi的多级特征和整张图的特征

- decoder：mask branch是反编码器

- fusion_decoder:fusion和decoder的结合

- fusion_context_guided_decoder:用全连接层引导反编码，跟Rynson汇报的版本

- p1:在fusion_context_guided_decoder的版本上使用了P1，效果反而降低了

- path_full:在fusion_context_guided_decoder的基础上使用了PANet的path augmentation，full就是指fusion_context_guided_decoder，提升的比较大

- post_relu:在path_full的基础上，在decoder部分，先进行卷积和反卷积，然后再relu激活。效果没path_full好。可能是非线性降低了，原来两个relu，现在只有一个了。

- attention：在path_full的基础上，加入了attention module。fusion和decoder部分都做了调整，包括特征层数和卷积核，去掉了一些卷积层

- attention2:在attention的基础上，在decoder部分增加了几个卷积层

- attention3:在attention2的基础上，将attention的maxpooling换成转置卷积，decoder部分直接复制过来。


# Mid-Autumn Festival
- edge: only use low-level features to perform mirror detection.

- content: using mid-level features to perform mirror detection.

- context: using high-level features to perform mirror detection.

- mod: mask-rcnn only detection. (no mask branch)(baseline). Used path augmentation.

- ad: aggregation for detection and decoder for mask prediction.

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

mean_mAP_box         0.8647127678839555
 
mean_mAP_mask        0.7521609940706835 

mean_mAP_range_mask  0.6975576265240376

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

mean_mAP_box         0.882203313534955 

mean_mAP_mask        0.8155501530838296 

mean_mAP_range_mask  0.7418377453768399

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

**full_path**

mean_mAP_box_50      0.9469656041993894 

mean_mAP_box_75      0.9021699982099054 

mean_mAP_mask_50     0.9408878084133278 

mean_mAP_mask_75     0.8560237709507306 

mean_mAP_range_mask  0.7995655501689177

**post_relu**

mean_mAP_box_50      0.941270484449669 

mean_mAP_box_75      0.8961372231387371 

mean_mAP_mask_50     0.9328291014094471 

mean_mAP_mask_75     0.8304069872275357 

mean_mAP_range_mask  0.7792004322133131

**attention**

mean_mAP_box_50      0.9476184044928888 

mean_mAP_box_75      0.8936160634025763 

mean_mAP_mask_50     0.9427111471571966 

mean_mAP_mask_75     0.8546056186006495 

mean_mAP_range_mask  0.8001035476515147

**attention2**

mean_mAP_box_50      0.9456825139832543 

mean_mAP_box_75      0.8902169998319925 

mean_mAP_mask_50     0.9341347019924208 

mean_mAP_mask_75     0.8358544930855948 

mean_mAP_range_mask  0.7901584729174047

mean_mAP_box_50      0.9455924725642016 

mean_mAP_box_75      0.8918152350201792 

mean_mAP_mask_50     0.9348325229847129 

mean_mAP_mask_75     0.8459616423715842 

mean_mAP_range_mask  0.7970939132175289

**attention3**

0.0649  35.h

mean_mAP_box_50      0.9538312624075296 

mean_mAP_box_75      0.895889609230975 

mean_mAP_mask_50     0.9441968305715672 

mean_mAP_mask_75     0.8542679632792017 

mean_mAP_range_mask  0.8090514136698815

0.0526 45.h

mean_mAP_box_50      0.947280749171441 

mean_mAP_box_75      0.894516477587737 

mean_mAP_mask_50     0.939807311392745 

mean_mAP_mask_75     0.8583873582008656 

mean_mAP_range_mask  0.8109265262209836

40.h

mean_mAP_box_50      0.9495092742929969 

mean_mAP_box_75      0.8952593192949223 

mean_mAP_mask_50     0.9403925806139045 

mean_mAP_mask_75     0.8538627768934642 

mean_mAP_range_mask  0.8087925445886296



# edge
mean_mAP_box_50      0.910858995159231 

mean_mAP_box_75      0.8636547812114274 

mean_mAP_box_85      0.7596119214865662

# content
mean_mAP_box_50      0.9269989195244389 

mean_mAP_box_75      0.8709031154411496 

mean_mAP_box_85      0.7824374212164418

# context
mean_mAP_box_50      0.9240950837626707
 
mean_mAP_box_75      0.8649153610795076 

mean_mAP_box_85      0.7579911759436163

# aggregation
mean_mAP_box_50      0.9437466234669115
 
mean_mAP_box_75      0.8789843328011353 

mean_mAP_box_85      0.7610750945461735

# mod
mean_mAP_box_50      0.9212362687010368 

mean_mAP_box_75      0.8502611201206199 

mean_mAP_box_85      0.7293580046848372

# ad 0.0634
mean_mAP_box_50      0.9448046101448064 

mean_mAP_box_75      0.8958896092336585 

mean_mAP_mask_50     0.9293399964278611 

mean_mAP_mask_75     0.8388258599116527 

mean_mAP_range_mask  0.7924567801364439

# ad 40.h
mean_mAP_box_50      0.9429812714063046
 
mean_mAP_box_75      0.8901269584089148 

mean_mAP_mask_50     0.9285071132908893 

mean_mAP_mask_75     0.8379929767854145 

mean_mAP_range_mask  0.7928597154849618

# ad 0.05358
mean_mAP_box_50      0.9407302359326688
 
mean_mAP_box_75      0.8922204214045749 

mean_mAP_mask_50     0.9243652080158038 

mean_mAP_mask_75     0.8350216099566733 

mean_mAP_range_mask  0.7900391680338044

