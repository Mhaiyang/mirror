#**** log_123
1:decoder_edge

2:context

3:decoder_mask

统一：头部训练15个epoch， 学习率1e-3；全部再训练5个epoch， 学习率1e-4.

Baseline: mod, mask rcnn only detection

实验一：context-n2-n5  **c25**

实验二：context-n2-n6  **c26**

实验三：context-n2-n6 + decoder_mask  **c26dm**

实验四：context-n2-n6 + decoder_edge  **c26de**
