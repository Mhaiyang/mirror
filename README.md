# Mirror Project


## Change Log
2018/6/4    train:593  val:51  test:50

2018/5/9    test model on server, begin to make presentation.

2018/5/8    training on the server. **But there still have an error when validaition batch size is two. one is ok.**
            Saving new image _tf1.7.0-keras2.1.6-gpu:9.0-cudnn7.0.5-devel-ubuntu16.04_

2018/5/7    test_mirror.py is ok, add author information template.

2018/5/6    add move.py, transform label.png to label8.png, mirror.py is ok, train network is ok.

2018/5/5    Create project.

## Environment
python3.5    

My Computer: tf 1.3.0 keras 2.1.3

Server:

tensorflow1.7.0    keras2.1.6   numpy1.14.3

CPU : Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz

GPU : NVIDIA GTX 1080Ti

Docker Image : _tf1.7.0-keras2.1.6-gpu:9.0-cudnn7.0.5-devel-ubuntu16.04_

'sudo pip3 install -r requirements.txt'

#### labelme tool
*python2*

sudo apt-get install python-pyqt5  

sudo pip install labelme

Modify app.py:

```
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
```

## Usage
`python train_mirror.py`

## TODO
- [x] Collecting images containing mirrors
- [x] Data preprocessing : Rename --> Resize --> label mirror --> move.py --> json_to_dataset --> transform(16 to 8)(label8.png and info.yaml)
- [x] mirror.py (what the occlusion?)
- [x] train_mirror.py (Training head and all.)
- [x] Using tensorboard to keep watch on training
- [x] test_mirror.py (Including save test results.)
- [ ] After a long pause, restart the project at June 29th.
- [ ] Adding evaluation code.(AP, mAP, AP-box, etc.)
- [ ] Data augmentation.(flipping(left-right and top-down), rotation(90, 180 and 270))
- [ ] Determine Baseline (Refined network based on our dataset) and Begin to improve the architecture of network.(Such as PANet.)

## Loss
![loss](assets/loss.png)

## License
* For academic and non-commercial use only.
* For commercial use, please contact [mhy845879017@gmail.com](https://www.google.com/gmail/).