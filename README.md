# Deep-Light-Field-Depth-Estimation
Learning Multi-modal Information for Robust Light Field Depth Estimation
## Introduction
we propose a multi-modal learning method for robust light field depth estimation.
Yongri Piao, Xinxin Ji, Miao Zhang and Yukun Zhang.
 
## Usage Instructions
Requirements
* Windows 10
* PyTorch 1.0.1
* CUDA 10.0
* Cudnn 7.6.5
* Python 3.6.5
* Numpy 1.14.3
* PIL

Train/Test

+ test

 Download related DUT-LFDD dataset [**link**](https://pan.baidu.com/s/1vOM5KUSAGfzwrZwcwzoO8g) [ fetch code **tcw4** ] and the checkpoint file [**link**]( https://pan.baidu.com/s/1OPoTreAQDO8lhP_G_Gschw) [ fetch code **d3sv** ], and you need to set **--test_root** and checkpoint root correctly in ```test.py```. Meanwhile, you can  to evaluate the results in ```eval.py```
 ```                 
 python test.py, eval.py   
 ```
+ Train

 Our train DUT-LFDD dataset [**link**](https://pan.baidu.com/s/1V8_wHSiZGVfDSXCLu5O2hw) [ fetch code **p4sj** ], and set the param '--test_root' correctly in ```train.py```. Meanwhile, you need to download the related file ```VGG16.pth``` [**link**](https://pan.baidu.com/s/1Nfymq7MJNZijpY8cZKFacg)[ fetch code **5xt9** ] and put it into the file ```model```.
```
python train.py
```

### Contact Us
If you have any questions, please contact us ( jxx0709@mail.dlut.edu.cn ).

