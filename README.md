# FCN-resnet101

This project uses resnet101 to extract features and do semantic segmentation.
Program is written by tensorflow

## TODO
- [x] Use resnet101 pretrained model
- [x] Input can be in any size(just in the test and eval task)
- [x] Train on the PASCAL VOC2012 train data 
- [x] Evaluate in the PASCAL VOC2012 validate data

## REQUIREMENTS
- Tensorflow 1.1
- Python 2.7.13 (I use anaconda2-4.3.1)
- Pascal VOC2012 dataset

## Train
1. Put the tfrecord file into the ./data/  you can download from  https://www.dropbox.com/s/rm46xxxswho9i8z/pascal_voc_segmentation.tfrecords?dl=0 (converted from the PASCAL VOC2012 train set)
2. Put resnet101 pretrained model into ./data/pretrained_model/   you can download from https://www.dropbox.com/s/ehkniglsvbkotc9/resnet_v1_101.ckpt?dl=0
3. Run train.py
4. During the training, some test pictures will be generated in ./data/demo


## Test
1. You need to prepare the resnet101 pretrained model at first or you have trained the model.
2. Put your test pictures in the ./test/demo/ with the format ".jpg" or ".png"
3. Run test.py

## Eval
The eval pictures use PASCAL VOC2012 validation dataset ,so you can download them from the official website.
(but you need to convert the segmentation pictures into the indexed pictures.) 
or you can just download from https://www.dropbox.com/s/7n0sr0m3b9u1ua5/VOC2012_val.zip?dl=0
1. Put the JPEGImage folder into ./eval/VOC2012_val
2. Put the Segmentation folder into ./eval/VOC2012_val
3. Put the text.txt into ./eval/VOC2012_val
4. Run eval.py


## Results
Classes | IoU Accuracy(%)
------------ | -------------
aeroplane | 0
bicycle | 0
bird | 0
boat | 0
bottle | 0
bus | 0
car | 0
cat | 0
chair | 0
cow | 0
diningtable | 0
dog | 0
horse | 0
motorbike | 0
person | 0
potted-plant | 0
sheep | 0
sofa | 0
train | 0
tv/monitor | 0
mean accuracy | 0

## Tensorboard


