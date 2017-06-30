# FCN-resnet101

This project uses resnet101 to extract features and do semantic segmentation.
Program used tensorflow.

## TODO
- [x] Use resnet101 pretrained model
- [x] Input can be in any size(just in the test and eval task)
- [x] Data augmentation(only horizontal flip)
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
1. You need to train the model first.
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
On the Pascal Voc Evaluation Server2012    

Classes | IoU Accuracy(%)
------------ | -------------
aeroplane | 39.89    
bicycle | 12.33    
bird | 17.44    
boat | 20.65    
bottle | 31.75    
bus | 53.09          
car | 42.74      
cat | 39.52      
chair | 4.50         
cow | 10.25         
diningtable | 13.39         
dog | 30.72         
horse | 21.71           
motorbike | 44.96         
person | 48.20         
potted-plant | 16.15             
sheep | 27.89              
sofa | 19.56        
train | 36.80               
tv/monitor | 30.94         
mean accuracy | 30.84       



