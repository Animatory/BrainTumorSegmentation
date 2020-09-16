## Brain Tumor Classification and Segmentation Project  
***Project Idea***: We have a dataset with 3 kinds of brain tumors. The idea is first to classify the type of brain tumor by the image and then to find the borders of the tumor on each image. Vanilla UNet architecture with ‘efficientnet b0’ encoder will be used as a baseline model. Classification and segmentation and done simulteniouslly in one model. We have also used EMA and attention to try to improve the results but it has not given a significant improvement.    

***Project Authors***  Vyacheslav Shults (v.shults@innopolis.ru), Polina Turishcheva (p.turishcheva@innopolis.university)
  
***Dataset Explanation and link*** 
https://figshare.com/articles/brain_tumor_dataset/1512427   

It is a brain tumor dataset containing 3064 T1*-weighted contrast-enhanced images from 233 patients with three kinds of brain tumor: meningioma (708 slices), glioma (1426 slices), and pituitary tumor (930 slices). No healthy people, the tumor distribution is more or less balanced. Each brain image has 4 attributes -raw data, a vector storing the coordinates of discrete points on tumor border, the label of the tumor and patient id.
  
*T1 (longitudinal relaxation time) is the time constant which determines the rate at which excited protons return to equilibrium. It is a measure of the time taken for spinning protons to realign with the external magnetic field.  

***Code Explanation***
This code is a light flexible framework based on pytorch.   
Mostly, you need to upload your own dataset to the framework (in our case it was done in `data/datasets/tumor_dataset.py`) and define a task (`tasks/tumor_segmentation_task.py`) -  training_step,validation_step, test_step, and forward functions necessary for network training are defined there. All enchancements, like EMA optimizers were also defined there.  
Everything else - input size, losses, metrics, weights size, data preeprocessing(normalization), data augmentation, - is defined in configurations files (`configs/`, 1 file per model).  
To train a model, run `python train.py --hparams path_to_config.yml`.   
Model validation process may be found in `notebooks/Validate Segmentation.ipynb`.

***Results***
Open `notebooks/Validate Segmentation.ipynb` for more examples.  
We have great classification accuracy.

![](https://i.imgur.com/jaQ4naP.png)
EMA stands for EMA and attention.Normstands for normalization type and may be of two types: GN - global normalization, IN - instance normalization.Float is for  weights  and  input  data  type,Size is  for  input/output  image resolution.M for  meningioma,G for  glioma, P for  pituitary  tumor  F1 classification  scores. 


For segmentation we have low IoU score when

* only tumor boundaries are slightly visible but the texture seems to be the same as healthy
* the tumor is not consistent - it has 2 or 3 focuses, but on the training masks there is alway 1 continious spot, hence, the model fails to merge them
* there are several suspisious regions and the model makes a wrong choice
* the model detect the tumor correctly but it is to unconfident to step over the threshold
* the initial tumor is very small

The best IoU score are obtained when

* The tumor is rather contrast, big and its texture/color is more or less uniform, but it may contain some not uniform contaminations. It may not have sharp edges
* The tumor may have the color very close to some other healthy region in the same picture, but it should be assimetric in this case  
 

***Sample Segmentation Output***
![](https://i.imgur.com/QocmdhE.jpg)


  


