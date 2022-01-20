~~# PoP-net
A novel popularity prediction network (PoP-Net), which consists of two branches for dealing with evolutional patterns of cascades and interactions between users respectively, for online content.~~
#TCSE-Net is a implemention of paper 'Trend and cascade based spatiotemporal evolution network to predict online content popularity', and is under reviewed by Multimedia Tools and Applications.

## Dependency Package

python 3.6

pytorch 1.7.1

## File Description

**data_preprocess** is used to process the raw data as the train data.

**PoPnet_model** contains the pytorch implementation of PoP-net. 

## Datasets
The datasets we used in our paper are Sina Weibo and Twitter. For the Sina Weibo dataset, you can download https://github.com/CaoQi92/DeepHawkes and the Twitter dataset is avilable https://github.com/majingCUHK/Rumor_RvNN.

##Steps to run PoP-net
1.process the raw data 
cd data_preprocess
python preprocess_graph_signal.py
#you can configure parameters and filepath in the file of "config.py"
2.trainsform the datasets to the format of ".pkl" command:
pyhton data_pretreat.py 


cd PoPnet_model
python main.py

# If you find this code useful, please let us know and cite our paper.
