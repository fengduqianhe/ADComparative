## ADComparative
Early Alzheimer's Disease Diagnosis with Paired Comparative Deep Learning
The code was written by Hezhe Qiao, Lin Chen Chongqing Institute of Green and Intelligent Technology, Chinese Academy of Sciences, 400714 Chongqing, China, University of
Chinese Academy of Sciences, 100049 BeiJing, China.

##Introduction
we propose a paired comparative deep learning method that measures the
differences of group category (G-CAT) and subject mini-mental state examination (S-MMSE), respectively,
to enhance the sMRI features of groups and individuals. This proposed model has been evaluated on
the ADNI-1, ADNI-2, and MIRIAD datasets.
Dataset

The dataset used in this study was obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) that is avaiable
at `http://adni.loni.usc.edu/` .
##Pre-process
All MRIs were prepocessed by a standard pipeline in CAT12 toolbox
which is avaiable at `http://dbm,neuro.uni-jena.de/cat/`.

The MIRIAD is also a database of sMRI brain volume scans of
Alzheimer’s disease patients and healthy elderly people.

##Prerequisites
Linux python 3.7
Pytorch version 1.2.0
NVIDIA GPU + CUDA CuDNN (CPU mode, untested) Cuda version 10.0.61

## Note
Please cite our paper if you use this code in your own work.
 Qiao H ,  Chen L ,  Ye Z , et al. Early Alzheimer's Disease diagnosis with the contrastive loss using paired structural MRIs[J]. Computer Methods and Programs in Biomedicine, 2021(1):106282.


