# Denstereo
This repository contains code for Deep Stereo RGB-only Dense 6D Object Pose Estimation.
Leveraging stereo, we extend the state-of-the-art in the task of direct 6D pose regression.

Training and Testing
----------------
Please directly run ./gdrn_denstereo_modeling/main_gdrn.py for training and testing.

Important parameters include
>> config-file : the path to the configuration file.

>> resume: if 'True', continue the training process from the last checkpoint.

>> eval-only: if 'True', directly evalute the model.

Related Citations
--------------

This work is based on follwing code bases and work:

```BibTeX
@misc{liu2022gdrnpp_bop,
  author =       {Xingyu Liu and Ruida Zhang and Chenyangguang Zhang and 
                  Bowen Fu and Jiwen Tang and Xiquan Liang and Jingyi Tang and 
                  Xiaotian Cheng and Yukang Zhang and Gu Wang and Xiangyang Ji},
  title =        {GDRNPP},
  howpublished = {\url{https://github.com/shanice-l/gdrnpp_bop2022}},
  year =         {2022}
}

@InProceedings{Wang_2021_GDRN,
    title     = {{GDR-Net}: Geometry-Guided Direct Regression Network for Monocular 6D Object Pose Estimation},
    author    = {Wang, Gu and Manhardt, Fabian and Tombari, Federico and Ji, Xiangyang},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {16611-16621}
}
```
