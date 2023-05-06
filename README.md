[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/chatty-coupled-holistic-adversarial-transport/unsupervised-domain-adaptation-on-fhist)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-fhist?p=chatty-coupled-holistic-adversarial-transport)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/chatty-coupled-holistic-adversarial-transport/unsupervised-domain-adaptation-on-office-31)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-office-31?p=chatty-coupled-holistic-adversarial-transport)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/chatty-coupled-holistic-adversarial-transport/unsupervised-domain-adaptation-on-office-home)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-office-home?p=chatty-coupled-holistic-adversarial-transport)
# CHATTY implemeneted in PyTorch

## Prerequisites
- pytorch = 1.0.1 
- torchvision = 0.2.1
- numpy = 1.17.2
- pillow = 6.2.0
- python3.6
- cuda10

## Training
The following are the commands for each task. Here, wt represents the parameter for weight of the transfer loss. 

Office-31
```
python train.py --gpu_id 0 --dset office --s_dset_path data/office/amazon_list.txt --t_dset_path data/office/dslr_list.txt --output_dir chatty/adn --wt 0.001 --domains A_to_D
```

Office-Home
```
python train.py --gpu_id 0 --dset office-home --s_dset_path data/office-home/Art.txt --t_dset_path data/office-home/Clipart.txt --output_dir chatty/ArCl --wt 0.0001 --domains Ar_to_Cl
```

Fhist
```
python train.py --gpu_id 0 --dset fhist --s_dset_path data/fhist/labeled_source.txt --t_dset_path data/fhist/unlabeled_target.txt --output_dir chatty/CrcNct --wt 0.0001 --domains CRC_to_NCT
```
The codes are heavily borrowed from [GVB](https://github.com/cuishuhao/GVB)
