# Pytorch SSANet
Scale Space Approximated Network for Vessel Segmentation

![](https://img.shields.io/static/v1?label=python&message=3.6|3.7&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=1.4&color=<COLOR>)

- Paper SSANet: https://doi.org/10.1016/j.cmpb.2019.06.030

- [x] Inference
- [x] Train

![image](https://drive.google.com/uc?export=view&id=1VyUkwe66ANMh8d80cy-4u8qM8UId7FDs)
![image](https://drive.google.com/uc?export=view&id=1_HzoUS3oaDol6_v3wKXslMS5CNTwaSsB)

# 0. DB Download

## 0.1 DRIVE
- google(https://drive.google.com/file/d/1GaShB-cwo3HDOHQpbcfyothFqxzWQSJz/view?usp=sharing)

## 0.2 STARE
- google(https://drive.google.com/file/d/1nynd4PxD8X5gvqQ3jnFHdPz7_LgOr8Hs/view?usp=sharing)

## 0.3 CHASE_DB
- google(https://drive.google.com/file/d/1fGgiYwFr3E_sQ2RrLo7YQt_F06a2tryb/view?usp=sharing)

## 0.4 HRF
- google(https://drive.google.com/file/d/1CzccqPIVTxUWq4PI7GntuMRa-aEyxXbC/view?usp=sharing)


# 1. Weights Download

## 0.1 DRIVE
- google(https://drive.google.com/file/d/1jCbcvVSih7U9rdcCELIgDH2PTAp8P1Gz/view?usp=sharing)

## 0.2 STARE
- google(https://drive.google.com/file/d/1rI6FfhqZ7oyZYlBGukuhe96biwGMQWOp/view?usp=sharing)

## 0.3 CHASE_DB
- google(https://drive.google.com/file/d/16GUHbAKXdh3K7PNDORT0LSSIkEyObUio/view?usp=sharing)

## 0.4 HRF
- google(https://drive.google.com/file/d/12bXAUBBvK7xIRjO6fAUzvNev_m6Aj8vf/view?usp=sharing)


# 3. Result Images Download

## 0.1 DRIVE
- google(https://drive.google.com/file/d/1kZctKOKBjBRSJCsN7fKGqKfEpLirb_UD/view?usp=sharing)

## 0.2 STARE
- google(https://drive.google.com/file/d/1JY8u7v03mH54D82HGTjQVE-Meukzw17q/view?usp=sharing)

## 0.3 CHASE_DB
- google(https://drive.google.com/file/d/1ZOX_UuI74Ue1cJbtjf_yUKw79S5iRGzh/view?usp=sharing)

## 0.4 HRF
- google(https://drive.google.com/file/d/1AnCq3WARHkZYc14rpV173ow7r2aTOLFH/view?usp=sharing)


# 2. Train

1. Train
    
    ex) DRIVE DB learning on default settings
    ```
     python main.py --DB 0 ...
     
    ```

# 2. Inference

1.
    ex) DRIVE DB inference
    ```
     python main.py --DB 0 --eval_only True --model_weight [model/weight/path] ...
     
    ```
   
Reference:
- 

```
@article{SSANet,
  title={Scale-space approximated convolutional neural networks for retinal vessel segmentation},
  author={Kyoung Jin Noh, Sang Jun Park, Soochahn Lee},
  journal = {Computer Methods and Programs in Biomedicine},
  year={2019}
}
```
