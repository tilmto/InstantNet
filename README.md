# InstantNet
## Goal
Given the target dataset and hardware platform, we search for the optimal network architecture and the mapping to hardware which work well under different bit-widths with both comparable or better accuracy and hardware efficiency as the optimal ones for each single bit-width.

(Current version doesn't include quantization, just full precision baselines)

## Different versions of Code
***InstantNet_base***: full precision baseline (which is the last version ***InstantNet***)

***InstantNet***: search and retrain with multiple quantization bit-widths

## Core Files
***train_search.py / train.py*** : main function for search / train the searched arch from scratch

***config_search.py / config_train.py*** : configurations for search / train the searched arch from scratch

***model_search.py / model_infer.py*** : supernet definition / searched arch definition

***other files*** : utils for operation definition, learning rate schedule, datasets, etc. 

## Usage
### Overiew
Specify the search settings in ***config_search.py*** and search the optimal network through ***train_search.py***. Then specify training settings in ***config_train.py*** and train the searched arch from scratch through ***train.py*** (The best searched arch will be saved at ***ckpt/search/arch.pt***, which is the default path for ***train.py*** to read.)

We search on ImageNet-100 (no distributed training support) and train the derived arch on ImageNet-1000 (support distributed training).

### Step by step
1. Specify the search setting in ***config_search.py*** (line 31~34):
```
C.dataset_path = "path-to-ImageNet-100"
C.batch_size = 192
C.num_workers = 16
C.flops_weight = 1e-9
```
`C.dataset_path` is the dataset path to ImageNet-100. `C.num_workers` is the workers number for dataloader (config based on your server). `C.flops_weight` is the weight of FLOPs (Floating-point Operations) loss in the total loss which is a hyper-param to control the trade-off between accuracy and FLOPs. You can edit the ***config_search.py*** file or specify the args in command line like `--flops_weight 1e-9`. No need to change other settings.

2. Run ***train_search.py***: 
```
python train_search.py
```

3. Specify the training setting (distributed) in ***config_train.py*** (line 32~42):
```
C.dataset_path = "path-to-ImageNet-1000" # Specify path to ImageNet-1000

C.world_size = 1  # num of nodes
C.multiprocessing_distributed = True
C.rank = 0  # node rank
C.dist_backend = 'nccl'
C.dist_url = 'tcp://IP-of-Node:Free-Port' # url used to set up distributed training

C.num_workers = 4  # workers per gpu
C.batch_size = 256
```
`C.dataset_path` is the dataset path to ImageNet-1000. `C.rank` is the rank of the current node. `C.dist_url` is the IP of the first node. Note that `C.num_workers` is the workers assigned to each process, i.e., each gpu. You can edit the ***config_train.py*** file or specify the args in command line like `--rank 0`. No need to change other settings.

4. Run ***train.py*** on each of your nodes: 
```
python train.py
```

5. Get the searched arch ***ckpt/search/arch.pt***, the search logs ***ckpt/search/logs.txt*** and the training logs ***ckpt/finetune/logs.txt***.

## Update 2020/05/25
Add distributed search in the directory ***InstantNet_distributed_search***. Please configure distributed settings in the ***config_search.py*** (the same parameters as ***config_train.py*** for distributed training from scratch).

## First Round Exp 2020/05/21
Use ***InstantNet_base***: Search under 3 settings of `C.flops_weight`: 1e-7, 1e-9, 1e-11 in ***config_search.py***.

## Second Round Exp 2020/05/24
1. Use ***InstantNet_base***: Search under 1 settings of `C.flops_weight`: 1e-15 in ***config_search.py***.

2. Use ***InstantNet*** (with quantization): Search under 2 settings of `C.flops_weight`: 0, 1e-15 in ***config_search.py***. (The search speed may be much slower than 1. If it's unacceptable searching speed, I will upgrade the searching process to support distributed training.)

