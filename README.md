# InstantNet
## Goal
Given the target dataset and hardware platform, we search for the optimal network architecture and the mapping to hardware which work well under different bit-widths with both comparable or better accuracy and hardware efficiency as the optimal ones for each single bit-width.

(Current version doesn't include quantization, just full precision baselines)

## Core Files
*train_search.py / train.py*: main function for search / train the searched arch from scratch
*config_search.py / config_train.py*: configurations for search / train the searched arch from scratch
*model_search.py / model_infer.py*: supernet definition / searched arch definition
*other files* : utils for operation definition, learning rate schedule, datasets, etc. 

## Usage
### Overiew
Specify the search/train settings in *config_search.py / config_train.py* and search the optimal network through *train_search.py*. Then train the searched arch from scratch through *train.py*. (The best searched arch will be saved at 'ckpt/search/arch.pt', which is the default path for *train.py* to read.)

We search on ImageNet-100 (no distributed training support) and train the derived arch on ImageNet-1000 (support distribute training).

### Step by step
1. Specify the search setting in *config_search.py* (line 31~34):
```
C.dataset_path = "path to ImageNet-100"
C.batch_size = 192
C.num_workers = 16
C.flops_weight = 1e-9
```
`C.dataset_path`
