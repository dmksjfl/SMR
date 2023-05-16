# How to use

Reproduce our results on OpenAI Gym in the submission by calling:

```
python main.py --policy SAC --env HalfCheetah-v2 --seed 4 --dir ./logs/SAC-SMR-10/HalfCheetah/r4
```

Reproduce our results on DMC suite in the submission by calling:

```
python main_dmc.py --policy SAC --env cheetah-run --seed 4 --dir ./logs/SAC-SMR-10/cheetah-run/r4
```

For results of TQC and DrQ-v2, please refer to their official implementation. It is very simple to incorporate SMR with them.