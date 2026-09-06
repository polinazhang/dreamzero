For how to launch normal inference, see

/coc/testnvme/xzhang3205/vla-adaptation/inference/run_one_dreamzero.py
/coc/testnvme/xzhang3205/vla-adaptation/inference/run_dreamzero.sbatch 

This is for your reference when writing static inference code. For static inference you should either use model `/storage/home/hcoda1/5/xzhang3205/scratch/vla-adaptation/models/dreamzero/checkpoints/DreamZero-DROID` or model `/storage/home/hcoda1/5/xzhang3205/scratch/vla-adaptation/models/dreamzero/checkpoints/DreamZero-AgiBot` and its statistic.json. You must not recompute statistics.json -- use that