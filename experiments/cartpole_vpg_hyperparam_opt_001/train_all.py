import json

from experiment import training
with open("hyperparams.json") as f:
    hyperparams = json.load(f)

lr_vals = hyperparams['lr']
numseed = hyperparams['numseed']
discount_vals = hyperparams['discount']
batch_size_vals = hyperparams['batch_size']

for lr in lr_vals:
    for discount in discount_vals:
        for batch_size in batch_size_vals:
            config = { 'lr':lr, 
                        'discount': discount, 
                        'batch_size': batch_size}
            for __ in range(numseed):
                training.run(config_updates=config)
            