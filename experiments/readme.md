## RL experiments

### vpg_generalization_001

#### Question I'm trying to answer:
How well do my policies generalize in the time domain? For a fixed RL task, say I train with vanilla policy gradient for a max episode length of T. If my agent really learns a good 
strategy, when I let it evolve for longer episodes, it should survive for a proportionally longer time: reward \propto T. 

I guess I mostly just want to quantify how good _my_ training procedure is. 