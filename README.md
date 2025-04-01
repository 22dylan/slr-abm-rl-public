# slr-abm-rl-public
 Galveston sea level rise (SLR), agent-based model (ABM), and reinforcement learning (RL) repository. Public version used for documenting this code. 

*Note*: This repository requires input data that is available at 
https://doi.org/10.5281/zenodo.12583277. After the dataset above is downloaded, add it to the main directory in a folder named `slr-scenarios`.

This repository contains steps 2 (train agents) and 3 (agent-based model) in the flowchart below. Step 1 of the flowchart is available for download at https://doi.org/10.5281/zenodo.11402964 and produces the data above that is necessary to run this repository. 

![alt text](https://github.com/22dylan/slr-abm-rl-public/blob/main/figures/flowchart.png?raw=true)


To run step (2), run the julia code `train_agents.jl`. Input options for this code can be modified in `InputStructsTrain.jl` and `agent-weights.csv`. To run step (3), run the julia code `run.jl`. Input options for this code can be modified in `InputStructs.jl`. Additional code is used for both training agents and running the ABM is available in the `model-backend` directory. 