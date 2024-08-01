# MOCC: Multi-Objective Congestion Control

This repository contains the implementation to reproduce the results of the paper "Multi-Objective Congestion Control". This project leverages code from two other repositories:

1. [PCC-RL](https://github.com/PCCproject/PCC-RL)
2. [PCC-Uspace](https://github.com/PCCproject/PCC-Uspace)

## Dependencies
- Python 3.5
- Tensorflow 1.15


## Usage

### Training
To train the model, follow these steps:

1. go to ./PCC-RL/src/gym/, install any missing requirements.
2. run neighbor.py to get the objective training trajectory for fast traversing.
3. run run.py to generate the bash script for training.
4. run the bash script to start training.

### Testing

Start a server:
```
cd PCC-Uspace/src
export LD\_LIBRARY\_PATH=$LD\_LIBRARY\_PATH:`pwd`/core/
./app/pccserver recv 9000
```

Start the udt side of the environment:
```
cd PCC-Uspace/src
export LD\_LIBRARY\_PATH=$LD\_LIBRARY\_PATH:`pwd`/core/
./app/pccclient send 127.0.0.1 9000 --pcc-rate-control=python -pyhelper=loaded\_client -pypath=/path/to/pcc-rl/src/udt-plugins/testing/ --history-len=10 --pcc-utility-calc=linear --model-path=/path/to/your/model/ --weight_throughput=0.6 --weight_delay=0.3 --weight_loss=0.1
```

This should set the weights for MOCC as `0.6, 0.3, 0.1` and begin running the specified agent on the localhost connection. To run on a real world link, run the sender and receiver on different machines and adjust the IP addresses appropriately. You can find a pre-trained model at ./PCC-RL/src/gym/model/ .

## Contributing
We welcome contributions to improve this project. Please fork the repository, make your changes, and submit a pull request.
