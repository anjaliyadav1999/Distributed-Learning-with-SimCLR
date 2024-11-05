# Distributed-Learning-with-SimCLR

Libraries that are required to install to run the code successfully is given in requirements.txt
Need to have CUDA toolkit installed in the systems which are being used to run the code.

!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

The pytorch should be installed with cuda version

To enable distributed training of model between devices or multiple gpus we use **DistributedDataParallel** a builtin pytorch function

_**from torch.nn.parallel import DistributedDataParallel**_

https://pytorch.org/tutorials/intermediate/ddp_tutorial.html - use this link to learn about DDP in pytorch

The model file used is directly a pytorch github model file 

To run the model in distributed environment use the command
**torchrun main.py**

https://pytorch.org/docs/stable/elastic/run.html  -- this link will make us understand how to use torchrun with arguments passing and node instances
