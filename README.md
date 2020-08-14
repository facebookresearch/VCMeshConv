This is code corresponding to the paper titled "Fully Convolutional Mesh Autoencoder using Efficient Spatially Varying Kernels"

https://arxiv.org/pdf/2006.04325.pdf

It contains two folder.

"GraphSampling" is a C++ code for sampling on the mesh template to generate the connectivity structures for learning. Please use the cmake to create the MakeFile for compiling.Please look at the help in the function to know how to run it. It will take an template mesh and output the sampled connectivity file to the output path. 

"GraphAutoEncoder" is a python code which uses PyTorch to train an autoencoder on a sequence of registered meshes with the same topology with the template mesh used in "GraphSampling". It takes a configuration file as input (One example of configure file is named "config_train.config"). The main training file is "graphVAE_train.py". The proposed mesh conv operator is defined in "graphVAESSW.py".



