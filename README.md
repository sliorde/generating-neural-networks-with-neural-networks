# generating-neural-networks-with-neural-networks

This repository contains a TensorFlow implementation of a hypternetwork, as described [here](http://arxiv.org/abs/1801.01952).
Also contained in the repo is the code required to reproduce results from the paper.

The hypernetwork model is implemented in `hypernetwork.py`. To train it, run `train.py`. It uses the hyper-parameters defined in `params.py`, which are the same parameters used in the paper.

The folder `MNF` has an implementation of [MNFG](https://arxiv.org/abs/1703.01961) . The implementation was cloned from [here](https://github.com/AMLab-Amsterdam/MNF_VBNN), but it was  modified so that it can be used in my experiments. I regret to say that my modifications are very messy, which might make it hard to use this code. I suggest that you go to the original repo, use that code to train the network, save the checkpoints in some folder, and then use my code to run experiments with these checkpoints.

The folder `analysis` has scripts required to run the experiments described in the paper. The scripts whose file name ends with `_data_creator` are used to create data, which can then be displayed using the other scripts, whose file names end with `_display` . Each script mentions in its body which data it requires. The name of these scripts (before the `_display` suffix) matches the title of the subsection in the paper in which the corresponding experiment was described.

Finally, the folder `toy_model` has the toy example from the paper. To train the model, run `toy_example.py`. 



