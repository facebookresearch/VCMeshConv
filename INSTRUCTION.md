## DFAUST

To test on DFAUST dataset:

1. Download the dataset from [here](https://dfaust.is.tue.mpg.de/downloads) and fallow the instruction to prepare directory with `.OBJ` files.
2. Build `GraphSampling` part of the repo. You can use such snippet to do it:
   ```
   $ cd cd GraphSampling/ && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release../ && make -j4
   ```
3. Take one mesh from the dataset and run sampling on it, for example:
   ```
   ./GraphSampling /<path_to_dataset>/00042.obj <path_to_output>
   ```
   It will produce you pre-computed pooling and unpooling files. You can also inspect produced `.OBJ` files, to see, how pooling is done. It will be shown by some vertices marked with red color. Native file manager on MacOS shows it perfectly.

   * On this step you may also change the structure of the pool/unpool layers in `GraphSampling/main.cpp`.

4. Change paths to configs in `GraphAutoEncoder/config_train.config`:
   ```
   mesh_train: <path to directory with prepared .OBJ files for training>
   connection_folder:  <path_to_sampling_output_with_pool.npy_files>
   initial_connection_file_names: <as_above_but_specify_path_to_first_pool_file>/_pool0.npy
   ```
5. Run training:
   ```
   cd GraphAutoEncoder && python graphVAE_train.py config_train.config
   ```
6. In another terminal run tensorboard:
   ```
   cd GraphAutoEncoder && tensorboard --logdir=testMC/train/log_00/
   ```
