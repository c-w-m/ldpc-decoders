# [Utilities](https://github.com/thadikari/utilities)

Helper functions/common code used in other projects

## Download datasets

* Prepare environment:
  ```
  ln -s ~/SCRATCH ~/scratch
  mkdir ~/scratch/DATASETS
  ln -s ~/scratch/DATASETS ~/tensorflow_datasets
  ```

* Download datasets:
  ```
  python -c 'import tensorflow_datasets as tfds; tfds.load("mnist")'
  python -c 'import tensorflow_datasets as tfds; tfds.load("fashion_mnist")'
  python -c 'import tensorflow_datasets as tfds; tfds.load("cifar10")'

  python -c 'import tensorflow_datasets as tfds; tfds.load("imagenet_resized/8x8")'
  python -c 'import tensorflow_datasets as tfds; tfds.load("imagenet_resized/16x16")'
  python -c 'import tensorflow_datasets as tfds; tfds.load("imagenet_resized/32x32")'
  python -c 'import tensorflow_datasets as tfds; tfds.load("imagenet_resized/64x64")'

  python -c 'import tensorflow_datasets as tfds; tfds.load("imagenet2012")
  ```

* Some downloading and unpacking errors can be fixed by prefixing `CUDA_VISIBLE_DEVICES=""` to download commands.

* The last requires manually downloading original imagenet data to `tensorflow_datasets/downloads/manual` from http://www.image-net.org/challenges/LSVRC/2012/downloads as instructed in https://www.tensorflow.org/datasets/catalog/imagenet2012. 

* Submit a download job on Compute Canada (requires internet connectivity on compute nodes):

  `submitjob single -d nodes=1 mem=30G time=0-04:00 ntasks-per-node=1 cpus-per-task=4 -D email -e "CUDA_VISIBLE_DEVICES=\"\" python -c 'import tensorflow_datasets as tfds; tfds.load(\"imagenet_resized/32x32\")'"`
