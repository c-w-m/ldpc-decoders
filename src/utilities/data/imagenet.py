import tensorflow_datasets as tfds
import tensorflow as tf
import os

from . import Dataset


# http://www.image-net.org/challenges/LSVRC/2012/downloads
# https://www.tensorflow.org/datasets/catalog/imagenet2012

def preproc_func(dd):
    dd['image'] = tf.image.resize_with_crop_or_pad(dd['image'], 227, 227)
    return dd

def make_dataset(ds_name, dataset_args):

    # solution to resource exhaustion: https://github.com/tensorflow/datasets/issues/1441
    # (name="imagenet_resized", builder_kwargs={'config':'64x64'}) # 16x16 32x32 64x64
    # config = tfds.download.DownloadConfig(manual_dir=MANUAL_DIR)
    # , download_and_prepare_kwargs={'download_config':config})
    splits, ds_info = tfds.load(ds_name, with_info=True)

    if 'imagenet_resized' in ds_name: preproc = lambda ds_:ds_
    else: preproc = lambda ds_:ds_.map(preproc_func, num_parallel_calls=16)

    dataset_args.update({'train_buffer':2**16, 'train_prefetch':64,
                         'test_buffer':2**16, 'test_prefetch':64})
    # fig = tfds.show_examples(ds_info, test_ds)
    return Dataset(preproc(splits['train']), preproc(splits['validation']),
                    train_num_examples=ds_info.splits['train'].num_examples,
                    **dataset_args)
