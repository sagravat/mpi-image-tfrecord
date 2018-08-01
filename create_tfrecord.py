from __future__ import division
from __future__ import print_function
import random

from socket import gethostname
import tensorflow as tf
import os
from dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset, _write_dataset

import numpy as np
from mpi4py import MPI                                                                                                                                                            


def pprint(str="", end="\n", comm=MPI.COMM_WORLD):
    """Print for MPI parallel programs: Only rank 0 prints *str*."""
    if comm.rank == 0:
        print(str+end, end=' ') 

comm = MPI.COMM_WORLD

pprint("-"*78)
pprint(" Running on %d cores" % comm.size)
pprint("-"*78)




#####k

#====================================================DEFINE YOUR ARGUMENTS=======================================================================
flags = tf.app.flags

#State your dataset directory
flags.DEFINE_string('dataset_dir', None, 'String: Your dataset directory')

# The number of images in the validation set. You would have to know the total number of examples in advance. This is essentially your evaluation dataset.
flags.DEFINE_float('validation_size', 0.3, 'Float: The proportion of examples in the dataset to be used for validation')

# The number of shards to split the dataset into
flags.DEFINE_integer('num_shards', 4, 'Int: Number of shards to split the TFRecord files')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', None, 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():

    #==============================================================CHECKS==========================================================================
    #Check if there is a tfrecord_filename entered
    if not FLAGS.tfrecord_filename:
        raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')
   
    #Check if there is a dataset directory entered
    if not FLAGS.dataset_dir:
        raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')
  
    #If the TFRecord files already exist in the directory, then exit without creating the files again
    if _dataset_exists(dataset_dir = FLAGS.dataset_dir, _NUM_SHARDS = FLAGS.num_shards, output_filename = FLAGS.tfrecord_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return None
    #==============================================================END OF CHECKS===================================================================

    #Get a list of photo_filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
    photo_filenames, class_names = _get_filenames_and_classes(FLAGS.dataset_dir)

    #Refer each of the class name to a specific integer number for predictions later
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    #Find the number of validation examples we need
    num_validation = int(FLAGS.validation_size * len(photo_filenames))

    # Divide the training datasets into train and test:
    random.seed(FLAGS.random_seed)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]

    if comm.rank == 0:
        # First, convert the training and validation sets.
        train_file_mappings = _convert_dataset('train', training_filenames, class_names_to_ids,
                     dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards)
        val_file_mappings = _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards)

        #file_mappings.extend(val_file_mappings)
    else:
        train_file_mappings = None
        val_file_mappings = None


    train_file_mappings = comm.bcast(train_file_mappings, root=0)
    rank_files = []
    outfile = None
    for rank,o,file in train_file_mappings:
        if comm.rank == rank:
            rank_files.append(file)
            outfile = o
            #print("rank %d len files = %d, outfile = %s\n" % (comm.rank, len(rank_files), outfile))

    print("rank: %d, %s, outfile = %s" % (comm.rank, gethostname(), outfile))

    # write training files
    if outfile != None:
    	_write_dataset(rank_files, outfile, class_names_to_ids)

    val_file_mappings = comm.bcast(val_file_mappings, root=0)
    rank_files = []
    outfile = None
    for rank,o,file in val_file_mappings:
	#if comm.rank == 29:
	    #print("rank %d, shards = %d, comm rank = %d, mod = %d" % (rank, FLAGS.num_shards, comm.rank, rank % FLAGS.num_shards))
        if rank % FLAGS.num_shards == comm.rank:
            rank_files.append(file)
            outfile = o
            #print("rannk: %d, file = %s" % (
        #pprint("rank: %s, %d, %s, %s" % (comm.rank, a,b,c))

    print("val rank: %d, %s, outfile = %s" % (comm.rank, gethostname(), outfile))
    # write validation files
    if outfile != None:
    	_write_dataset(rank_files, outfile, class_names_to_ids)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, FLAGS.dataset_dir)

    print('Finished converting the %s dataset for %s rank %d' % (FLAGS.tfrecord_filename, gethostname(), comm.rank))

if __name__ == "__main__":
    main()
