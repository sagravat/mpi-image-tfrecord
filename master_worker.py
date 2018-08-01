from __future__ import division
from __future__ import print_function
import random

from socket import gethostname
import tensorflow as tf
import os
from dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset, _write_dataset, write_tfrecords 

#from dataset_utils import get_cloud_filenames_and_classes

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

WORKTAG = 0
DIETAG = 1
 


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

    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()
    size = MPI.COMM_WORLD.Get_size() 
    
    #Get a list of photo_filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
    photo_filenames, class_names = _get_filenames_and_classes(FLAGS.dataset_dir)

    #Refer each of the class name to a specific integer number for predictions later
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    if rank == 0:
        all_dat = master()
                    
    else:
        #config = tf.ConfigProto()
        #config.intra_op_parallelism_threads = 8
        #config.inter_op_parallelism_threads = 4
        #worker_sess = tf.Session(config=config)
        worker(class_names_to_ids)


class Work():
    def __init__(self, work_items):
        self.work_items = []
        for key in work_items:
            self.work_items.append(work_items[key])
 
    def get_next_item(self):
        if len(self.work_items) == 0:
            return None
        return self.work_items.pop()
 
def master():

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/agravat/key.json'
    #Get a list of photo_filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
    photo_filenames, class_names = _get_filenames_and_classes(FLAGS.dataset_dir)
    """
    cloud_filenames, cloud_class_names = get_cloud_filenames_and_classes("gs://agravat-demo/images")
    
    for f in cloud_filenames:
        print(f)
    """

    
    #Refer each of the class name to a specific integer number for predictions later
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    #Find the number of validation examples we need
    num_validation = int(FLAGS.validation_size * len(photo_filenames))

    # Divide the training datasets into train and test:
    random.seed(FLAGS.random_seed)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]

    # First, convert the training and validation sets.
    train_file_mappings = _convert_dataset('train', training_filenames, class_names_to_ids,
                 dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards)
    val_file_mappings = _convert_dataset('validation', validation_filenames, class_names_to_ids,
                 dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards)



    file_mappings = train_file_mappings
    file_mappings.update(val_file_mappings)
    rank_files = []
    outfile = "out.tfrecord"
    """
    for rank,o,file in train_file_mappings:
        if comm.rank == rank:
            rank_files.append(file)
            outfile = o
            #print("rank %d len files = %d, outfile = %s\n" % (comm.rank, len(rank_files), outfile))

    """
    #print("rank: %d, %s, outfile = %s" % (comm.rank, gethostname(), outfile))

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, FLAGS.dataset_dir)

    #print('Finished converting the %s dataset for %s rank %d' % (FLAGS.tfrecord_filename, gethostname(), comm.rank))
    all_data = []
    size = MPI.COMM_WORLD.Get_size()
    current_work = Work(train_file_mappings) 
    comm = MPI.COMM_WORLD
    status = MPI.Status()

    # this is the loop where the master distributes all the work based on the number of workers
    # that are available
    for i in range(1, size): 
        # the master gets the next element in the list
        anext = current_work.get_next_item() 
        if not anext: break
        # master sends the element to a worker
        comm.send(obj=anext, dest=i, tag=WORKTAG)
 
    # this is a fallback if there are more work items than workers
    while 1:
        # get the next work item but we break if there are None
        anext = current_work.get_next_item()
        if not anext: break

        # get the result from any worker
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        #print("more work rank %d, host %s data %s" % (comm.rank, gethostname(), data))
        # add the processed result to the list of results
        all_data.append(data)
        # send another work item to the worker who completed the last task
        print("spillover %d %s" % (comm.rank, gethostname() ))
        comm.send(obj=anext, dest=status.Get_source(), tag=WORKTAG)
 
    
    # get the results back from the workers
    for i in range(1,size):
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        print("recieved from %d" %(i))
        all_data.append(data)
    
    
    # end the tasks
    for i in range(1,size):
        comm.send(obj=None, dest=i, tag=DIETAG)
     
    return all_data
        
    
def worker(class_names_to_ids):
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    while 1:
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag(): break
        print("%d %s %s %s" % (comm.rank, gethostname(), data["outfile"], len(data["files"])))
        result = _write_dataset(data["files"], data["outfile"], class_names_to_ids)
        """
        with tf.python_io.TFRecordWriter(result["outfile"]) as tfrecord_writer:
            for example in result["images"]:
                # Read the filename:
                tfrecord_writer.write(example)
        """
        #write_tfrecords(1, data["files"], data["outfile"])
        # send the filename to the master
        comm.send(obj=None, dest=0)

if __name__ == "__main__":
    main()
