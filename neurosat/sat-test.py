import sys, os, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import build_neurosat
import instance_loader
import itertools
from logutil import test_with
from util import timestamp, memory_usage
from cnf import ensure_datasets

if __name__ == "__main__":
  print( "{timestamp}\t{memory}\tMaking sure ther datasets exits ...".format( timestamp = timestamp(), memory = memory_usage() ) )
  ensure_datasets( make_critical = True )
  if not os.path.isdir( "tmp" ):
    sys.exit(1)
  #end if
  d = 128
  batch_size = 64
  if 1 < len( sys.argv ):
    test_time_steps = int( sys.argv[1] ) # Use a much bigger number of time steps
  else:
    test_time_steps = 28
  #end if
  test_batch_size = batch_size
  
  # Build model
  print( "{timestamp}\t{memory}\tBuilding model testing with {time_steps} time_steps ...".format( timestamp = timestamp(), memory = memory_usage(), time_steps = test_time_steps ) )
  solver = build_neurosat( d )
  
  # Create model saver
  saver = tf.train.Saver()

  with tf.Session() as sess:

    # Initialize global variables
    print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
    sess.run( tf.global_variables_initializer() )
    
    # Restore saved weights
    print( "{timestamp}\t{memory}\tRestoring saved model ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
    saver.restore(sess, "./tmp/neurosat.ckpt")

    # Test SR distribution
    test_with(
      sess,
      solver,
      "./test-instances",
      "SR",
      time_steps = test_time_steps
    )
    # Test Phase Transition distribution
    test_with(
      sess,
      solver,
      "./critical-instances-40",
      "PT40",
      time_steps = test_time_steps
    )
    test_with(
      sess,
      solver,
      "./critical-instances-80",
      "PT80",
      time_steps = test_time_steps
    )
