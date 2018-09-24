import os
import random
from cnf import CNF, BatchCNF, create_batchCNF
from functools import reduce


class InstanceLoader(object):

  def __init__(self,path):
    assert os.path.isdir( path ), "Path is not a directory. Path {}".format( path ) 
    if path[-1] == "/":
      path = path[0:-1]
    #end if

    sat_folder = path + '/sat/'
    unsat_folder = path + '/unsat/'
    
    self.filenames = reduce(lambda x,y: x + y, [ [sat.path,unsat.path] for (sat,unsat) in zip(os.scandir(sat_folder),os.scandir(unsat_folder)) ])

    self.reset()
  #end

  def get_instances(self, n_instances):
    for i in range(n_instances):
      yield CNF.read_dimacs(self.filenames[self.index])
      self.index += 1
    #end
  #end

  def get_batches(self, batch_size):
    for i in range( len(self.filenames) // batch_size ):
      yield create_batchCNF(self.get_instances(batch_size))
    #end
  #end

  def reset(self):
    self.index = 0
  #end
#end


#class InstanceLoader(object):
#
#  def __init__(self,path):
#    assert os.path.isdir( path ), "Path is not a directory. Path {}".format( path ) 
#    if path[-1] == "/":
#      path = path[0:-1]
#    #end if
#    folders = [path]
#    self.filenames = []
#    while len(folders)>0:
#      newfolders = []
#      for folder in folders:
#        newfolders += [ f.path for f in os.scandir(folder) if f.is_dir() ]
#        self.filenames += [ f.path for f in os.scandir(folder) if f.is_file() and f.path.endswith(".cnf") ]
#      #end for
#      folders = newfolders
#    #end while
#    self.reset()
#  #end
#
#  def get_instances(self, n_instances):
#    for i in range(n_instances):
#      yield CNF.read_dimacs(self.filenames[self.index])
#      self.index += 1
#    #end
#  #end
#
#  def get_batches(self, batch_size):
#    for i in range( len(self.filenames) // batch_size ):
#      yield create_batchCNF(self.get_instances(batch_size))
#    #end
#  #end
#
#  def reset(self):
#    random.shuffle( self.filenames )
#    self.index = 0
#  #end
##end


if __name__ == '__main__':

  instance_loader = InstanceLoaderPaired("instances")

#end