
import os
import random
from cnf import CNF, BatchCNF, create_batchCNF

class InstanceLoader(object):

	def __init__(self,path):
		self.path = path

		self.filenames = [ 'instances/sat/'+x for x in os.listdir(path + '/sat')] + [ 'instances/unsat/'+x for x in os.listdir(path + '/unsat') ]
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
		random.shuffle( self.filenames )
		self.index = 0
	#end
#end

if __name__ == '__main__':

	instance_loader = InstanceLoader("instances")

	batches = instance_loader.get_batches(32)

	for batch in batches:
		print( batch.get_sparse_matrix() )
	#end
#end
