
import numpy as np
import copy
import itertools
import pycosat
import time

def SR(n):
	# Add clauses while satisfiable
	F1 = []
	# Current solutions
	buffer_size = 1
	solutions = list(itertools.islice(pycosat.itersolve(F1,n),buffer_size))
	while True:
		"""
			Randomly chooses 'k', the number of literals in this clause.
			k has mean 5 and is described by the distribution:
				2 + Bernouilli(.3) + Geo(.4)
		"""
		k = 2 + np.random.binomial(1,.3) + np.random.geometric(.4)
		"""
			Now we sample k variables uniformly at random
			and negate each one with equal probability to
			create a clause
		"""
		C = [ int(np.random.choice([-1,+1]) * np.random.randint(1,n+1)) for i in range(k) ]
		# Add C to the formula
		F1.append(C)
		
		# Check to see F1 is unsatisfiable
		
		# Filter solutions list for satisfiable solutions
		solutions = [ X for X in solutions if any([ np.sign(l) * np.sign(X[int(abs(l))-1]) == 1 for l in C ]) ]

		# If there are no satisfiable solutions in our buffer, run PycoSAT to look for new solutions
		if len(solutions) == 0:
			solutions = list(itertools.islice(pycosat.itersolve(F1,n),buffer_size))
			# If PycoSAT couldn't find any new solutions, the algorithm is done
			if solutions == []:
				break
			#end if
		elif len(solutions) < buffer_size:
			solutions += list(itertools.islice(pycosat.itersolve(F1,n),buffer_size-len(solutions)))
		#end if

	#end def

	"""
		By now, F1 is unsatisfied by X. But because it was
		satisfied up until the penultimate clause, we can
		create a satisfiable variant F2 by flipping the polarity
		of a single literal in the last clause
	"""
	F2 = copy.deepcopy(F1)
	F2[-1][ np.random.randint(0,len(F2[-1])) ] *= -1

	return F1, F2
#end def

def to_matrix(n,m,F):
	"""
		Converts a SAT instance from list format:
			i.e. a formula is a list of clauses, each of
			which is a list of literals, each of which is
			a 2-list with a polarity p ϵ {-1,+1} and a
			variable index i ϵ {0,n-1})
		into adjacency matrix format:
			i.e. a formula is a binary matrix M ϵ {0,1}²ⁿˣᵐ
			where
				M(i,j)		= 1 iff  xi ϵ Cj else 0,
				M(n+i,j) 	= 1 iff ¬xi ϵ Cj else 0
	"""
	M = np.zeros((2*n,m))
	for (j,C) in enumerate(F):
		for (p,i) in [ (np.sign(l), int(abs(l))-1) for l in C ]:
			if p == +1:
				M[i,j] = 1
			else:
				M[n+i,j] = 1
			#end if
		#end for
	#end for
	return M
#end def

def generate(n, m, batch_size = 32):
	while True:

		"""
			First we create a list of (batch_size//2) pairs of SAT formulas,
			filtering those which exceed m clauses
		"""
		unsat_formulas 	= []
		sat_formulas 	= []
		while len(unsat_formulas) < batch_size//2:
			unsat, sat = SR(n)
			if len(unsat) > m:
				continue
			else:
				unsat_formulas.append(unsat)
				sat_formulas.append(sat)
			#end if
		#end while

		"""
			Features are adjacency matrices with the following structure:
			M ϵ {0,1}²ⁿˣᵐ,
			M(i,j)		= 1 iff  xi ϵ Cj else 0,
			M(n+i,j) 	= 1 iff ¬xi ϵ Cj else 0
			where n is the number of variables and m the number of clauses

			Labels are binary scalars (+1 for satisfiable instances and -1 otherwise)
		"""
		features 	= np.zeros((batch_size, 2*n, m))
		labels 		= np.zeros((batch_size,))

		# We populate the batch in pairs, so we need just batch_size//2 iterations
		for i in range(batch_size//2):
			M1, M2 =  to_matrix(n,m,unsat_formulas[i]),  to_matrix(n,m,sat_formulas[i])

			features[2*i, 	:] = M1
			features[2*i+1, :] = M2

			labels[2*i] 	= -1
			labels[2*i+1] 	= +1
		#end for
		yield features, labels
	#end while
#end def

if __name__ == '__main__':

	n = 5
	m = 50

	generator = generate(n,m,batch_size=32)

	last_time = time.time()
	for batch in generator:
		print("Created batch in {} seconds".format(time.time()-last_time))
		last_time = time.time()
	#end for

#end if