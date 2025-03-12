
from numpy import *
from matplotlib.pyplot import *
from tqdm import tqdm


class myRBM:
	vh = []
	vbias = []
	hbias = []

	def copy(self):
		nn = myRBM()
		nn.vh = self.vh.copy()
		nn.vbias = self.vbias.copy()
		nn.hbias = self.hbias.copy()
		return nn

	def IncrementStats(self, vis, hid):
		'''
		stats = IncrementStats(vis, hid, old_stats)

		Given the binary states of vis, and hid, increment the
		counting statistics in old_stats.

		Inputs
		  vis        row-vector of 1s and 0s
		  hid        row-vector of 1s and 0s
		  old_stats  a network data structure like that used to store
					 connection weights and node biases.  See CreateNet for
					 more information.  The dimensions of vis, hid and vis2
					 should be consistent with old_stats.

		Output
		  stats      another network data structure with incremented counts
					 Note that loopback connections are not counted.  That is,
					 counts are not tabulated for a node's connection to
					 itself.  A node's activity level is stored in its bias.
		'''

		#pdb.set_trace()
		dim = len(shape(vis))
		if dim==1:
			self.vh     += outer(vis, hid)
			self.vbias  -= vis
			self.hbias  -= hid
		else:
			self.vh     += vis.T @ hid
			self.vbias  -= sum(vis, axis=0)
			self.hbias  -= sum(hid, axis=0)

		return

	def MA2self(self, netA, C):
		'''
		MA2self(netA, C)
		Increments self with C*netA.
		'''
		self.vh += C*netA.vh
		self.vbias += C*netA.vbias
		self.hbias += C*netA.hbias
		return

	def save(self, fname):
		savez(fname, vh=self.vh, vbias=self.vbias, hbias=self.hbias)
		return

	def load(self, fname):
		blah = load(fname)
		self.vh = blah['vh']
		self.vbias = blah['vbias']
		self.hbias = blah['hbias']
		return

	def EnergyFrequencies(self, iters=5000):
		'''
		energy, freq = EnergyFrequencies(iters=5000)

		Estimate the energy and visitation frequencies of all network states.

		Input:
		  iters is the number of iterations to estimate visitation freq.

		Output:
		  energy is the normalized energy of each state, saved as a vector
		       Normalized so that 0 <= energy <= 1
		  freq is the normalized visitation frequencies, saved as a vector
		       Normalized so that sum(freq) = 1
		'''
		numvis = len(self.vbias)
		numhid = len(self.hbias)

		vis_states = 2**numvis
		hid_states = 2**numhid
		total_states = vis_states * hid_states

		# Evaluate the energy of the selfwork states
		Energy = zeros([vis_states, hid_states])
		for vis_idx in range(vis_states):
			v = Int2Vec(vis_idx, numvis)
			for hid_idx in range(hid_states):
				h = Int2Vec(hid_idx, numhid)
				Energy[vis_idx,hid_idx] = -dot(dot(v,self.vh),h) + dot(v,self.vbias) + dot(h,self.hbias)

		# Sample visitation frequency of the states
		# Start with a random state
		vis = around(random.rand(numvis))
		hid = around(random.rand(numhid))

		s_count = zeros([vis_states, hid_states])
		for k in range(iters):
			vis = around(random.rand(numvis))
			hid = around(random.rand(numhid))
			# First we run the self for 5 steps
			for j in range(5):
				hid = to_binary( Recognize(vis, self, 3) )
				vis = to_binary( Generate(hid, self, 3) )
			# The run it some more and record states visited
			for j in range(10):
				hid = to_binary( Recognize(vis, self, 1) )
				vis = to_binary( Generate(hid, self, 1) )
				vis_idx = Vec2Int(vis)
				hid_idx = Vec2Int(hid)
				s_count[vis_idx, hid_idx] += 1

		# Plot of Energy/Frequency over all selfwork states
		sc = (s_count - min(s_count.flatten()))/(max(s_count.flatten())-min(s_count.flatten()))
		En = (Energy - min(Energy.flatten()))/(max(Energy.flatten())-min(Energy.flatten()))

		return En, sc


	def EvaluateAllStates(self, fig1=1, fig2=2):
		'''
		EvaluateAllStates(fig1, fig2)

		Plot the energies and visitation frequencies for all network states.

		Input:
		 fig1 and fig2 are figure numbers for the plots.
		      (default to 1 and 2)
		'''

		Energy, freq = self.EnergyFrequencies(2000)

		figure(fig1), clf()
		plot(Energy.flatten(),'r:')
		numvis = len(self.vbias)
		stem(freq.flatten(), linefmt='b-', markerfmt='bo')#, use_line_collection=True)
		title('Energy/Frequency')
		xlabel('State Index')
		ylabel('Normalized Freq or Energy')
		legend(['Energy', 'Freq'])

		fig = figure(fig2)
		fig.clf()
		ax = fig.add_subplot(111, projection='3d')
		r,c = mgrid[0:8,0:4]
		max_freq = np.max(freq)
		for rr,cc,zz in zip(r.flatten(), c.flatten(), freq.flatten()):
			ax.plot([rr,rr], [cc,cc], [0,zz], 'o-', color=(1.-zz/max_freq)/2.*np.ones(3))
		#ax.plot_surface(r,c,freq, rstride=1, cstride=1, cmap=cm.magma)
		title('Frequency')

		return


	def EvaluateVisStates(self,fig1=1,fig2=2):
		'''
		evaluate(fig1, fig2)
		fig1 and fig2 are figure numbers for the plots.
		(default to 1 and 2)
		'''

		Energy, freq = self.EnergyFrequencies(2000)

		Ev = sum(Energy,1)  # sum over all hidden states
		Ev = (Ev-min(Ev))/(max(Ev)-min(Ev))

		figure(fig1), clf()
		Ev = sum(Energy,1)  # sum over all hidden states
		Ev = (Ev-min(Ev))/(max(Ev)-min(Ev))
		Fvis = sum(freq,1) / sum(freq.flatten())
		plot(Ev,'r:')
		numvis = len(self.vbias)
		stem(range(2**numvis), Fvis, linefmt='b-', markerfmt='bo', use_line_collection=True)
		title('Energy/Frequency')
		xlabel('Visible State Index')
		ylabel('Normalized Freq or Energy')
		legend(['Energy', 'Freq'])

		fig = figure(fig2)
		fig.clf()
		ax = fig.add_subplot(111, projection='3d')
		r,c = mgrid[0:8,0:4]
		max_freq = np.max(freq)
		for rr,cc,zz in zip(r.flatten(), c.flatten(), freq.flatten()):
			ax.plot([rr,rr], [cc,cc], [0,zz], 'o-', color=(1.-zz/max_freq)/2.*np.ones(3))
		#ax.plot_surface(r,c,freq, rstride=1, cstride=1, cmap=cm.gray)
		title('Frequency')

		return

	def train2(self, stimulus, npasses=200):
		numvis = len(self.vbias)
		numhid = len(self.hbias)

		#npasses = 200  # was 200

		# Unsupervised learning is an iterative process that takes multiple passes.
		#
		# For each pass, you collect free and clamped statistics and use those
		# to update the weights and biases at the end of each pass.
		#
		# For each pass you collect activity statistics by...
		#   1) clamp vis if needed (randomly choosing stim/percept pair)
		#   2) set unclamped nodes randomly
		#   2) run network to equilibrium
		#   3) run network to collect Gibbs samples (statistics)
		#
		# To run the network to equilibrium, you gradually decrease the
		# temperature using "sched".

		#sched = [[  1   1   1   1   1   4    # # of iterations
		#         [30  25  20  15  12  10 ]  # temperature (for logistic)
		sched = array([[   1,   1,   3],    # # of iterations
				   [  20,  12,  10] ])  # temperature (for logistic)

		# sched(i,1) says how many iterations to perform at temp. sched(i,2)

		# In each pass, you can process more than one stim/percept pair.
		# In fact, you can process a whole bunch of cases, and the linear algebra
		# turns out to look the same.

		ncases = 40  # was 40

		# eg. let vis be many rows with one stimulus per row
		# Then vis*self.vh returns a matrix with a corresponding hid
		# vector on each row.

		# Computing the statistics used to look like this...
		# vis' * hid  (that's a col-vector times a row-vector)
		# The same formula still works with matrices.  It just adds up over
		# the multiple cases, resulting in a matrix the same shape as before.

		# How many network steps to take when tabulating counts.
		gibbs_samples = 10

		# Finally, here is a parameter for learning rate.
		epsilon = 0.8 # Learning rate (probably between 0.01 and 2)


		# how many stim/percept pairs do we have to choose from?
		numstates = shape(stimulus)[0]

		# This can take a while, so give some progress feedback.
		#h = waitbar(0,'Please wait...')

		epochs = shape(sched)[1]  # or len(sched[0])

		for passes in range(npasses):

			# Reset counters for tabulating statistics
			clamped_stats = CreateRBM(numvis, numhid, 0)
			free_stats = CreateRBM(numvis, numhid, 0)

			for cases in range(ncases):
				# Randomly choose an environmental state
				env_idx = floor(random.rand()*numstates)
				vis_orig = stimulus[env_idx,:]

				# Add noise to visible nodes
				noise1 = ( random.rand(numvis)>0.95 ).astype(float)
				vis = vis_orig*(1-noise1) + (1-vis_orig)*noise1

				# Randomize unclamped units
				hid = ( random.rand(numhid)>0.5 ).astype(float)

				# Run clamped network to equilibrium
				for n in range(epochs):
					T = sched[1,n]
					hid = ClampedStep(vis, hid, self, T)

				# Collect statistics on clamped network
				for n in range(gibbs_samples):
					T = 5
					hid = ClampedStep(vis, hid, self, T)
					clamped_stats.IncrementStats(vis, hid)

				# Randomize unclamped units
				hid = ( random.rand(numhid)>0.5 ).astype(float)
				vis = vis_orig #bool2s( rand(1,numvis)>0.5 )

				# Run free network to equilibrium
				for n in range(epochs):
					T = sched[1,n]
					vis, hid = FreeStep(vis, hid, self, T)

				# Collect statistics on unclamped network
				for n in range(gibbs_samples):
					T = 5
					vis, hid = FreeStep(vis, hid, self, T)
					free_stats.IncrementStats(vis, hid)

			# Get difference in statistics
			stat_diff = MANet(clamped_stats, free_stats, -1)

			# And update network parameters according to the stats
			self.MA2self(stat_diff, epsilon/gibbs_samples/ncases)

			#waitbar(passes/npasses, h)
		return


	def Train(self, train, epochs=200, temps=[20,5,1]):
		numvis = len(self.vbias)
		numhid = len(self.hbias)

		#npasses = 200  # was 200

		# Unsupervised learning is an iterative process that takes multiple passes.
		#
		# For each pass, you collect free and clamped statistics and use those
		# to update the weights and biases at the end of each pass.
		#
		# For each pass you collect activity statistics by...
		#   1) clamp vis if needed (randomly choosing stim/percept pair)
		#   2) set unclamped nodes randomly
		#   2) run network to equilibrium
		#   3) run network to collect Gibbs samples (statistics)
		#
		# To run the network to equilibrium, you gradually decrease the
		# temperature using "sched".

		#sched = epochs * ones([2, len(temps)])
		#sched[1,:] = temps[:]
		#sched = array([[   10, 10,  10,  10, 10],   # # of iterations
		#			   [    20,  8,  5,  3, 1] ])  # temperature (for logistic)
		#sched = array([[      1,   1,   3],    # # of iterations
		#	            [   6, 3, 1] ])  # temperature (for logistic)

		# sched[i,0] says how many iterations to perform at temp. sched[i,1]

		# In each pass, you can process more than one stim/percept pair.
		# In fact, you can process a whole bunch of cases, and the linear algebra
		# turns out to look the same.

		batch_size = 40   # was 40

		# eg. let vis be many rows with one stimulus per row
		# Then vis*self.vh returns a matrix with a corresponding hid
		# vector on each row.

		# Computing the statistics used to look like this...
		# vis' * hid  (that's a col-vector times a row-vector)
		# The same formula still works with matrices.  It just adds up over
		# the multiple cases, resulting in a matrix the same size as before.

		# Finally, here is a parameter for learning rate.
		epsilon = 0.1 # Learning rate (probably between 0.01 and 2)

		# how many stim/percept pairs do we have to choose from?
		#numstates = shape(stimulus)[0]

		# This can take a while, so give some progress feedback.
		#h = waitbar(0,'Please wait...')
		#epochs = len(sched[0])
		#total_work = sum(sched[0])*npasses
		counter = 0

		for T in tqdm(temps):
			for sched_idx in range(epochs):

				#T = sched[1,sched_idx]

				batches = MakeBatches(train, batch_size=batch_size)

				for batch in batches:

					# Reset counters for tabulating statistics
					# There aren't really RBMs; they are just used to duplicate
					# the weights data structures.
					clamped_stats = CreateRBM(numvis, numhid, 0)
					free_stats = CreateRBM(numvis, numhid, 0)

					noise1 = ( random.random(shape(batch))>0.99 )
					noise1 = noise1.astype(float)
					vis = batch*(1-noise1) + (1-batch)*noise1
					#vis = batch[:]

					hid = Recognize(vis, self, T)
					clamped_stats.IncrementStats(vis, hid)

					vis2 = Generate(to_binary(hid), self, T)

					hid2 = Recognize(vis2, self, T)
					free_stats.IncrementStats(vis2, hid2)

					# for samp in batch:
					#
					# 	# Randomly choose an environmental state
					# 	#env_idx = int(floor(random.rand()*numstates))
					# 	vis_orig = samp[:]
					#
					# 	# Add noise to visible nodes
					# 	noise1 = ( random.rand(numvis)>0.95 )
					# 	noise1 = noise1.astype(float)
					# 	vis = vis_orig*(1-noise1) + (1-vis_orig)*noise1
					#
					# 	# First up pass
					# 	hid = Recognize(vis, self, T)
					# 	clamped_stats.IncrementStats(vis, hid)
					#
					# 	# Down pass
					# 	vis2 = Generate(hid, self, T)
					#
					# 	# Second up pass
					# 	hid2 = Recognize(vis2, self, T)
					# 	free_stats.IncrementStats(vis2, hid2)


					# Get difference in statistics
					stat_diff = MANet(clamped_stats, free_stats, -1)

					# And update network parameters according to the stats
					self.MA2self(stat_diff, epsilon/batch_size)

					counter = counter + 1
					#waitbar(counter/total_work, h)

		print('Done.')
		return

def CreateRBM(numvis=0, numhid=0, kind=1):
	'''
	net = CreateNet(numvis, numhid, kind)

	Creates a network data structure, and fills it either with
	zeros or random numbers.

	Inputs
	  numvis   number of vis nodes
	  numhid   number of hid nodes
	  kind     either 0 (for all zeros)
				or 1 (for random fill)

	Output
	  net  data structure containing the network parameters
		  BIASES
			vbias  bias for vis nodes
			hbias   bias for hid nodes
		  RECURRENT CONNECTIONS
			vv    weights between vis nodes \
			hh      weights between hid nodes   > zero on diagonal
			(Note that nodes do not connect to themselves.)
		  INTER-LAYER CONNECTIONS
			vh     weights between vis and hid nodes
	'''

	net = myRBM()

	if numvis==0 and numhid==0:
		return net
	elif kind==0:
		# Create a zero net
		#net.vv   = zeros(numvis, numvis)
		#net.hh     = zeros(numhid, numhid)
		net.vh    = zeros([numvis, numhid])
		net.vbias = zeros(numvis)
		net.hbias  = zeros(numhid)
	else:
		# Create a random net
		# Make the recurrent weights zero on the diagonal so that
		# nodes do not connect to themselves.
		#net.vv   = rand(numvis, numvis, 'norm') .* (1-eye(numvis,numvis))
		#net.hh     = rand(numhid, numhid, 'norm') .* (1-eye(numhid,numhid))
		net.vh    = random.normal(0,1,[numvis, numhid])
		net.vbias = random.normal(0,1,[numvis])
		net.hbias  = random.normal(0,1,[numhid])

	return net




def MakeBatches(data_in, batch_size=10, shuffle=True):
    '''
    batches = MakeBatches(data_in, batch_size=10, shuffle=True)

    Breaks up the dataset into batches of size batch_size.

    Inputs:
      data_in    is a list of inputs
      batch_size is the number of samples in each batch
      shuffle    shuffle samples first (True)

    Output:
      batches is a list containing batches, where each batch is
	             an array

    Note: The last batch might be incomplete (smaller than batch_size).
    '''
    N = len(data_in)
    r = range(N)
    if shuffle:
        r = np.random.permutation(N)
    batches = []
    for k in range(0, N, batch_size):
        if k+batch_size<=N:
            din = data_in[r[k:k+batch_size]]
        else:
            din = data_in[r[k:]]
        if isinstance(din, (list, tuple)):
            batches.append(np.stack(din, dim=0))
        else:
            batches.append(din)

    return batches


def FreeStep(vis, hid, net, T):
	'''
	[v h] = FreeStep(vis, hid, net, T)

	Perform one step of the network stored in net, using the initial
	node states in vis, and hid.

	Input
	vis  current state of vis nodes
	hid  current state of hid nodes
	net  structure containing the network parameters
		 BIASES
		  vbias  bias for vis nodes
		  hbias  bias for hid nodes
		 RECURRENT CONNECTIONS
		  vv    weights between vis nodes  (zero on diagonal)
		  hh    weights between hid nodes  (zero on diagonal)
		 INTER-LAYER CONNECTIONS
		  vh    weights between vis and hid nodes
	T    temperature for Boltzmann energy

	Output
	v    new state of vis nodes
	h    new state of hid nodes
	'''

	hid_curr = net.hbias + dot(vis,net.vh)  # + hid*net.hh
	hidprob = logistic( hid_curr , T )
	d = len(hidprob)
	hid_copy = ( hidprob > random.rand(d) )

	#pdb.set_trace()

	vis_curr = net.vbias + dot(net.vh,hid)  #' # + vis*net.vv
	visprob = logistic( vis_curr , T )
	d = len(visprob)
	#vis_copy = ( visprob > random.rand(d) )

	v = vis_copy.astype(float)
	h = hid_copy.astype(float)

	return v, h


def ClampedStep(vis, hid, net, T):
	'''
	h = ClampedStep(vis, hid, net, T)

	Perform one step of the network stored in net, using the initial
	node states in vis, and hid. Only the hidden nodes are
	updated, since vis is clamped.

	Input
	  vis  current state of vis nodes
	  hid  current state of hid nodes
	  net  structure containing the network parameters
		  BIASES
		   vbias   bias for vis nodes
		   hbias   bias for hid nodes
		  RECURRENT CONNECTIONS
		   vv      weights between vis nodes  (zero on diagonal)
		   hh      weights between hid nodes  (zero on diagonal)
		  INTER-LAYER CONNECTIONS
		   vh      weights between vis and hid nodes
	  T    temperature for Boltzmann energy

	Output
	  h    new state of hid nodes (binary vector)
	'''

	hid_curr = net.hbias + dot(vis, net.vh)   # + hid*net.hh
	hidprob = logistic( hid_curr , T )
	d = len(hidprob)
	#hid_copy = ( hidprob > random.rand(d) )

	h = hid_copy.astype(float)

	return h


def Recognize(vis, net, T):
	'''
	h = Recognize(vis, net, T)

	Perform one recognition step of the network stored in net,
	using the initial node states in vis.

	Input
	  vis  current state of vis nodes
	  net  structure containing the network parameters
		    BIASES
		     vbias  bias for vis nodes
		     hbias  bias for hid nodes
		    INTER-LAYER CONNECTIONS
		     vh    weights between vis and hid nodes
	  T    temperature for Boltzmann energy

	Output
	  h    new state of hid nodes
	'''

	dim = len(shape(vis))
	if dim==1:
		hid_curr = -net.hbias + dot(vis, net.vh)
	else:
		hid_curr = -outer(ones(len(vis)), net.hbias) + vis@net.vh
	hidprob = logistic( hid_curr , T )
	#d = len(hidprob)
	#hid_copy = ( hidprob > random.random(np.shape(hidprob)) )

	#h = hid_copy.astype(float)
	return hidprob

def to_binary(p):
	v = ( p > random.random(np.shape(p)) )
	return v.astype(float)

def Generate(hid, net, T):
	'''
	v = Generate(hid, net, T)

	Perform one generative step of the network stored in net,
	using the initial node states in hid.

	Input
	  hid  current state of hid nodes
	  net  structure containing the network parameters
		   BIASES
		    vbias  bias for vis nodes
		    hbias  bias for hid nodes
		   INTER-LAYER CONNECTIONS
		    vh    weights between vis and hid nodes
	  T    temperature for Boltzmann energy

	Output
	  v    new state of vis nodes
	'''
	dim = len(shape(hid))
	if dim==1:
		vis_curr = -net.vbias + dot(net.vh, hid)
	else:
		vis_curr = -outer(ones(len(hid)), net.vbias) + dot(net.vh, hid.T).T
	visprob = logistic( vis_curr , T )
	#d = len(visprob)
	#vis_copy = ( visprob > random.random(np.shape(visprob)) )

	#v = vis_copy.astype(float)
	return visprob


def MANet(netA, netB, C):
	'''
	net = MANet(netA, netB, C)

	Multiply-Add function to combine two net structures arithmetically.
	In particular,

	net = netA + C*netB

	Inputs
	  netA and netB are network data structures, described in the help
		   for the function CreateNet.
	  C    scalar

	Output
	  net  a network data structure
	'''

	net = myRBM()
	net.vh = netA.vh + C*netB.vh

	net.vbias = netA.vbias + C*netB.vbias
	net.hbias = netA.hbias + C*netB.hbias

	return net


def logistic(x, T=1):
	'''
	p = logistic(x, T=1)
	Compute the logistic function at x, using temperature T (default T=1).
	'''
	p = 1.0 / (1 + exp(-x/T))
	return p



def Vec2Int(v):
	'''
	Converts a vector (v) of 1s and 0s into an integer.
	eg. 3 = Vec2Int([0, 1, 1])
	'''
	val = 0
	N = len(v)
	for k in range(N):
		val += v[-(k+1)]*2**k
	return int(val)



def Int2Vec(val,N):
	'''
	Converts a decimal value (val) to an 1xN row-vector of 1s and 0s.
	eg. [1, 0, 1, 1] = Int2Vec(11, 4)
	'''
	v = zeros(N)
	for k in range(N):
		v[-(k+1)] = floor( ( val*2**(-k) ) % 2)

	return v
