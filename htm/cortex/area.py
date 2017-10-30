import compute
import pyopencl
import numpy
import system

class Area:
    def __init__(self, cs = None, cp = None, numNpA = None):
        if not isinstance(cs, compute.ComputeSystem):
            raise ValueError("[stimuli] Expected cs to be of type: {}, but got: {}".format(compute.ComputeSystem, type(cs)))
        self.cs = cs

        if not isinstance(cp, compute.ComputeProgram):
            raise ValueError("[stimuli] Expected cp to be of type: {}, but got: {}".format(compute.ComputeProgram, type(cp)))
        self.cp = cp.getProgram()

        if not isinstance(numNpA, int):
            raise ValueError("[Stimuli] Expected numDenPFor to be of type {} but got: {}".format(int, type(numNpA)))
        self.numNpA = numNpA # number of dendrites per forest  

        self.numAn = numpy.uint32(1) # What?

        self.numbytesNBoosts = numpy.uint32().nbytes * self.numNpA # Bytes required for all boost values
        self.numbytesNStates = numpy.uint8().nbytes * self.numNpA # Bytes required for all state values
        self.numbytesNOverlaps = numpy.uint8().nbytes * self.numNpA # Bytes required for all overlap values
        self.numbytesInhibit = numpy.uint8().nbytes # Bytes required for inhibits (why 1 byte?)

        self.bufferNBoosts = pyopencl.Buffer(self.cs.getContext(), pyopencl.mem_flags.READ_WRITE, self.numbytesNBoosts)
        self.bufferNStates = pyopencl.Buffer(self.cs.getContext(), pyopencl.mem_flags.READ_WRITE, self.numbytesNStates)
        self.bufferNOverlaps = pyopencl.Buffer(self.cs.getContext(), pyopencl.mem_flags.READ_WRITE, self.numbytesNOverlaps)
        self.bufferInhibit = pyopencl.Buffer(self.cs.getContext(), pyopencl.mem_flags.READ_WRITE, self.numbytesInhibit)

        self.clearNBoosts()
        self.clearNStates()
        self.clearNOverlaps()
        self.clearNInhibit()

        self.kernelOverlapSynapses = self.cp.overlapSynapses
        self.kernelActivateNeurons = self.cp.activateNeurons
        self.kernelLearnSynapses = self.cp.learnSynapses
        self.kernelPredictNeurons = self.cp.predictNeurons
        self.kernelDecodeNeurons = self.cp.decodeNeurons

    def encode(self, arrStimuli=None, arrForest=None):
        # Overlap Synapses
        self.clearNOverlaps()

        for i in xrange(len(arrForest)):
            self.overlapSynapses(arrStimuli[i], arrForest[i])

        # Activate (and potentially Inhibit) Neurons
        self.clearNStates()
        self.clearNInhibit()

        _nThresh = numpy.uint32(arrForest.len())

        self.activateNeurons()

        _inhibit = 0
        # Read back current inhibit-value from opencl
        pyopencl.enqueue_copy_buffer(self.cs.getQueue(), _inhibit, self.bufferInhibit)

        # If no Neuron Inhibition, activate Neuron with highest Boost value
        if _inhibit == 0:
            # Create arrays to hold data copied out from opencl
            arrNBoosts = numpy.ndarray()
            arrNStates = numpy.ndarray()

            # Copy data from opencl
            pyopencl.enqueue_copy_buffer(self.cs.getQueue(), arrNBoosts, self.bufferNBoosts)
            pyopencl.qneueue_copy_buffer(self.cs.getQueue(), arrNStates, self.bufferNStates)

            for an in xrange(self.numAn):
                maxValue = 0
                maxIndex = 0

                for n in xrange(self.numNpA):
                    if arrNBoosts[n] > maxValue:
                        maxValue = arrNBoosts[n]
                        maxIndex = n

                arrNBoosts[maxIndex] = 0
                arrNStates[maxIndex] = 1

            pyopencl.enqueue_copy_buffer(self.cs.getQueue(), self.bufferNBoosts, arrNBoosts)
            pyopencl.enqueue_copy_buffer(self.cs.getQueue(), self.bufferNStates, arrNStates)



    def learn(self, arrStimuli=None, arrForest=None):
        for f in xrange(len(arrForest)):
            self.learnSynapses(arrStimuli[f], arrForest[f])

    def predict(self, arrStimuli=None, arrForest=None):
        # Overlap Synapses
        self.clearNOverlaps()

        for f in xrange(len(arrForest)):
            self.overlapSynapses(arrStimuli[f], arrForest[f])

        # Predict Neurons
        self.clearNStates()

        self._nThresh = len(arrForest)

        self.predictNeurons()

    def decode(self, arrStimuli=None, arrForest=None):
        for f in xrange(len(arrForest)):
            arrStimuli[f].clearstates()
            self.decodeNeurons(arrStimuli[f], arrForest[f])

    def getStates(self):
        arr = numpy.ndarray()
        pyopencl.enqueue_copy_buffer(cs.getQueue(), self.bufferNStates, arr)
        return arr

    def printStates(self):
        arr = numpy.ndarray()
        pyopencl.enqueue_copy_buffer(cs.getQueue(), self.bufferNStates, arr)
        print("States: ")

        for i in xrange(self._numNpA):
            print("{} ".format(arr[i]))

    def clearNBoosts(self):
        pyopencl.enqueue_fill_buffer(self.cs.getQueue(), self.bufferNBoosts, numpy.uint8(0), 0, self.numbytesNBoosts)

    def clearNStates(self):
        pyopencl.enqueue_fill_buffer(self.cs.getQueue(), self.bufferNStates, numpy.uint8(0), 0, self.numbytesNStates)

    def clearNOverlaps(self):
        pyopencl.enqueue_fill_buffer(self.cs.getQueue(), self.bufferNOverlaps, numpy.uint8(0), 0, self.numbytesNOverlaps)

    def clearNInhibit(self):
        pyopencl.enqueue_fill_buffer(self.cs.getQueue(), self.bufferInhibit, numpy.uint8(0), 0, self.numbytesInhibit)

    def overlapSynapses(self, Stimuli=None, Forest=None):
        self.kernelOverlapSynapses.set_args(self.bufferNOverlaps, Stimuli.bufferSStates, 
                Forest.bufferSAddrs, Forest.bufferSPerms, Forest.numSpD, Forest.dThresh)

        pyopencl.enqueue_nd_range_kernel(cs.getQueue(), self.kernelOverlapSynapses, self.numNpA)

    def activateNeurons(self):
        self.kernelActivateNeurons.set_args(self.bufferNBoosts, self.bufferNStates, 
                self.bufferNOverlaps, self.bufferInhibit, numpy.uint32(4294967295), self.nThresh)

        pyopencl.enqueue_nd_range_kernel(cs.getQueue(), self.kernelActivateNeurons, self.numNpA)

    def learnSynapses(self, Stimuli=None, Forest=None):
        self.kernelLearnSynapses.set_args(Stimuli.bufferSStates, Stimuli.numS, Forest.bufferSAddrs,
                Forest.bufferSPerms, forest.numSpD, self.bufferNStates, numpy.uint8(99))

        pyopencl.enqueue_nd_range_kernel(cs.getQueue(), self.kernelLearnSynapses, self.numNpA)

    def predictNeurons(self):
        self.kernelPredictNeurons.set_args(self.bufferNStates, self.bufferNOverlaps, self.nThresh)

        pyopencl.enqueue_nd_range_kernel(cs.getQueue(), self.kernelPredictNeurons, self.numNpA)

    def decodeNeurons(self, Stimuli=None, Forest=None):
        self.kernelDecodeNeurons.set_args(Stimuli.bufferSStates, self.bufferNStates,
                Forest.bufferSAddrs, Forest.bufferSPerms, Forest.numSpD)

        pyopencl.enqueue_nd_range_kernel(cs.getQueue(), self.kernelDecodeNeurons, self.numNpa)


if __name__ == "__main__":
    cs = compute.ComputeSystem(devicetype=compute.GPU)
    cp = compute.ComputeProgram()
    cp.loadFromFile(cs, "behavior.cl")
    s = Area(cs=cs, cp=cp, numNpA=1)

    numDataPoints = 30

    numForests = 2
    numNeurons = 1000
    # Placeholder for stimuli
    vecStimuli = []

    # Add inputlayer
    vecStimuli.append(system.Stimuli(cs, numDataPoints))
    # Add first neuron layer
    vecStimuli.append(system.Stimuli(cs, numNeurons))
    # Add second neuron layer
    vecStimuli.append(system.Stimuli(cs, numNeurons))
    # Add outputlayer
    vecStimuli.append(system.Stimuli(cs, numDataPoints))

    # Placeholder for forests
    vecForest = []

    vecForest.append(system.Forest(cs, cp, numNeurons, 50, 0.25))
    vecForest.append(system.Forest(cs, cp, numNeurons, 1, 1.0))

    area = system.Area(cs, cp, numNeurons)


    initstates = numpy.zeros(30, dtype=numpy.uint8)
    vecStimuli[0].setStates(initstates)
