import compute
import pyopencl
import numpy

class Forest:
    def __init__(self, cs = None, cp = None, numDenPFor=None, numSynPDen=None, threshPercent=None):
        if not isinstance(cs, compute.ComputeSystem):
            raise ValueError("[forest] Expected cs to be of type: {}, but got: {}".format(type(compute.ComputeSystem), type(cs)))
        self.cs = cs

        if not isinstance(cp, compute.ComputeProgram):
            raise ValueError("[forest] Expected cp to be of type: {}, but got: {}".format(compute.ComputeProgram, type(cp)))
        self.cp = cp.getProgram()

        if not isinstance(numDenPFor, int):
            raise ValueError("[forest] Expected numDenPFor to be of type {} but got: {}".format(type(int), type(numDenPFor)))
        self.numDpF = numDenPFor # number of dendrites per forest  

        if not isinstance(numSynPDen, int):
            raise ValueError("[Forest] Expected numSynPDen to be of type {} but got: {}".format(type(int), type(numSynPDen)))
        self.numSpD = numSynPDen # number of synapses per dendrite 

        if not isinstance(threshPercent, float):
            raise ValueError("[Forest] Expected thresPercent to be of type {} but got: {}".format(type(float), type(threshPercent)))
        self.numSpF = self.numDpF * self.numSpD # number of synapses per forest

        self.dThresh = self.numSpD * threshPercent -1 # dendrite activation threshold (how many active synapses needed to activate dendrite)
        if self.dThresh == 0:
            self.dThresh = 1

        self.numbytesSAddrs  = numpy.uint32().nbytes * self.numSpF
        self.numbytesSPerms  = numpy.uint8().nbytes * self.numSpF

        self.bufferSAddrs  = pyopencl.Buffer(cs.getContext(), pyopencl.mem_flags.READ_WRITE, self.numbytesSAddrs)
        self.bufferSPerms  = pyopencl.Buffer(cs.getContext(), pyopencl.mem_flags.READ_WRITE, self.numbytesSPerms)

        pyopencl.enqueue_fill_buffer(cs.getQueue(), self.bufferSAddrs, numpy.uint32(numpy.iinfo(numpy.uint32()).max), 0, self.numbytesSAddrs)
        pyopencl.enqueue_fill_buffer(cs.getQueue(), self.bufferSPerms, numpy.uint8(0), 0, self.numbytesSPerms)

    def encode(self, bufferStimulae=None):
        pass
        #pyopencl.enqueue_fill_buffer(cs.getQueue(), self.bufferSStates, numpy.uint8(0), 0, self._numbytesSStates)

    def learn(self):
        pass
        #pyopencl.enqueue_write_buffer(cs.getQueue(), self.bufferSStates, True, 0, self._numbytesSStates, states)

if __name__ == "__main__":
    cs = compute.ComputeSystem(devicetype=compute.GPU)
    cp = compute.ComputeProgram()
    cp.loadFromFile(cs, "../compute/test.cl")
    s = Forest(cs=cs, cp=cp, numDenPFor=1, numSynPDen=1, threshPercent=0.5)
