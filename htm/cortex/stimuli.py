import compute
import pyopencl
import numpy

class Stimuli:
    def __init__(self, cs = None, numStimulus=None):
        if not isinstance(cs, compute.ComputeSystem):
            raise ValueError("[stimuli] Expected cs to be of type: {}, but got: {}".format(type(ComputeSystem), type(cs)))
        self.cs = cs

        if not isinstance(numStimulus, int):
            raise ValueError("[Stimuli] Expected numStimulus to be of type {} but got: {}".format(type(int), type(numStimulus)))
        self.numS = numStimulus

        self._numbytesSStates = numpy.uint32().nbytes * self.numS

        self.bufferSStates = pyopencl.Buffer(cs.getContext(), pyopencl.mem_flags.READ_WRITE, size=self._numbytesSStates)

        self.clearStates()

    def clearStates(self):
        pyopencl.enqueue_fill_buffer(self.cs.getQueue(), self.bufferSStates, numpy.uint32(0), 0, self._numbytesSStates)

    def setStates(self, states = []):
        pyopencl.enqueue_copy(self.cs.getQueue(), self.bufferSStates, states)
        #pyopencl.enqueue_write_buffer(self.cs.getQueue(), self.bufferSStates, numpy.uint32(0), self._numbytesSStates, states)

    def getStates(self):
        vecStates = None
        pyopencl.enqueue_read_buffer(self.cs.getQueue(), True, 0, self._numbytesSStates, vecStates)
        return vecStates

    def printStates(self):
        vecStates = None
        pyopencl.enqueue_read_buffer(self.cs.getQueue(), True, 0, self._numbytesSStates, vecStates)
        for i in xrange(self.numS):
            print("{} ".format(vecStates[i]))

if __name__ == "__main__":
    cs = compute.ComputeSystem(devicetype=compute.GPU)
    s = Stimuli(cs=cs, numStimulus=4)
