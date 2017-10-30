from compute.computesystem import ComputeSystem, CPU, GPU, ALL
import pyopencl

class ComputeProgram:

    def __init__(self):
        self.program = None

    def loadFromFile(self, cs, filename):
        if not isinstance(cs, ComputeSystem):
            raise ValueError("[compute] loadFromFile expected ComputeSystem, got: {}".format(type(cs)))

        # Read in filename
        kernel = ""
        with open(filename, 'r') as f:
            kernel = f.read()

        self.loadFromString(cs, kernel)

    def loadFromString(self, cs, kernel):
        if not isinstance(cs, ComputeSystem):
            raise ValueError("[compute] loadFromString expected ComputeSystem, got: {}".format(type(cs)))

        # Load program into opencl
        self.program = pyopencl.Program(cs.getContext(), kernel)
        # Compile program
        self.program = self.program.build(devices=[cs.getDevice()])

    def getProgram(self):
        return self.program


if __name__ == "__main__":
    cs = ComputeSystem(devicetype=GPU)
    cp = ComputeProgram()
    cp.loadFromFile(cs, "test.cl")
