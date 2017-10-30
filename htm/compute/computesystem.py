import pyopencl

GPU = "GPU"
CPU = "CPU"
ALL = "ALL"

class ComputeSystemNotFoundError(Exception):
    # Exception to handle missing compute system
    pass

class AmbiguousCopmuteSystems(Exception):
    # Exception to handle ambigous compute-system definitions
    pass

class DeviceTypeNotFound(Exception):
    # Exception to handle an undefined device-type
    pass

class AmbiguousDevices(Exception):
    # Exception to handle amgiuous device definitions
    pass

class ComputeSystem:
    def __init__(self, devicetype=""):
        # Find the opencl platform to use
        allPlatforms = pyopencl.get_platforms()
        if len(allPlatforms) == 0:
            raise ComputeSystemNotFoundError("[compute] No opencl platforms found")
        elif len(allPlatforms) > 1:
            raise AmbiguousComputeSystems("[compute] Found multiple opencl platforms but only has support for one")

        self.platform = allPlatforms[0]


        # Find the opencl device to use
        allDevices = pyopencl.Device
        if devicetype == GPU:
            allDevices = self.platform.get_devices(pyopencl.device_type.GPU)
        elif devicetype == CPU:
            allDevices = self.platform.get_devices(pyopencl.device_type.CPU)
        elif devicetype == ALL:
            allDevices = self.platform.get_devices(pyopencl.device_type.ALL)
        else:
            raise DeviceTypeNotFound("[compute] device-type {} not handled".format(devicetype))

        if len(allDevices) == 0:
            raise DeviceTypeNotFound("[compute] device-type {} not found".format(devicetype))
        elif len(allDevices) > 1:
            raise AmbiguousDevices("[compute] found multiple opencl devices matching: {}, only have support for one")

        self.device = allDevices[0]

        # Grab a context
        self.context = pyopencl.Context(devices=[self.device])

        # Grab a Queue
        self.queue = pyopencl.CommandQueue(self.context, device=self.device)



    def getPlatform(self):
        return self.platform

    def getDevice(self):
        return self.device

    def getContext(self):
        return self.context

    def getQueue(self):
        return self.queue

    def printClInfo(self):
        print("[compute] OpenCL Version: {}".format(self.getPlatform().get_info(pyopencl.platform_info.VERSION)))
        print("[compute] OpenCL Platform: {}".format(self.getPlatform().get_info(pyopencl.platform_info.NAME)))
        print("[compute] OpenCL Device: {}".format(self.getDevice().get_info(pyopencl.device_info.NAME)))


if __name__ == "__main__":
    cs = ComputeSystem(devicetype=GPU)
    print("Got platform: {}".format(cs.getPlatform()))
    print("Got device: {}".format(cs.getDevice()))
    print("Got context: {}".format(cs.getContext()))
    print("Got queue: {}".format(cs.getQueue()))
    cs.printClInfo()
