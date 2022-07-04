import cv2
import ctypes
import numpy as np
import tisgrabber as tis
ic = ctypes.cdll.LoadLibrary("./tisgrabber_x64.dll")
tis.declareFunctions(ic)


class ICCamera(object):
    def __init__(self, device_name=None, dev_format="RGB32 (1024x768)"):
        self.device_name = device_name
        self.g = ic.IC_CreateGrabber()
        ic.IC_OpenDevByUniqueName(self.g, tis.T(device_name))
        if not ic.IC_IsDevValid(self.g):
            raise Exception("Failed to open camera " + device_name)

        # Set camera resolution and FPS here
        ic.IC_SetVideoFormat(self.g, tis.T(dev_format))
        ic.IC_SetFrameRate(self.g, ctypes.c_float(20.0))
        ic.IC_StartLive(self.g, 0)

    @staticmethod
    def device_count():
        ic.IC_InitLibrary(0)
        return ic.IC_GetDeviceCount()

    @staticmethod
    def enum_names():
        device_cnt = ICCamera.device_count()
        device_names = list()
        for i in range(device_cnt):
            device_name = tis.D(ic.IC_GetUniqueNamefromList(i))
            device_names.append(device_name)
        return device_names

    @staticmethod
    def enum_devices():
        device_names = ICCamera.enum_names()
        devices = list()
        for i, device_name in enumerate(device_names):
            devices.append(ICCamera(device_name=device_name))
        return devices

    def snap(self):
        if ic.IC_SnapImage(self.g, 50) == tis.IC_SUCCESS:
            # Declare variables of image description
            Width = ctypes.c_long()
            Height = ctypes.c_long()
            BitsPerPixel = ctypes.c_int()
            colorformat = ctypes.c_int()

            # Query the values of image description
            ic.IC_GetImageDescription(self.g, Width, Height, BitsPerPixel, colorformat)

            # Calculate the buffer size
            bpp = int(BitsPerPixel.value / 8.0)
            buffer_size = Width.value * Height.value * BitsPerPixel.value

            # Get the image data
            imagePtr = ic.IC_GetImagePtr(self.g)

            imagedata = ctypes.cast(imagePtr, ctypes.POINTER(ctypes.c_ubyte * buffer_size))

            # Create the numpy array
            image = np.ndarray(buffer=imagedata.contents,
                               dtype=np.uint8,
                               shape=(Height.value, Width.value, bpp))
            image = cv2.flip(image, 0)
            return image
        else:
            return None


if __name__ == '__main__':
    print(ICCamera.enum_names())
    devices = ICCamera.enum_devices()
    while True:
        img = devices[0].snap()
        if img is not None:
            cv2.imshow("xx", img)
            cv2.waitKey(50)