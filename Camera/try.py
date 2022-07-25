from webcamz import webCam

runner = webCam()

runner.open(address="holamundo")

runner.example_frame()

runner.close()