from geniecolor import GenieColor as genie
import cv2 as cv

cam = genie()
cam.set_cti_path('C:\\LTS\\app-tde-luminaires\\SourceCode\\python_environment\\Instruments\\Camera\\cti\\mvGenTLProducer.cti')
ADDRESS = '172.16.5.130'
cam.open(ADDRESS)
cam.set_data(exposure=47000, threshold='100', ROI='0,500,0,500', led_number=50, apply_ROI=False, fixture_size=40)


cam.get_image()
frame = cam.frame_opencv
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

scale = 0.4
Width = int(frame.shape[1] * scale)  # Resize Width to 50%
Height = int(frame.shape[0] * scale)  # Resize Height to 50%
resized = cv.resize(gray, (Width, Height))  # Resize image

'''cv.imshow('frame', resized)
cv.waitKey(5000)
cv.destroyAllWindows()'''



cam.frame_opencv = gray

cam.apply_ROI()
cam.apply_filters()

print(cam.filtered_image)
result = cam.count_leds()
cam.show_image(gray, scale=0.5, timeout=5)
print(result)

cam.get_image()
frame = cam.frame_opencv
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

scale = 0.4
Width = int(frame.shape[1] * scale)  # Resize Width to 50%
Height = int(frame.shape[0] * scale)  # Resize Height to 50%
resized = cv.resize(gray, (Width, Height))  # Resize image

'''cv.imshow('frame', resized)
cv.waitKey(5000)
cv.destroyAllWindows()'''



cam.frame_opencv = gray

cam.apply_ROI()
cam.apply_filters()

print(cam.filtered_image)
result = cam.count_leds()
cam.show_image(gray, scale=0.5, timeout=5)
print(result)

cam.close()