'''
* evaluations.py
* Contains open-cv image evaluations. Part of vision box project.
* @author Sasso, David [david.sasso@task.com.mx] Feb, 2022
*
* Integrated evaluations: 
*   -pixelcount
*   -objectcount
*   -colordetection
*
* Process for adding new evaluation: 
*   1. Add the fuction from the new evaluation on I. FUNCTIONS.
*   2. Add the case, options and asserts on II. CASES
*   3. Test
'''

from contextlib import nullcontext
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

# DEFINITIONS

class NumericLimitTest():

    def __init__(self, datasource, low: int, high: int, units='Units', comparision_type='GELE', testname='TestName'):
        self.datasource = datasource
        self.comparision_type = comparision_type
        self.low = low
        self.high = high
        self.units = units
        self.result = False
        self.checked = False
        self.testname = testname

    def check_params(self):
        self.checked = self.low < self.high
        pass

    def eval(self):
        if self.checked:
            if self.comparision_type == 'GELE':
                self.result = (self.datasource >= self.low) and (self.datasource <= self.high)
            elif self.comparision_type == 'GE':
                self.result = self.datasource >= self.low
            elif self.comparision_type == 'EQ':
                self.result = (self.datasource == self.low)
            else: #GELE default
                self.result = (self.datasource >= self.low) and (self.datasource <= self.high)
        else:
            raise Exception('Invalid Parameters.')
        return self.result

    def getResult(self):
        result = f'Testname:[{self.testname}];Result:[{self.result}];Measurement:[{self.datasource}];LowLimit:[{self.low}];HighLimit:[{self.high}];Units:[{self.units}]'
        return result

    def __str__(self):
        return f'Testname[{self.testname}];Result[{self.result}];Measurement[{self.datasource}];LowLimit[{self.low}];HighLimit[{self.high}];Units[{self.units}]'

class StringValueTest():
    
    def __init__(self, datasource: str, expected_string_value: str, comparision_type='IgnoreCase'):
        self.datasource = datasource
        self.comparision_type = comparision_type
        self.expected_string_value = expected_string_value
        self.result = False
    
    def eval():
        pass

# UTILITIES

def map_value(x, in_min, in_max, out_min, out_max):
    '''Map the value from one range to another.'''
    mapped_value = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return mapped_value

def get_pixel_percent(meanValue, maxPercentageChange):
    '''Calculate pixels by percent'''
    deltaPixels = map_value(maxPercentageChange, 0, 100, 0, 255) #from 100% scale to 255 bit scale
    return deltaPixels

def arrayToImage():
    '''Convert an array to a numpy array to be used on opencv operations.'''
    frame = None
    return frame

# I. FUNCTIONS

def pixelCount(frame, pixel_type): #TODO image filtered or grayscale
    '''Pixel Count Evaluation Test.'''
    if pixel_type == 'blank':
        pixels = cv.countNonZero(frame)
    elif pixel_type == 'black':
        pixels = cv.countNonZero(frame)
    return pixels

def objectCount(frame, mode, method): #TODO defensive programming and add another attributes
    '''
    Object Count Evaluation Test: returns detected contours. Each contour is stored as a vector of points.

    frame: 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero pixels remain 0's, so the image is treated as binary.
    You can use compare, inRange, threshold, adaptiveThreshold, Canny to create a binary image.

    Mode: Contour retrieval mode, see retrieval modes at https://docs.opencv.org/4.4.0/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
    Method: Contour approximation method, see contour approximation modes at https://docs.opencv.org/4.4.0/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff

    resource: https://www.educba.com/opencv-findcontours/ and https://docs.opencv.org/4.4.0/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
    '''
    #mode = objectcount_params['Mode']
    #method = objectcount_params['Method']
    # 1. select mode
    if mode == 'RETR_EXTERNAL':
        mod = cv.RETR_EXTERNAL
    elif mode == '':
        mod == cv.RETR_EXTERNAL
    else:
        raise ValueError

    # 2. select method
    if method == 'CHAIN_APPROX_NONE':
        met = cv.CHAIN_APPROX_NONE
    elif method == '':
        met = cv.CHAIN_APPROX_NONE
    else:
        raise ValueError
    cnt, hierarchy = cv.findContours(frame, mod, met)
    return len(cnt)

def colorDetection(frame, desviationAllowed, reference):
    '''
    Color Detection Test: returns pixel ammount of a given color sample.

    frame: 8-bit BGR image to be evaluated.
    reference: 8-bit BGR reference image.

    resource: https://www.codespeedy.com/color-detection-using-opencv-in-python/
    '''

    # split channels from reference frame
    blueReference, greenReference, redReference = cv.split(reference)

    blueReferenceMean = round(blueReference.mean())
    greenReferenceMean = round(greenReference.mean())
    redReferenceMean = round(redReference.mean())

    referenceValues = np.array([blueReferenceMean, greenReferenceMean, redReferenceMean])

    deltaBlue = get_pixel_percent(blueReferenceMean, desviationAllowed)
    deltaGreen = get_pixel_percent(greenReferenceMean, desviationAllowed)
    deltaRed = get_pixel_percent(redReferenceMean, desviationAllowed)

    lower_mean_range = np.array(
    [
        round(blueReferenceMean - deltaBlue), 
        round(greenReferenceMean - deltaGreen), 
        round(redReferenceMean - deltaRed)
    ])

    upper_mean_range = np.array(
    [
        round(blueReferenceMean + deltaBlue), 
        round(greenReferenceMean + deltaGreen), 
        round(redReferenceMean + deltaRed)
    ])


    # split channels from frame
    blueFrame, greenFrame, redFrame = cv.split(frame)

    blueFrameMean = round(blueFrame.mean())
    greenFrameMean = round(greenFrame.mean())
    redFrameMean = round(redFrame.mean())

    # start test
    mask = cv.inRange(frame,lower_mean_range,upper_mean_range)

    colorPixels = pixelCount(mask, 'blank')

    frameDimensions = frame.shape

    totalPixels = frameDimensions[0] * frameDimensions[1]

    colorPercentage = (100 * colorPixels) / totalPixels

    return colorPercentage

def template (template, frame):
        (score, diff) = compare_ssim(template, frame, win_size=None, full=True)
        diff = (diff * 255).astype("uint8")
        score=score*100
        return int(score)

#  II. CASES

def run(frame, evaluation, params_raw): #TODO 
    '''
    Main function to implement vision evaluation.

    frame: numpy frame.
    evaluation: case to evaluate or method.
    params_raw: String with params separated by ',' for the same evaluation.
    
    '''
    status = False
    values = params_raw.split(',')

    if evaluation == 'pixelcount':
        print(f'Test: {evaluation}')
        pixel_type = values[0]
        datasource = pixelCount(frame, pixel_type)
        NLTparams = {'LowLimit':int(values[1]), 'HighLimit':int(values[2])} # cast to int
        Test = NumericLimitTest(datasource, NLTparams['LowLimit'], NLTparams['HighLimit'], units='Pixels', comparision_type='GELE')

    elif evaluation == 'objectcount':
        objectcount_params = {'Mode':values[0], 'Method':values[1]}
        datasource = objectCount(frame, objectcount_params)
        NLTparams = {'LowLimit':int(values[2]), 'HighLimit':int(values[3])} # cast to int
        Test = NumericLimitTest(datasource, NLTparams['LowLimit'], NLTparams['HighLimit'], units='Objects', comparision_type='EQ')

    elif evaluation == 'colordetection':
        #TODO curl image from database
        colordetection_params = {'desviationAllowed': values[0], 'passPercentage': values[1], 'reference': values[2]}
        #reference = arrayToImage() # TBD
        datasource = colorDetection(frame, colordetection_params)
        NLTparams = {'LowLimit':int(100 + values[1]), 'HighLimit':int(100 - values[1])} # cast to int

    elif evaluation == 'default':
        pass
    else:
        raise Exception('Error: Unknown Evaluation.')
    Test.check_params()
    status = Test.eval()
    #print(Test)
    result = Test.getResult()
    return result
