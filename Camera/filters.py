'''
* vision_filters.py
* Contains open-cv image filters. Part of vision box project.
* @author Sasso, David [david.sasso@task.com.mx] Feb, 2022
*
* Integrated filters: 
*   -crop
*   -canny
*   -threshold
*   -dilate
*   -rode
*
* Process for adding new filter: 
*   1. Add the fuction from the new filter on I. FUNCTIONS.
*   2. Add the case, options and asserts on II. CASES
*   3. Test
'''

import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


#I. FUNCTIONS

def crop(frame, crop_params): #TODO: description/invalid parameters and default
    '''
    Crop Filter
    
    First_row: Start row to slice the numpy array (image).
    Last_row: Last row to slice the numpy array (image).
    First_column: Start column to slice the numpy array (image).
    Last_column: Last column to slice the numpy array (image).
    '''
    a, b, c, d = int(crop_params['First_row']), int(crop_params['Last_row']), int(crop_params['First_column']), int(crop_params['Last_column'])
    max_width = int(crop_params['width'])
    max_height = int(crop_params['height'])
    if a < b and c < d:
        new_frame = frame[a:b,c:d]
    else:
        raise ValueError
    return new_frame

def canny(frame, canny_params): #TODO: description/invalid parameters and default
    '''
    Canny Filter

    T_lower: Lower threshold value in Hysteresis Thresholding
    T_upper: Upper threshold value in Hysteresis Thresholding
    '''
    a, b = int(canny_params['T_lower']), int(canny_params['T_upper'])
    
    new_frame = cv.Canny(frame, a, b)
    return new_frame

def threshold(frame, threshold_params): #TODO: description/invalid parameters and default
    '''
    Threshold Filter

    Threshold_value: Value of Threshold below and above which pixel values will change accordingly. 
    Max_val: Maximum value that can be assigned to a pixel. 
    Thresholding_technique: The type of thresholding to be applied. 
    '''
    a, b, c = int(threshold_params['Threshold_value']), int(threshold_params['Max_val']), threshold_params['Threshold_technique']

    #Select hreshold_technique
    if c == 'binary':
        x = cv.THRESH_BINARY
    elif c == 'binary_inv':
        x = cv.THRESH_BINARY_INV
    elif c == 'tozero':
        x = cv.THRESH_TOZERO
    elif c == 'tozero_inv':
        x = cv.THRESH_TOZERO_INV
    else:
        x = cv.THRESH_BINARY
    ret, new_frame = cv.threshold(frame, a, b, x)
    return new_frame

def dilate(frame, dilate_params): #TODO: description/invalid parameters and default
    '''
    Dilate Filter

    Kernel: A matrix of odd size(3,5,7) is convolved with the image.
    Iterations: Number of iterations to make.
    '''
    x = int(dilate_params['Kernel']) #parse dimensions
    a = np.ones((x,x), np.uint8)
    b = int(dilate_params['Iterations'])

    new_frame = cv.dilate(frame, a, b)
    return new_frame

def erode(frame, erode_params): #TODO: description/invalid parameters and default
    '''
    Erode Filter

    Kernel: A matrix of odd size(3,5,7) is convolved with the image.
    Iterations: Number of iterations to make.
    '''
    x = int(erode_params['Kernel']) #parse dimensions
    a = np.ones((x,x), np.uint8)
    b = int(erode_params['Iterations'])

    new_frame = cv.erode(frame, a, b)
    return new_frame

def preview(frame, filter_name): #TODO time to preview
    try:
        cv.imshow(filter_name, frame)
        #cv.waitKey(1000)
        #cv.destroyAllWindows()
    except:
        print('Error: No image to preview.')
    return 0

def grayscale(frame, grayscale_params):
    '''
    Grayscale image

    This fiilter does not uses any params.
    '''
    new_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return new_frame

def gaussianblur(frame, gaussianblur_params):
    '''
    Gaussian Blur Filter

    shapeOfTheKernel: A matrix of odd size(3,5,7) is convolved with the image.
    sigmaX: The Gaussian kernel standard deviation which is the default set to 0.
    '''
    x = int(gaussianblur_params['Kernel'])
    a = (x,x) # this is how kernel should pass.
    b = int(gaussianblur_params['sigmaX'])
    new_frame = cv.GaussianBlur(frame, a, b)
    return new_frame

def cvtcolor(frame, cvtcolor_params):
    x = cvtcolor_params['code']
    
    #select color space
    if x == 'BGR2GRAY':
        c = cv.COLOR_BGR2GRAY
    elif x == 'BGR2RGB':
        c = cv.COLOR_BGR2RGB
    elif x == 'GRAY2BGR':
        c = cv.COLOR_GRAY2BGR
    elif x == 'GRAY2RGB':
        c = cv.COLOR_GRAY2RGB
    else:
        raise ValueError
    new_frame = cv.cvtColor(frame, c)
    return new_frame

def difference(template, frame): #TODO this is not to use as a filter, just to pass the filtered image
        (score, diff) = compare_ssim(template, frame, win_size=None, full=True)
        diff = (diff * 255).astype("uint8")
        new_frame = diff
        return new_frame

# II. CASES

def run(frame, filters_raw, params_raw, show=False, timeout = 500): #TODO add warnings
    '''
    Main function to implement vision filters.

    frame: numpy frame.
    filters_raw: filter's name separated by ';'
    params_raw: String with params separated by ',' for the same filter and by ';' for next filter.
    show: Shows the preview from each filter.
    
    '''
    filters = filters_raw.split('-')
    params = params_raw.split('-')
    #print(filters)
    #print(params)
    assert(len(filters) == len(params)), 'Filters-Parameters do not match.'
    for id in range(0,len(filters)):

        #print(f'Applying Filter: {filters[id]}')

        values = params[id].split(',')

        try:
            if crop.__name__ == filters[id]:
                height = str(frame.shape[0])
                width = str(frame.shape[1])
                crop_params = {'First_row':values[0],'Last_row':values[1],'First_column':values[2],'Last_column':values[3],'height':height,'width':width}
                #print(crop_params)
                try:
                    frame = crop(frame,crop_params)
                except:
                    raise Exception(f'WARNING: Image dimensions are height: {height} and  width: {width}.')

            elif canny.__name__ == filters[id]:
                canny_params = {'T_lower':values[0],'T_upper':values[1]}
                #print(canny_params)
                frame = canny(frame, canny_params)

            elif threshold.__name__ == filters[id]:
                threshold_params = {'Threshold_value':values[0],'Max_val':values[1],'Threshold_technique':values[2]}
                #print(threshold_params)
                frame = threshold(frame, threshold_params)

            elif dilate.__name__ == filters[id]:
                dilate_params = {'Kernel':values[0], 'Iterations':values[1]}
                #print(dilate_params)
                frame = dilate(frame, dilate_params)

            elif erode.__name__ == filters[id]:
                erode_params = {'Kernel':values[0], 'Iterations':values[1]}
                #print(erode_params)
                frame = erode(frame, erode_params)

            elif grayscale.__name__ == filters[id]:
                grayscale_params = None
                frame = grayscale(frame, grayscale_params)
            
            elif gaussianblur.__name__ == filters[id]:
                gaussianblur_params = {'Kernel':values[0], 'sigmaX':values[1]}
                frame = gaussianblur(frame, gaussianblur_params)
            
            elif cvtcolor.__name__ == filters[id]:
                cvtcolor_params = {'code':values[0]}
                frame = cvtcolor(frame, cvtcolor_params)

            else:
                print('Filters Done.')
        except:
            raise Exception(f'Error: Invalid parameters on {filters[id]} filter.')
        if show:
            preview(frame, (f'{id}-{filters[id]}'))
            cv.waitKey(timeout)
            cv.destroyAllWindows()
    return frame


# III. Debug and Testing

def debug():
    show = False
    filters_data = 'grayscale-crop-threshold-dilate'
    params_data = 'gray-100,260,100,540-200,255,binary-9,1'
    path = 'C:\\LTS\\smart_cell\\station_application\\source_code\\vision_box\\img\\calibration\\IBG\\IBG4x2x6_PASS_0.jpg'
    #frame = cv.imread(path, cv.IMREAD_GRAYSCALE)
    frame = cv.imread(path)
    result = run(frame, filters_data, params_data, show)

# Debugg only: Uncomment for testing from CLI.
#debug()
