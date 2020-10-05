"""
@author Phu Vuong
3rd October 2020, Fall Semester @ Jeonbuk National University, Computer Vision Course
Instructor: Prof. Hyojong Lee
"""
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import copy

class canny_detector():
    def __init__(self,img_path = None,gaussian_size=3,gaussian_sigma=1,highThresholdRatio=0.17,lowThresholdRatio=0.4,weak_pixel =75, strong_pixel =255,
                                        strides=1, padding='same', verbose=True, title=None, double_thresh=True, threshold_=30):
        """ This is the canny detector implemented in Python3. This class will initialize some parameters for edge detection. 
        gaussian_size: is the kernel size of Gaussian filter
        gaussian_sigma: gaussian sigma hyperparameter
        highThresholdRatio and lowThresholdRatio are the parameters we want to tune. High highThresholdRatio means the less edge we want, lower highThresholdRatio means we want more edge
        and this could result in high noise. lowThresholdRatio is the ratio due to highthreshold.

                                    |--------   those pixels defined as strong pixels => edge pixels
            highthreshold---------- |        ---------                                }  
                                    |                  -----------                    }  => weak pixels, these pixels will be reconsidered by edge_hysteresis    
            lowthreshold----------- |                             ---------           }  
                                    |                                      ------------ => eliminate  

        strides is the how much you want to slide your kernel through the image. This function only work with strides = 1 at the moment
        padding is how you want to output to be. If padding = 'same', then the output image will be the same size as the input. 
        double_thresh is set to True when we want to use canny detector, if it set to False we will threshold the image by only one threshold
        threshold_ is the value of threshold we want to set when we don't use double threshold
        verbose = True when we want to show the intermediate image as well as the final result
        title is the title of the graph
        highthresholdratio is proportional to the maximum pixel in the image. it uses to compute the highthreshold
        lowthresholdratio is proportional the highthreshold. it uses to compute lowthreshold 
        how to use:
        firstly we have to import this class to the command prompt or if you use notebook can also do by: from canny_detector import canny_detector
        then initialize this class: canny = canny_detector() we also need to set the parameter when calling this class
        finally call the function detect_edge()

        if you have any problem can contact me via vmphu@outlook.com

        """
        self.img_path = img_path
        self.gaussian_size = gaussian_size
        self.sigma = gaussian_sigma
        self.highThresholdRatio = highThresholdRatio
        self.lowThresholdRatio = lowThresholdRatio
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.strides = strides
        self.padding = padding
        self.verbose = verbose
        self.title = title
        self.double_thresh =double_thresh
        self.threshold_ = threshold_

    def convolve(self, img_path, kernel, title = None):
        """
        This function will take an image, a kernel as input and return the image which is filtered by the kernel.
        The kernel should be in square size, for example (3,3), (5,5), (7,7), (9,9)....

        strides is the how much you want to slide your kernel through the image. This function only work with strides = 1 at the moment

        padding is how you want to output to be. If padding = 'same', then the output image will be the same size as the input. 
        """
        if type(img_path)!=str:
            image = img_path
        else:
        # read image
            image = np.array(Image.open(img_path))

        if len(image.shape) ==3:
            image_row, image_col,image_cha = image.shape
        else:
            image_row, image_col =image.shape


        kernel = np.flip(kernel)    # flip the kernel before convolve
        kernel_row, kernel_col = kernel.shape

        # pad the image with zero
        if self.padding == 'same': # padding = same when we want the output equal input
            out_height = int(np.ceil(float(image_row) / float(self.strides)))
            out_width  = int(np.ceil(float(image_col) / float(self.strides)))
        elif self.padding == 'valid': # padding = valid, the output size and input size are different but be able to convolve 
            out_height = int(np.ceil(float(image_row - kernel_row + 1) / float(self.strides)))
            out_width  = int(np.ceil(float(image_col - kernel_col + 1) / float(self.strides)))
        pad_along_height = max((out_height - 1) * self.strides +
                        kernel_row - image_row, 0)
        pad_along_width = max((out_width - 1) * self.strides +
                        kernel_col - image_col, 0)
        pad_top = int(pad_along_height // 2)
        pad_bottom = int(pad_along_height - pad_top)
        pad_left = int(pad_along_width // 2)
        pad_right = int(pad_along_width - pad_left)
    
        
        #create a empty numpy array to store padded image
        padded_image = np.ndarray(((image_row +(pad_top+pad_bottom)), 
                                (image_col +(pad_left+pad_right)) ),np.int) 

        # pad image 
        padded_image[:,:] = np.pad( image,((pad_top,pad_bottom), (pad_left,pad_right)),'constant')

        # create an output to store the result
        output = np.zeros((out_height, out_width))

        # naive two for loops, there is another way but I did not have time to research further 
        for row in range(image_row):
            for col in range(image_col):
                output[row, col] = (1/(kernel_row * kernel_col)) *np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col]) 

        # retrieve img from numpy array
        output1= Image.fromarray(np.array(np.uint8(output)))  

        if self.verbose:
            plt.imshow(output1, cmap ='gray')
            plt.title("Output Image using {} ".format(title))
            plt.show()
    
        return output


    def gaussian_kernel(self):
        size = self.gaussian_size
        sigma = self.sigma
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g   

    def sobel_detection(self, img):

        # create x_kernel and y_kernel
        sobel_x_kernel =np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_y_kernel =np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        # convolve the image with kernels
        dx = self.convolve(img,sobel_x_kernel, title="gradient in x-direction")
        dy = self.convolve(img,sobel_y_kernel, title="gradient in y-direction")

        #compute gradient magnitude
        G = np.sqrt((np.square(dx)+np.square(dy)))
        G = G / G.max() * 255
        theta = np.arctan(dy, dx) # compute theta
        if self.verbose:
            plt.imshow(G,cmap='gray')
            plt.title("Output Image using sobel filter")
            plt.show()
        return G,theta  

    def non_max_suppression(self,img, theta):
        M, N = img.shape
        # create an output
        Z = np.zeros((M,N), dtype=np.int32)

        # convert theta from radian to degree
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180

        
        for i in range(M):
            for j in range(N):
                try:  # use try to excaspe the border pixels
                    q = 255
                    r = 255
                    
                    #angle 0, horizontal 
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45, diagonal 
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135, diagonal
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0

                except:
                    pass
        if self.verbose:
            plt.imshow(Z,cmap='gray')
            plt.title("Output Image using non maximum supression ")
            plt.show()        
        return Z

    def double_threshold(self, img):
        
        # compute the highthreshold and lowthreshold respectively based on the ratio 
        highThreshold = img.max() * self.highThresholdRatio
        lowThreshold =  self.lowThresholdRatio*highThreshold
        
        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32) # create the result matrix for the output
        
        weak = self.weak_pixel
        strong = self.strong_pixel
        
        strong_i, strong_j = np.where(img >= highThreshold)  # find where the strong pixels which are greater than the highthreshold
        
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold)) # find weak pixels which are greater the the lowthreshold but smaller than highthreshold
        
        # finally get the image with only strong and weak pixels
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        if self.verbose:
            plt.imshow(res,cmap='gray')
            plt.title("Output Image using double thresholding ")
            plt.show()   
        
        return res
    def edge_hysteresis(self, img):
        M, N = img.shape  
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try: # a pixel is considered to be strong if it connected to at least one strong pixel neighbor 
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else: # if not eliminate it
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        if self.verbose:
            plt.imshow(img,cmap='gray')
            plt.title("Output Image using hysteresis ")
            plt.show()                           
        return img
    def normal_threshold(self,img):
        # normal thresholding 
        threshold_ = self.threshold_
        img_thres = copy.copy(img)
        img_thres[img < threshold_] = 0
        img_thres[img_thres[:]!=0]=255
        if self.verbose:
            plt.imshow(img_thres, cmap ='gray')
            plt.title("Output Image using normal threshold={}".format(threshold_))
            plt.show()
    def detect_edge(self):
        smooth = self.convolve(self.img_path,self.gaussian_kernel(), title ="Gaussian smooth")
        after_sobel, theta = self.sobel_detection(smooth)
        after_nms = self.non_max_suppression(after_sobel, theta)
        if self.double_thresh:
            thresholding = self.double_threshold(after_nms)
            result = self.edge_hysteresis(thresholding)
        else:
            result = self.normal_threshold(after_nms)
        return result


