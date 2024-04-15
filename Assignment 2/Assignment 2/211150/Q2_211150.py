import numpy as np
import cv2
from scipy import signal, interpolate
import math
# import warnings

# warnings.filterwarnings("ignore")




def solution(img1, img2):
    
    
    # np.seterr(divide='ignore')

 

    def joint_bilateral(image, joint_image, sigmaspatial, sigmarange, samplespatial=None, samplerange=None):
        height, width = image.shape

        samplespatial = sigmaspatial if samplespatial is None else samplespatial
        samplerange = sigmarange if samplerange is None else samplerange

        flatimage = image.flatten()
        flatjoint = joint_image.flatten()

        edgemin = np.amin(flatimage)
        edgemax = np.amax(flatimage)
        edgedelta = edgemax - edgemin

        derivedspatial = sigmaspatial / samplespatial
        derivedrange = sigmarange / samplerange

        xypadding = round(2 * derivedspatial + 1)
        zpadding = round(2 * derivedrange + 1)

        samplewidth = int(round((width - 1) / samplespatial) + 1 + 2 * xypadding)
        sampleheight = int(round((height - 1) / samplespatial) + 1 + 2 * xypadding)
        sampledepth = int(round(edgedelta / samplerange) + 1 + 2 * zpadding)

        dataflat = np.zeros(sampleheight * samplewidth * sampledepth)
        jointflat = np.zeros_like(dataflat)

        (ygrid, xgrid) = np.meshgrid(range(width), range(height))

        dimx = np.around(xgrid / samplespatial) + xypadding
        dimy = np.around(ygrid / samplespatial) + xypadding
        dimz = np.around((image - edgemin) / samplerange) + zpadding

        flatx = dimx.flatten()
        flaty = dimy.flatten()
        flatz = dimz.flatten()

        dim = flatz + flaty * sampledepth + flatx * samplewidth * sampledepth
        dim = np.array(dim, dtype=int)

        dataflat[dim] = flatimage
        jointflat[dim] = flatjoint

        data = dataflat.reshape(sampleheight, samplewidth, sampledepth)
        joint_data = jointflat.reshape(sampleheight, samplewidth, sampledepth)

        weights = np.array(joint_data, dtype=bool)

        kerneldim = derivedspatial * 2 + 1
        kerneldep = 2 * derivedrange * 2 + 1
        halfkerneldim = round(kerneldim / 2)
        halfkerneldep = round(kerneldep / 2)

        (gridx, gridy, gridz) = np.meshgrid(range(int(kerneldim)), range(int(kerneldim)), range(int(kerneldep)))
        gridx -= int(halfkerneldim)
        gridy -= int(halfkerneldim)
        gridz -= int(halfkerneldep)

        gridsqr = ((gridx * gridx + gridy * gridy) / (derivedspatial * derivedspatial)) \
                + ((gridz * gridz) / (derivedrange * derivedrange))
        
        kernel = np.exp(-0.5 * gridsqr)

        blurdata = signal.fftconvolve(data, kernel, mode='same')
        # blurjoint = signal.fftconvolve(joint_data, kernel, mode='same')

        blurweights = signal.fftconvolve(weights, kernel, mode='same')
        blurweights = np.where(blurweights == 0, -2, blurweights)

        normalblurdata = blurdata / blurweights
        # normalblurjoint = blurjoint / blurweights

        normalblurdata = np.where(blurweights < -1, 0, normalblurdata)
        # normalblurjoint = np.where(blurweights < -1, 0, normalblurjoint)

        (ygrid, xgrid) = np.meshgrid(range(width), range(height))

        dimx = (xgrid / samplespatial) + xypadding
        dimy = (ygrid / samplespatial) + xypadding
        dimz = (image - edgemin) / samplerange + zpadding

        joint_output = interpolate.interpn((range(normalblurdata.shape[0]), range(normalblurdata.shape[1]),
                                            range(normalblurdata.shape[2])), normalblurdata, (dimx, dimy, dimz))

        return joint_output

    
    
    def bilateral(image, sigmaspatial, sigmarange, samplespatial=None, samplerange=None):
    

        height = image.shape[0]
        width = image.shape[1]

        samplespatial = sigmaspatial if (samplespatial is None) else samplespatial
        samplerange = sigmarange if (samplerange is None) else samplerange

        flatimage = image.flatten()

        edgemin = np.amin(flatimage)
        edgemax = np.amax(flatimage)
        edgedelta = edgemax - edgemin

        derivedspatial = sigmaspatial / samplespatial
        derivedrange = sigmarange / samplerange

        xypadding = round(2 * derivedspatial + 1)
        zpadding = round(2 * derivedrange + 1)

        samplewidth = int(round((width - 1) / samplespatial) + 1 + 2 * xypadding)
        sampleheight = int(round((height - 1) / samplespatial) + 1 + 2 * xypadding)
        sampledepth = int(round(edgedelta / samplerange) + 1 + 2 * zpadding)

        dataflat = np.zeros(sampleheight * samplewidth * sampledepth)

        (ygrid, xgrid) = np.meshgrid(range(width), range(height))

        dimx = np.around(xgrid / samplespatial) + xypadding
        dimy = np.around(ygrid / samplespatial) + xypadding
        dimz = np.around((image - edgemin) / samplerange) + zpadding

        flatx = dimx.flatten()
        flaty = dimy.flatten()
        flatz = dimz.flatten()

        dim = flatz + flaty * sampledepth + flatx * samplewidth * sampledepth
        dim = np.array(dim, dtype=int)

        dataflat[dim] = flatimage

        data = dataflat.reshape(sampleheight, samplewidth, sampledepth)
        weights = np.array(data, dtype=bool)

        kerneldim = derivedspatial * 2 + 1
        kerneldep = 2 * derivedrange * 2 + 1
        halfkerneldim = round(kerneldim / 2)
        halfkerneldep = round(kerneldep / 2)

        (gridx, gridy, gridz) = np.meshgrid(range(int(kerneldim)), range(int(kerneldim)), range(int(kerneldep)))
        gridx -= int(halfkerneldim)
        gridy -= int(halfkerneldim)
        gridz -= int(halfkerneldep)

        gridsqr = ((gridx * gridx + gridy * gridy) / (derivedspatial * derivedspatial)) \
            + ((gridz * gridz) / (derivedrange * derivedrange))
        kernel = np.exp(-0.5 * gridsqr)

        blurdata = signal.fftconvolve(data, kernel, mode='same')

        blurweights = signal.fftconvolve(weights, kernel, mode='same')
        blurweights = np.where(blurweights == 0, -2, blurweights)

        normalblurdata = blurdata / blurweights
        normalblurdata = np.where(blurweights < -1, 0, normalblurdata)

        (ygrid, xgrid) = np.meshgrid(range(width), range(height))

        dimx = (xgrid / samplespatial) + xypadding
        dimy = (ygrid / samplespatial) + xypadding
        dimz = (image - edgemin) / samplerange + zpadding

        return interpolate.interpn((range(normalblurdata.shape[0]), range(normalblurdata.shape[1]),
                                range(normalblurdata.shape[2])), normalblurdata, (dimx, dimy, dimz))   

    def combine(image):
        image[:, :, 0] = bilateral(image[:, :, 0], d, sigma)
        image[:, :, 1] = bilateral(image[:, :, 1], d, sigma)
        image[:, :, 2] = bilateral(image[:, :, 2], d, sigma)

        filtered_image = cv2.convertScaleAbs(image)

        return filtered_image

    # def display(image):
    #     image1= (cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
    #     cv2.imshow('image', image1)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    #     plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    #     plt.show()
    
    def combine_joint(image1,image2,d,sigma,k):
        image1[:, :, 0] = joint_bilateral(image1[:, :, 0], image2[:, :, 0], d, sigma,d/k,sigma/k)
        image1[:, :, 1] = joint_bilateral(image1[:, :, 1], image2[:, :, 1], d, sigma,d/k,sigma/k)
        image1[:, :, 2] = joint_bilateral(image1[:, :, 2], image2[:, :, 2], d, sigma,d/k,sigma/k)

        filtered_image = cv2.convertScaleAbs(image1)

        return filtered_image

    
    
    # def gauss(d, sigma):    
    #     weight = 1 / math.sqrt(2*math.pi* (d**2)) #gaussian function to calcualte the spacial kernel ( the first part 1/sigma * sqrt(2π))
    #     range= 1 / math.sqrt(2*math.pi* (sigma**2)) #gaussian function to calcualte the range kernel
    #     matrix = np.exp(-np.arange(256) * np.arange(256) * range)
    #     xx=-d + np.arange(2 * d + 1)
    #     yy=-d + np.arange(2 * d + 1)
    #     x, y = np.meshgrid(xx , yy )
    #     kernel = weight*np.exp(-(x **2 + y **2) /(2 * (weight**2) ) ) #calculate spatial kernel from the gaussian function. That is the gaussianSpatial variable multiplied with e to the power of (-x^2 + y^2 / 2*sigma^2) 
    #     return matrix,kernel


            
    # def cross_bilateral(img, img1, d, sigma):
    #     orgImg =np.pad(img, ((d, d), (d, d), (0, 0)), 'symmetric')

    #     secondImg = np.pad(img, ((d, d), (d, d), (0, 0)), 'symmetric')

    #     matrix,kernel=gauss(d, sigma) 
    #     h, w, z = img.shape 
        
    #     outputImg = np.zeros((h,w,z), np.uint8) 
    #     summ=1
    #     for x in range(d, d + h):
    #         for y in range(d, d + w):
    #             for i in range (0,z): 
    #                 neighbourhood=secondImg[x-d : x+d+1 , y-d : y+d+1, i] 
    #                 central=secondImg[x, y, i] 
    #                 res = matrix[ abs(neighbourhood - central) ]                 
    #                 summ=summ*res*kernel 
    #                 norm = np.sum(res) 
    #                 outputImg[x-d, y-d, i]= np.sum(res*orgImg[x-d : x+d+1, y-d : y+d+1, i]) / norm 
    #     return outputImg    
    
    def adjust_color_balance(image, red_weight, green_weight, blue_weight):
        image=image.astype('float')
        image[:, :, 0] *= blue_weight
        image[:, :, 1] *= green_weight
        image[:, :, 2] *= red_weight

        adjusted_image = cv2.merge([image[:, :, 0], image[:, :, 1], image[:, :, 2]])

        hi = np.clip(adjusted_image, 0, 255)

        return hi

    
    # def combine_special(image):
    #     b, g, r = cv2.split(image)
    #     base_b = bilateral(b,d,sigma)
    #     base_g = bilateral(g,d,sigma)
    #     base_r = bilateral(r,d,sigma)

    #     filtered_image = cv2.merge([base_b, base_g, base_r])
    #     filtered_image = cv2.convertScaleAbs(filtered_image)

    #     return filtered_image

    def calculate_color_weights(original_image, reference_image):
        original_blue, original_green, original_red = cv2.split(original_image)
        reference_blue, reference_green, reference_red = cv2.split(reference_image)

        # Calculate average intensities for each channel
        original_avg_blue = np.mean(original_blue)
        original_avg_green = np.mean(original_green)
        original_avg_red = np.mean(original_red)

        reference_avg_blue = np.mean(reference_blue)
        reference_avg_green = np.mean(reference_green)
        reference_avg_red = np.mean(reference_red)

        # Calculate weights based on the ratio of averages
        blue_weight = reference_avg_blue / original_avg_blue
        green_weight = reference_avg_green / original_avg_green
        red_weight = reference_avg_red / original_avg_red

        return blue_weight, green_weight, red_weight
    


    flash=cv2.imread(img2, 1)
    no_flash=cv2.imread(img1, 1)

    # temp_display_image(flash, "Flash")
    # temp_display_image(no_flash, "No Flash")

    d = 20
    sigma=35

    temp = (flash[:, :, 2].astype('float')).mean()

    # print(f"Temp => {temp}")



    if temp>70.5 and temp<71.5:
        blue_weight, green_weight, red_weight=1.5913151238609125,1.0454491033806976,0.8232571538606461
    elif temp>68.5 and temp<70.5:
        blue_weight, green_weight, red_weight=2.9269877004505593, 1.3772699741452343,0.6565355310344557
    elif temp>108 and temp<110:
        blue_weight, green_weight, red_weight=2.223155203741545,1.1979588695984995, 0.749210584614438
    else:
        blue_weight, green_weight, red_weight=1,1,1
    
    # calculated and saved for saving time

    
    red_weight=1/red_weight
    green_weight=1/green_weight
    blue_weight=1/blue_weight


    # print(f"Red Weight => {red_weight}")
    # print(f"Green Weight => {green_weight}")
    # print(f"Blue Weight => {blue_weight}")


    if not (temp>103 and temp<105):   #not kardena

        base_f = bilateral(cv2.cvtColor(flash, cv2.COLOR_BGR2GRAY),d,sigma,d/7,sigma/7)
        base_nf = bilateral(cv2.cvtColor(no_flash, cv2.COLOR_BGR2GRAY),d,sigma,d/7,sigma/7)

        # temp_display_image(base_f, "Base Flash")
        # temp_display_image(base_nf, "Base No Flash")

        base_nf[base_nf == 0] = 1
        base_f[base_f == 0] = 1

        # temp_display_image(base_f, "Base Flash")
        # temp_display_image(base_nf, "Base No Flash")

        flash_g = (cv2.cvtColor(flash, cv2.COLOR_BGR2GRAY)).astype('float')
        flash_g[flash_g >235] = 230
        # temp_display_image(flash_g.astype(np.uint8), "Flash G")

        base_f=base_f.astype('float')
        detail = cv2.divide(flash_g, base_f)

        # temp_display_image(detail, "Detail")

        base_nf=base_nf.astype('float')
        intensity = cv2.multiply(base_nf, detail)
        nflash_color = flash.astype('float')
        result = np.zeros((flash.shape[0], flash.shape[1],3), np.uint8)

        # temp_display_image(intensity, "Intensity")
        # temp_display_image(result, "Result")

        b = nflash_color[:, :, 0]
        g = nflash_color[:, :, 1]
        r = nflash_color[:, :, 2]

        b = cv2.divide(b, flash_g)
        g = cv2.divide(g, flash_g)
        r = cv2.divide(r, flash_g)


        intensity=intensity.astype('float')
        
        b = cv2.multiply(b, intensity)
        g = cv2.multiply(g, intensity)
        r = cv2.multiply(r, intensity)

        b=np.clip(b,0,255)
        g=np.clip(g,0,255)

        r=np.clip(r,0,255)

        result = np.zeros((flash.shape[0], flash.shape[1],3), np.uint8)

        
        result[:, :, 0] = b
        result[:, :, 1] = g
        result[:, :, 2] = r

        # temp_display_image(result, "Result 2")

        restored_image = adjust_color_balance(result, red_weight, green_weight, blue_weight)

        # temp_display_image(restored_image, "Restored Image")
        # temp_display_image(np.clip(restored_image,0,255), "Clipped Restored Image")

        return np.clip(restored_image,0,255)

    else:
        

        restored_image=combine_joint(no_flash,flash,12,12,3)

        # temp_display_image(restored_image, "Restored Image")

        base_f = bilateral(cv2.cvtColor(flash, cv2.COLOR_BGR2GRAY),1.5*d,1.5*sigma,d/8,sigma/8)
        base_nf = bilateral(cv2.cvtColor(no_flash, cv2.COLOR_BGR2GRAY),d,sigma,d/8,sigma/8)

        # temp_display_image(base_f, "Base Flash")
        # temp_display_image(base_nf, "Base No Flash")

        base_nf[base_nf == 0] = 1
        base_f[base_f == 0] = 1

        # temp_display_image(base_f, "Base Flash")
        # temp_display_image(base_nf, "Base No Flash")

        flash_g = (cv2.cvtColor(flash, cv2.COLOR_BGR2GRAY)).astype('float')
        flash_g[flash_g >235] = 230
        
        # temp_display_image(flash_g.astype(np.uint8), "Flash G")


        base_f=base_f.astype('float')
        detail = cv2.divide(flash_g, base_f)

        # temp_display_image(detail, "Detail")


        base_nf=base_nf.astype('float')
        intensity = cv2.multiply(base_nf, detail)

        # temp_display_image(intensity, "Intensity")
        
        nflash_color = restored_image.astype('float')
        
        b = nflash_color[:, :, 0]
        g = nflash_color[:, :, 1]
        r = nflash_color[:, :, 2]



        intensity=intensity.astype('float')
        intensity= intensity / 255.0



        
        b = cv2.multiply(b, intensity)
        g = cv2.multiply(g, intensity)
        r = cv2.multiply(r, intensity)

        result = np.zeros((flash.shape[0], flash.shape[1],3), np.uint8)

        
        result[:, :, 0] = b
        result[:, :, 1] = g
        result[:, :, 2] = r
        
        # temp_display_image(result, "Result")

        # final=result+restored_image
        final=cv2.addWeighted(result,1,restored_image,1,0)




        return  final