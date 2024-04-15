import numpy as np
import cv2
from scipy import signal, interpolate
 



def solution(img1, img2):



    


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
        blurjoint = signal.fftconvolve(joint_data, kernel, mode='same')

        blurweights = signal.fftconvolve(weights, kernel, mode='same')
        blurweights = np.where(blurweights == 0, -2, blurweights)

        normalblurdata = blurdata / blurweights
        normalblurjoint = blurjoint / blurweights

        normalblurdata = np.where(blurweights < -1, 0, normalblurdata)
        normalblurjoint = np.where(blurweights < -1, 0, normalblurjoint)

        (ygrid, xgrid) = np.meshgrid(range(width), range(height))

        dimx = (xgrid / samplespatial) + xypadding
        dimy = (ygrid / samplespatial) + xypadding
        dimz = (image - edgemin) / samplerange + zpadding

        joint_output = interpolate.interpn((range(normalblurdata.shape[0]), range(normalblurdata.shape[1]),
                                            range(normalblurdata.shape[2])), normalblurdata, (dimx, dimy, dimz))

        return joint_output

    
    
    def bilateral(image, sigmaspatial, sigmarange, samplespatial=None, samplerange=None):
        """
        :param image: np.array
        :param sigmaspatial: int
        :param sigmarange: int
        :param samplespatial: int || None
        :param samplerange: int || None
        :return: np.array
        sigmaspatial: An integer representing the standard deviation of the spatial Gaussian filter. It controls the spatial extent of the filter.
        sigmarange: An integer representing the standard deviation of the range Gaussian filter. It controls how much the intensity values can vary.
        Note that sigma values must be integers.

        The 'image' 'np.array' must be given gray-scale. It is suggested that to use OpenCV.
        """

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
        b, g, r = cv2.split(image)
        base_b = bilateral(b,d,sigma)
        base_g = bilateral(g,d,sigma)
        base_r = bilateral(r,d,sigma)

        filtered_image = cv2.merge([base_b, base_g, base_r])
        filtered_image = cv2.convertScaleAbs(filtered_image)

        return filtered_image
    
    def combine_joint(image1,image2):
        b, g, r = cv2.split(image1)
        b1, g1, r1 = cv2.split(image2)


        base_b = joint_bilateral(b,b1,d,sigma)
        base_g = joint_bilateral(g,g1,d,sigma)
        base_r = joint_bilateral(r,r1,d,sigma)

        filtered_image = cv2.merge([base_b, base_g, base_r])

        filtered_image = cv2.convertScaleAbs(filtered_image)

        return filtered_image
    
    
    d = 3
    sigma= 3

        
    flash=cv2.imread(img2, 1)
    no_flash=cv2.imread(img1, 1)
    

    f1=combine(flash).astype(np.float32)

    # print(f1.shape)
    # print(no_flash.shape)
    f1_no_f_joint=combine_joint(no_flash,f1).astype(np.float32)
    f1_no_f_joint_1 = cv2.normalize(f1_no_f_joint, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the image to uint8 type for display
    f1_no_f_joint_1 = np.uint8(f1_no_f_joint_1)


    final=cv2.addWeighted(f1_no_f_joint, 0.8, flash, 0.2, 0,dtype=cv2.CV_32F)


    # flash=cv2.imread(img2, 0)
    # no_flash=cv2.imread(img1, 0)


    # Assuming final is your image to be normalized
    final_normalized = cv2.normalize(final, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the image to uint8 type for display
    final_normalized = np.uint8(final_normalized)

    

    # base_f=bilateral(flash,d,sigma)
    # base_nf=bilateral(no_flash,d,sigma)



    # flash = flash.astype('float')
    # base_f = base_f.astype('float')
    # detail = cv2.divide(flash, base_f)

    # base_nf = base_nf.astype('float')
    # intensity = cv2.multiply(base_nf, detail)

    # no_flash = no_flash.astype('float')
    # nflash_color = cv2.imread(img2, 1)
    # nflash_color = nflash_color.astype('float')
    # b = nflash_color[:, :, 0]
    # g = nflash_color[:, :, 1]
    # r = nflash_color[:, :, 2]

    # b = cv2.divide(b, no_flash)
    # g = cv2.divide(g, no_flash)
    # r = cv2.divide(r, no_flash)

    # intensity=intensity.astype('float')
    # b = cv2.multiply(b, intensity)
    # g = cv2.multiply(g, intensity)
    # r = cv2.multiply(r, intensity)

    # result = np.zeros((flash.shape[0], flash.shape[1],3), np.uint8)
    # result[:, :, 0] = b
    # result[:, :, 1] = g
    # result[:, :, 2] = r
    
    
    return final_normalized

    

                         