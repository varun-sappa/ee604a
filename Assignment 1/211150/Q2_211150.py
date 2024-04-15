import cv2
import numpy as np
import librosa

def solution(audio_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    # class_name = 'cardboard'
    # y, sr = librosa.load(audio_path, sr=None)  # sr=None preserves the sampling rate

    # # Parameters for spectrogram calculation
    # n_fft = 2048  # FFT points, adjust as needed
    # hop_length = 512  # Sliding amount for windowed FFT, adjust as needed
    # fmax = 22000  # Maximum frequency to include in the spectrogram, adjust as needed

    # spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=fmax)

    # spec_db = librosa.power_to_db(spec, ref=np.max)
    # spec_gray = (255 * (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())).astype(np.uint8)

    # mask = spec_gray > 0
    # selected_pixels = spec_gray[mask]
    # average_value = np.mean(selected_pixels)

    # print("Average value of pixels greater than 100:", average_value)


    # threshold_value = 0  # Change this threshold value as per your requirement

    # # Threshold the image to convert it to black and white
    # _, filter = cv2.threshold(spec_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # # Find white pixels (pixel value above the white threshold)
    # white_pixels = np.sum(filter > threshold_value)

    # # Find black pixels (pixel value below the black threshold)
    # black_pixels = np.sum(filter < threshold_value)

    # # Calculate the area occupied by white and black regions
    # total_pixels = spec_gray.size
    # white_area_percentage = (white_pixels / total_pixels) * 100
    # black_area_percentage = (black_pixels / total_pixels) * 100

    # print(f"White area: {white_area_percentage}%")
    # print(f"Black area: {black_area_percentage}%")

    # if (average_value>150):
    #     class_name = 'metal'
    # else:
    #     class_name = 'cardboard'
    
    # return class_name
    y, sr = librosa.load(audio_path, sr=None)

    # Parameters for spectrogram calculation
    n_fft = 2048
    hop_length = 512

    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=22000)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    # Flatten the spectrogram into a 1D array
    spec_flat = spec_db.flatten()

    # Calculate a simple metric for classification based on the mean of the flattened spectrogram
    metric = np.median(spec_flat)
    mean = np.mean(spec_flat)
    std_dev = np.std(spec_flat)
    skewness = np.mean(np.power((spec_flat - mean) / std_dev, 3))
    kurt = np.mean(np.power((spec_flat - mean) / std_dev, 4))

    spec_gray = (255 * (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())).astype(np.uint8)
    
    mask = spec_gray > 125
    selected_pixels = spec_gray[mask]
    average_value = np.mean(selected_pixels)

    # print("Average value of pixels greater than 100:", average_value)


    threshold_value = 150  # Change this threshold value as per your requirement

    # Threshold the image to convert it to black and white
    _, filter = cv2.threshold(spec_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Find white pixels (pixel value above the white threshold)
    white_pixels = np.sum(filter > threshold_value)

    # Find black pixels (pixel value below the black threshold)
    black_pixels = np.sum(filter < threshold_value)
    # print(black_pixels)
    # print(white_pixels)

    # Calculate the area occupied by white and black regions
    total_pixels = spec_gray.size
    white_area_percentage = (white_pixels / total_pixels) * 100
    black_area_percentage = (black_pixels / total_pixels) * 100

    spec_flat = spec_gray.flatten()
    metric = np.median(spec_flat)
    mean = np.mean(spec_flat)
    std_dev = np.std(spec_flat)
    skewness = np.mean(np.power((spec_flat - mean) / std_dev, 3))
    kurt = np.mean(np.power((spec_flat - mean) / std_dev, 4))


    top_percentage = 0.1
    spec_flat_sorted = np.sort(spec_flat)
    num_values_to_keep = int(top_percentage * len(spec_flat_sorted))
    top_x_percent_values = spec_flat_sorted[-num_values_to_keep:]
    mean1 = np.mean(top_x_percent_values)
    std_dev1 = np.std(top_x_percent_values)
    metric1 = np.median(top_x_percent_values)



    # print(average_value,metric, mean, std_dev, skewness, kurt, mean1, std_dev1,metric1)

    final=average_value+std_dev
    # print(f"White area: {white_area_percentage}%")
    # print(f"Black area: {black_area_percentage}%")
    # print(final)
    # threshold = 19.0  # Adjust as needed
    # print(std_dev)

    # if (std_dev >= threshold):
    #     class_name = 'metal'
    # else:
    #     class_name = 'cardboard'
    
    # return class_name

    if (final>225):
        class_name = 'metal'
    else:
        class_name = 'cardboard'
    
    return class_name



