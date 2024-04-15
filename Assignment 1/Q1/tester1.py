import cv2
import numpy as np
import os
import re
import importlib
import csv
from skimage.metrics import structural_similarity as ssim


def test_student_function(module_name, function_name):
    try:
        module = importlib.import_module(module_name)
        student_function = getattr(module, function_name)
    except Exception as e:
        return [((), "", f"Error importing function: {e}", False)]
    subdirectory_path = os.path.join(os.getcwd(), "test")

    test_images = [filename for filename in os.listdir(subdirectory_path) if filename.endswith(".png")]
    total_score = 0
    is_passed = True
    for test_image_name in test_images:
        try:
            test_image_path = os.path.join(os.path.join(os.getcwd(),'test'),test_image_name)
            output_image = student_function(test_image_path)
            ground_truth_image=cv2.imread(os.path.join(os.path.join(os.getcwd(),'ground truth'),test_image_name))
            score = ssim(ground_truth_image,output_image,channel_axis=-1)
            total_score += score
            print(score)
            print(test_image_name)

        except Exception as e:
            result = f"Error: {e}"
            is_passed = False
    if is_passed:
        return total_score
    return 0


if __name__ == "__main__":
    student_files = [filename for filename in os.listdir() if filename.endswith(".py") and filename != "tester.py"]
    with open(f"marks.csv","w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        
        csv_writer.writerow(["Roll Number", "Score"])
        for student_file in student_files:
            try:
                match = re.match(r"Q1_(\d+).py", student_file)   
                if match:
                    roll_number = match.groups()[0]
                    results = test_student_function(student_file[:-3], "solution")
                    csv_writer.writerow([ roll_number,results])
            except Exception as e:
                print(f"Error testing {student_file}: {e}")
