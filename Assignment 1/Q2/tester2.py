import cv2
import numpy as np
import os
import re
import importlib
import csv
from skimage.metrics import structural_similarity as ssim


def test_student_function(module_name, function_name, label_dict):
    try:
        module = importlib.import_module(module_name)
        student_function = getattr(module, function_name)
    except Exception as e:
        return [((), "", f"Error importing function: {e}", False)]
    subdirectory_path = os.path.join(os.getcwd(), "test")



    
    test_images = [filename for filename in os.listdir(subdirectory_path) if filename.endswith(".mp3")]
    total_score = 0
    is_passed = True
    for test_image_name in test_images:
        try:
            test_image_path = os.path.join(os.path.join(os.getcwd(),'test'),test_image_name)
            output = student_function(test_image_path)
            total_score += (output == label_dict.get(test_image_name, -1))
        except Exception as e:
            result = f"Error: {e}"
            is_passed = False
    if is_passed:
        return total_score
    return 0

def create_label_dict(csv_file):
    label_dict = {}
    with open(csv_file, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            label_dict[row['filename']] = row['label']
    return label_dict

if __name__ == "__main__":
    student_files = [filename for filename in os.listdir() if filename.endswith(".py") and filename != "tester.py"]
    label_dict = create_label_dict('ground truth.csv')
    with open(f"marks.csv","w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Roll Number", "Score"])
        for student_file in student_files:
            try:
                match = re.match(r"Q2_(\d+).py", student_file)              
                if match:
                    roll_number = match.groups()[0]
                    results = test_student_function(student_file[:-3], "solution",label_dict)
                    csv_writer.writerow([roll_number,results])
            except Exception as e:
                print(f"Error testing {student_file}: {e}")
