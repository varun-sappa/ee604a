import cv2
import numpy as np
import os
import re
import importlib
import shutil

def test_student_function(module_name, function_name,roll_number):
    try:
        module = importlib.import_module(module_name)
        student_function = getattr(module, function_name)
    except Exception as e:
        return [((), "", f"Error importing function: {e}", False)]
    subdirectory_path = os.path.join(os.getcwd(), "test")
    
    test_images = [filename for filename in os.listdir(subdirectory_path) if filename.endswith(".png")]
    try:

        shutil.rmtree(os.path.join(os.getcwd(), roll_number))
        os.mkdir(roll_number)
    except OSError as ex:
        os.mkdir(roll_number)

    
    for test_image_name in test_images:
        # try:
        test_image_path = os.path.join(os.path.join(os.getcwd(),'test'),test_image_name)
        output_image = student_function(test_image_path)
        output_path = os.path.join(roll_number, test_image_name)
        cv2.imwrite(output_path, output_image)
    return 0

if __name__ == "__main__":
    student_files = [filename for filename in os.listdir() if filename.endswith(".py") and filename != "tester.py"]
    for student_file in student_files:
        try:
            match = re.match(r"Q3_(\d+).py", student_file)              
            if match:
                roll_number= match.groups()[0]
                results = test_student_function(student_file[:-3], "solution",roll_number)
        except Exception as e:
                print(f"Error testing {student_file}: {e}")
