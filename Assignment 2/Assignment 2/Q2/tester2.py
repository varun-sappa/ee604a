from hmac import new
import cv2
import numpy as np
import os
import re
import importlib
import shutil
import time
import csv

def test_student_function(module_name, function_name,roll_number):
    try:
        module = importlib.import_module(module_name)
        student_function = getattr(module, function_name)
    except Exception as e:
        return [((), "", f"Error importing function: {e}", False)]
    subdirectory_path = os.path.join(os.getcwd(), "ultimate_test")
    
    test_images = [filename for filename in os.listdir(subdirectory_path) if filename.endswith("a.jpg")]
    try:
        shutil.rmtree(os.path.join(os.getcwd(), roll_number))
        os.mkdir(roll_number)
    except OSError as ex:
        os.mkdir(roll_number)
        
    total_time = 0
    for test_image_name in test_images:
        # try:
        test_image_path_a = os.path.join(os.path.join(os.getcwd(),'ultimate_test'),test_image_name)
        base, ext = test_image_name.rsplit('.', 1)
        base = base[:len(base)-2]
        # Create the new filename with the updated suffix "b"
        new_filename = f"{base}_b.{ext}"
        print(new_filename)
        test_image_path_b = os.path.join(os.path.join(os.getcwd(),'ultimate_test'),new_filename)
        start_time = time.time()
        output_image = student_function(test_image_path_a,test_image_path_b)
        end_time = time.time()
        output_path = os.path.join(roll_number, f"{base}.{ext}")
        cv2.imwrite(output_path, output_image)
        total_time = end_time - start_time
        print(total_time)
    return total_time

if __name__ == "__main__":
    student_files = [filename for filename in os.listdir() if filename.endswith(".py") and filename != "tester3.py"]
    with open(f"marks.csv","w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        
        csv_writer.writerow(["Roll Number", "Score"])
        for student_file in student_files:
            try:
                match = re.match(r"Q2_(\d+).py", student_file)              
                if match:
                    roll_number= match.groups()[0]
                    print("Roll Number", roll_number)
                    results = test_student_function(student_file[:-3], "solution",roll_number)
                    csv_writer.writerow([roll_number,results])
            except Exception as e:
                    print(f"Error testing {student_file}: {e}")
