import cv2
import numpy as np
import os
import re
import importlib
import csv

def dice_coefficient(prediction, ground_truth):
  """
  Calculates the Dice coefficient between two segmentation masks.

  Args:
    prediction: The predicted segmentation mask.
    ground_truth: The ground truth segmentation mask.

  Returns:
    The Dice coefficient.
  """
  matched_pixels = 0
  total_prediction_pixels = prediction.shape[0] * prediction.shape[1]
  total_ground_truth_pixels = ground_truth.shape[0] * ground_truth.shape[1]
  for i in range(prediction.shape[0]):
      for j in range(prediction.shape[1]):
          if np.all(prediction[i][j] == ground_truth[i][j]):
              matched_pixels += 1
  return 2*(matched_pixels) / ( total_prediction_pixels + total_ground_truth_pixels)

def test_student_function(module_name, function_name):
    try:
        module = importlib.import_module(module_name)
        student_function = getattr(module, function_name)
    except Exception as e:
        return [((), "", f"Error importing function: {e}", False)]
    subdirectory_path = os.path.join(os.getcwd(), "test")

    test_images = [filename for filename in os.listdir(subdirectory_path) if filename.endswith(".jpg")]
    total_score = 0.0
    is_passed = False
    i =0
    for test_image_name in test_images:
        i+=1
        try:
            test_image_path = os.path.join(os.path.join(os.getcwd(),'test'),test_image_name)
            output_image = student_function(test_image_path)
            ground_truth_image=cv2.imread(os.path.join(os.path.join(os.getcwd(),'ground truth'),test_image_name))
            # print(ground_truth_image)
            score = dice_coefficient(output_image, ground_truth_image)
            # print(f"Test image name: {test_image_name} and score: {score}")
            total_score += score
            print(score)
            is_passed=True

        except Exception as e:
            result = f"Error: {e}"
    if is_passed:
        print(total_score)
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
                    print("Roll Number", roll_number)
                    results = test_student_function(student_file[:-3], "solution")
                    csv_writer.writerow([ roll_number,results])
            except Exception as e:
                print(f"Error testing {student_file}: {e}")
