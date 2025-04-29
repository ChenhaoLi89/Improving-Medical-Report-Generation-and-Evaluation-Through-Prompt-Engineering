import openai
import os
import base64
from PIL import Image
import numpy as np
from skimage.transform import resize
import pdb
import matplotlib.pyplot as plt
import pandas as pd

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY #'您的 OpenAI API 密钥'

def ask_gpt4o_mini(report, error_rate):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a knowledgeable medical assistant, helping users rewrite a medical report based on the provided report and requirements."},
                {"role": "user","content": [{"type": "text", "text": "The following is the original report:" + report},
                                            {"type": "text", "text": "Please rewrite the report. However, "+ str(error_rate)+"%"+" of the facts need to be wrong. Half of the errors may be output that is contrary to the original report, and the other half may be information that is not in the original report. Your report should be around 200 words."},
                                            ]}
            ],
            max_tokens=1000, 
            temperature=1.3
        )
        answer = response['choices'][0]['message']['content']
        return answer
    except Exception as e:
        print(f"Error: {e}")
        return None

def read_nii_files(directory):
    """
    Retrieve paths of all NIfTI files in the given directory.

    Args:
    directory (str): Path to the directory containing NIfTI files.

    Returns:
    list: List of paths to NIfTI files.
    """
    nii_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                nii_files.append(os.path.join(root, file))
    return nii_files

error_rates = [20, 30, 40, 50, 60, 80]
for error_rate in error_rates:
    reference_path = './CT_RATE/validation_reports.csv'
    L_image_url_npz = read_nii_files("./DRR_preprocessed_GPT/")
    L_image_url_npz.sort()
    # for i in range(len(L_image_url_npz)):
    for i in range(1):
        image_url = L_image_url_npz[i]
        reference_report = pd.read_csv(reference_path)
        name = image_url.split("/")[4].split(".")[0] + ".nii.gz"
        filtered_data = reference_report[reference_report['VolumeName'] == name]
        original_report = filtered_data['Findings_EN'].tolist()
        answer = ask_gpt4o_mini(original_report[0], error_rate)

        save_folder = "./xxxxxxxx/" + str(error_rate)+"/"
        path = image_url
        folder_path_new = os.path.join(save_folder + path.split("/")[2] + '/' + path.split("/")[3])
        os.makedirs(folder_path_new, exist_ok=True)
        file_name = path.split("/")[4].split('.')[0]+'.txt'
        save_path = os.path.join(folder_path_new, file_name)
        with open(save_path, "w", encoding="utf-8") as file:
            file.write(answer)