import openai
import os
import base64
import pandas as pd
import pdb
import numpy as np

import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from skimage.transform import resize

def llama_infer(generated_report, original_report):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a medical evaluator to evaluate clinical accuracy for AI-generated reports."},
                {"role": "user","content": [
                            {"type": "text", "text": """
                                Process Overview: 
                                You will be presented with: 
                                    1. The criteria for evaluation.
                                    2. The reference report.
                                    3. The candidate report.
                                    4. The desired format for your assessment
                                """},
                            {"type": "text", "text":"""
                            1. Criteria for prompt: 
                                Content Accuracy: Check if the descriptions in both reports are consistent, particularly in terms of lesion location, shape, size, and other imaging characteristics.
                                Terminology Consistency: Verify that medical terminology is used in a standardized and consistent manner.
                                Structure and Formatting: Evaluate if the structure of both reports is aligned, including paragraph division, logical sequence, etc.
                                Conciseness and Clarity: Ensure the descriptions in the report are clear and concise, avoiding unnecessary information.
                                """},
                            {"type": "text", "text": "2. The reference report:" + original_report},
                            {"type": "text", "text": "3. The candidate report:" + generated_report},
                            {"type": "text", "text": """ 
                                    4. Reporting Your Assessment: Follow this specific format for your output:
                                    **Final Score:** <score> (0-1, accurate to 0.001)
                                    **Reasons:** <reasons> (detailed reasons for giving such score)
                                    """}]}
                ],
            max_tokens=1000, 
            temperature=0.7 
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
            if file.endswith('.txt'):
                nii_files.append(os.path.join(root, file))
    return nii_files

def calculate_average(scores, urls):
    dic = {}

    for i in range(len(scores)):
        if str(urls[i]) not in dic:
            dic[str(urls[i])] = [scores[i]]
        else:
            dic[str(urls[i])].append(scores[i])
    averages = {}
    for key, values in dic.items():
        averages[key] = sum(values) / len(values) if values else 0
        print(f"{sum(values) / len(values):.6f}")
    return averages

reference_path = './CT_RATE/validation_reports.csv'


ans_score = []
ans_std = []
print("GPT_Baseline")
evaluate_l = [10]
for k in evaluate_l:
    evaluate = "./Baseline_result_qa_gpt_new/llama_infer" + str(k)
    scores_ = []
    urls = []
    s_std = []
    print("#########")
    print(evaluate)
    L_image_url = read_nii_files(evaluate)
    L_image_url.sort()
    from tqdm import tqdm
    for i in tqdm(range(len(L_image_url))):
        image_url = L_image_url[i]
        # print(image_url)
        with open(image_url, 'r', encoding='utf-8') as file:
                generated_report = file.read()
        reference_report = pd.read_csv(reference_path)
        name = image_url.split("/")[5].split(".")[0] + ".nii.gz"
        filtered_data = reference_report[reference_report['VolumeName'] == name]
        original_report = filtered_data['Findings_EN'].tolist()
        
        one_case_score = []
        
        for _ in range(10):
            if len(one_case_score) == 1:
                break
            answer = llama_infer(generated_report, original_report[0])
            try:
                answer = answer.split("**Final Score:**")[1][:5]
                score = float(answer)
                one_case_score.append(score)
            except:
                continue
    
        s_std.append(np.std(one_case_score))
        scores_.append(np.mean(one_case_score))
        urls.append(name.split("_")[1:3])
    print(np.mean(scores_))
    ans_score.append(np.mean(scores_))
for k in ans_score:
    print(k)
    

