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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY #'您的 OpenAI API 密钥'

def llama_infer(generated_report, original_report):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a medical evaluator to evaluate clinical accuracy for AI-generated reports. "},
                {"role": "user","content": [{"type": "text", "text": """Given a machine-generated medical report and a corresponding reference report, compare the content of the two and evaluate their similarity based on the following criteria. Provide a similarity score from 0 to 10. 10 represents the generated response is fully aligned with the ground truth—completely accurate, relevant, comprehensive, clear, and clinically sound. 5 indicates that the generated response contains noticeable errors or omissions that could impact clinical understanding, though some correct information is present. 0 means the generated response is completely incorrect or irrelevant, offering no clinical value.
                                                                                Evaluation Criteria:
                                                                                Content Accuracy: Check if the descriptions in both reports are consistent, particularly in terms of lesion location, shape, size, and other imaging characteristics.
                                                                                Terminology Consistency: Verify that medical terminology is used in a standardized and consistent manner.
                                                                                Structure and Formatting: Evaluate if the structure of both reports is aligned, including paragraph division, logical sequence, etc.
                                                                                Conciseness and Clarity: Ensure the descriptions in the report are clear and concise, avoiding unnecessary information.
                                                                                """},
                                            {"type": "text", "text": "The following is the AI-generated report." + generated_report},
                                            {"type": "text", "text": "Next is the reference report." + original_report},
                                            {"type": "text", "text": "Output: Provide a similarity score (0-10, accurate to 0.1). Please do not output anything other than the similarity score."},
                                            ]}
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

evaluate_l = [
"./GeneratedReport_worst/20",
"./GeneratedReport_worst/30",
'./GeneratedReport_worst/40',
"./GeneratedReport_worst/50",
"./GeneratedReport_worst/60",
"./GeneratedReport_worst/80"
]
ans_score = []
ans_std = []
print("Baseline")
for evaluate in evaluate_l:
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
        
        # original_report = ["Trachea and lumen of both main bronchi are open. No occlusive pathology was detected in the trachea and lumen of both main bronchi. Mediastinal structures were evaluated as suboptimal since the examination was unenhanced. As far as can be observed: A 29 mm diameter hypodense nodule was observed in the right thyroid lobe. Calibration of thoracic main vascular structures is natural. No dilatation was detected in the thoracic aorta. Heart contour size is natural. Pericardial thickening-effusion was not detected. Thoracic esophagus calibration was normal and no significant pathological wall thickening was detected. Mediastinal millimetric lymph nodes were observed. When examined in the lung parenchyma window; In both lungs, ground-glass density increases that are widespread in the upper and lower lobes, tending to coalesce in the peripheral subpleural area and peribronchovascular localization, and consolidative areas in the lower lobes are observed. There are imaging features that are frequently reported in Covid-19 pneumonia. Clinical - laboratory correlation is recommended. In the upper abdominal sections included in the examination area, two millimeter-sized hypodense lesions that could not be characterized in this examination were observed in the posterior right lobe of the liver. A 2.5 mm diameter calculus was observed in the middle zone of the right kidney. No lytic-destructive lesion was detected in bone structures."]
        # generated_report = "The image is a chest CT scan, which provides detailed information about the structures within the chest cavity. In this particular image, the mediastinal structures were evaluated as suboptimal, meaning that they were not clearly visible or well-defined. However, a 32 mm diameter hypodense nodule was observed in the right thyroid lobe. The trachea and lumen of both main bronchi were found to be open, with no occlusive pathology detected. The diameter of the main pulmonary artery, right pulmonary artery, and left pulmonary artery were measured, showing fusiform dilatation. The heart size increased, and pericardial thickening-effusion was not detected. Calcific atherosclerotic changes were observed in the thoracic aorta and coronary artery walls. The thoracic esophagus calibration was normal, and no significant pathological wall thickening was detected. Multiple lymph nodes measuring 22 mm in the short axis of the largest were observed in the mediastinal, upper-lower paratracheal, aorticopulmonary window, prevascular area, and subcarinal area. When examined in the lung parenchyma window, parenchymal fibrosis areas causing structural distortion in both lungs, emphysematous changes, prominence in interlobular septa, and honeycomb appearance were observed, along with frosted glass-like density increases. Traction bronchiectasis were present in both lungs, and peribronchial thickenings were observed in both lungs. Bilateral pleural thickening-effusion was not detected. No gall bladder was observed in the upper abdominal sections included in the examination area. Subcapsular parenchymal calcifications with a diameter of 1 cm were observed in the posterior right lobe of the liver. Thoracic kyphosis has increased, and degenerative changes were observed in bone structures."
        one_case_score = []
        
        for _ in range(10):
            if len(one_case_score) == 1:
                break
            answer = llama_infer(generated_report, original_report[0])
            try:
                score = float(answer[:3])
                one_case_score.append(score)
            except:
                continue
    
        s_std.append(np.std(one_case_score))
        scores_.append(np.mean(one_case_score))
        urls.append(name.split("_")[1:3])
    ans_score.extend(scores_)
    ans_std.extend(s_std)
for k in ans_score:
    print(k)
    
print("#########")
for k in ans_std:
    print(k)
print("###########")
print(np.std(ans_score))
print(np.mean(ans_std))

print("*****************")
from scipy.stats import spearmanr
x = np.array(ans_score)
y_2 = []
for i in range(6, 0, -1):
    y_2 = y_2 + [i for _ in range(47)] 
y = np.array(y_2)
spearman_corr, p_value = spearmanr(x, y)
print(spearman_corr)
print(p_value)
