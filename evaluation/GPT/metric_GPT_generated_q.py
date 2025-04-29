import openai
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import os
import pdb
import numpy as np
from skimage.transform import resize
import pandas as pd


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY #'您的 OpenAI API 密钥'


def gpt4o_eval(generated_report, original_report, question):
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
                            2. The common questions list.
                            3. The reference report.
                            4. The candidate report.
                            5. The desired format for your assessment
                        """},
                    {"type": "text", "text":"""
                    1. Criteria for evaluation: 
                        For each question in the common question list, please find the corresponding description in the reference report and candidate report. 
                        If there is a corresponding description in both reports, continue. If not, score 0. 
                        Furthermore, if the description has a consistent tendency (both believe that the disease exists or both believe that the disease does not exist), score 1. 
                        If the description of the disease content is similar, score 2.
                        For each question, the score should be a number among 0, 1, 2.
                        We have a total of 15 questions, and you need to output the sum of the scores in the end.
                        """},
                    {"type": "text", "text": "2. The common questions list:" + question},
                    {"type": "text", "text": "3. The reference report:" + original_report},
                    {"type": "text", "text": "4. The candidate report:" + generated_report},
                    {"type": "text", "text": """ 
                            5. Reporting Your Assessment: Follow this specific format for your output:
                            [Reasons and Score for Each Qustion]: <reason for each qustion> <score for each qustion> (detailed reasons for giving such score for each question).
                            [Total Score]: <sum of scores for 15 questions> 
                            """}]}
            ],
            max_tokens=2000, 
            temperature=0.7 
        )
        answer = response['choices'][0]['message']['content']
        return answer
    except Exception as e:
        print(f"Error: {e}")
        return None

def ask_gpt_generated_q(report):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a medical professional."},
                {"role": "user","content": [
                                            {"type": "text", "text": "The following is a medical report. Could you please tell me 15 parts and organs are generally included in this chest medical report?"},
                                            {"type": "text", "text": "Next is the report:" + report},
                                            {"type": "text", "text": "The answer should follow this format: Part 1:..., Part 2:... Do not return anything except the 15 parts."}
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

question_list = ["1.Trachea and Main Bronchi - Assessment for patency (openness) and any occlusive pathology.",
                "2.Lungs - Evaluation of aeration, nodules, infiltrative lesions, consolidations, ground-glass opacities, and pleural irregularities.",
                "3.Pleura - Presence of pleural effusion or thickening.",
                "4.Mediastinal Structures - Includes lymph nodes and other mediastinal components.",
                "5.Heart and Pericardium - Assessment of heart contour, size, and pericardial effusion or thickening.",
                "6.Thoracic Aorta and Main Vascular Structures - Checking for any dilation or abnormal calibration.",
                "7.Thoracic Esophagus - Evaluation for normal calibration and any significant wall thickening.",
                "8.Mediastinal and Hilar Lymph Nodes - Presence of enlarged or pathological lymph nodes.",
                "9.Liver - Presence of lesions or any space-occupying abnormalities.",
                "10.Kidneys - Detection of stones (calculi) or abnormalities.",
                "11.Adrenal Glands - Presence or absence of any space-occupying lesion.",
                "12.Bone Structures (Thoracic Vertebrae, Ribs, Sternum, Clavicles, Scapulae) - Evaluation for any lytic-destructive lesions or fractures.",
                "13.Thyroid - Presence of nodules, enlargement, or heterogeneity.",
                "14.Post-treatment Changes - Observed in some cases, including pleural contour irregularities, subcutaneous changes, and residual lesions.",
                "15.Others - If find any other abnormality, its description is included."]
question_txt = ""
for ques in question_list:
    question_txt = question_txt + ques + "\n"



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
print("generated_question")
for i in range(len(evaluate_l)):
    evaluate = evaluate_l[i]
    L_image_url = read_nii_files(evaluate)
    L_image_url.sort()
    scores_ = []
    from tqdm import tqdm
    for i in tqdm(range(len(L_image_url))):
        image_url = L_image_url[i]
        with open(image_url, 'r', encoding='utf-8') as file:
            generated_report = file.read()  
        name = image_url.split("/")[5].split(".")[0] + ".nii.gz"
        # multi_questions, multi_ans = load_vqa(name)
        candidate = generated_report
        
        reference_report = pd.read_csv(reference_path)
        filtered_data = reference_report[reference_report['VolumeName'] == name]
        original_report = filtered_data['Findings_EN'].tolist()[0]
        
        question_list = []
        while len(question_list) != 15:
            question_txt = ask_gpt_generated_q(original_report)
            question_list = question_txt.split("Part")[1:]
        print(question_txt)
        
        one_case_score = []
        for i in range(100):
            if len(one_case_score) == 1:
                break
            score_reason = gpt4o_eval(candidate, original_report, question_txt)
            print(score_reason)
            temp = score_reason.split("[Total Score]:")[1]
            try:
                score = float(temp)
                one_case_score.append(score)
            except:
                continue
        if one_case_score == []:
            one_case_score.append(0)

        scores_.append(one_case_score[0])
    ans_score.extend(scores_)
for k in ans_score:
    print(k)

print("*****************")
from scipy.stats import spearmanr
x = np.array(ans_score)
y_2 = []
for i in range(6, 0, -1):
    y_2 = y_2 + [i for _ in range(47)] 
y = np.array(y_2)
spearman_corr, p_value = spearmanr(x, y)
print(spearman_corr)
