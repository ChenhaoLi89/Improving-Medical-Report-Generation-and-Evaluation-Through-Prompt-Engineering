import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import os
import pdb
import numpy as np
from skimage.transform import resize
import pandas as pd
import openai
import base64

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY #'您的 OpenAI API 密钥'


default_prompt = """Below are requirements for generating a paragraph of report: 
                                            Avoid referring to specific facts, terms, abbreviations, dates, numbers, or names.
                                            Focus on the visual aspects of the images that can be helpful for doctors and patients. 
                                            Walk through the important details of the images and analyze the images in a comprehensive and detailed manner.
                                            Ensure to answer question: What findings do you observe in this CT scan?
                                            If you find something abnormal, ensure to answer question: What abnormalities are present in this CT scan?
                                            Answer responsibly, avoiding overconfidence, and do not provide medical advice or diagnostic information. 
                                            The report should be a paragraph of about 200 words."""
def gpt4o(conversation):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
            messages=conversation,
            max_tokens=1000, 
            temperature=0.7 
        )
        answer = response['choices'][0]['message']['content']
        return answer
    except Exception as e:
        print(f"Error: {e}")
        return None

def gpt4o_infer(image, prompt = default_prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a knowledgeable medical assistant, helping users generate a paragraph of medical report based on the provided image."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                    {"type": "text", "text": prompt}
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
                        If there is a corresponding description in both reports, score 1. If not, score 0. 
                        Furthermore, if the description has a consistent tendency (both believe that the disease exists or both believe that the disease does not exist), score 2. 
                        If the description of the disease content is similar, score 3.
                        For each question, the score should be a number among 0, 1, 2, and 3.
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

def gpt4o_reflect(image, generated_report, score_reason, prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a knowledgeable medical assistant, helping users generate a prompt to better generate medical report."},
                {"role": "user","content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                    {"type": "text", "text": """
                        Process Overview: 
                        You will be presented with: 
                            0. The reference medical image.
                            1. The criteria for generating a prompt.
                            2. The original prompt.
                            3. The candidate report generated based on the original prompt.
                            4. Another large language model evaluation score and reasons based on the reference report.
                            5. The desired format for your assessment
                        """},
                    {"type": "text", "text":"""
                        1. Criteria for prompt: 
                        You need to update the prompt based on the information we provide, so that you can generate better report based on image and prompt.
                        When we need to generate a report, we will input the image and the prompt you generated into the large language model. This picture will be different from the one provided now, but it is also a medical image of the patient's chest.
                        Note: The prompt should not contain a specific description of the symptom, but can contain content that guides big-picture thinking, such as what parts the report should contain, etc.
                        """},
                    {"type": "text", "text": "2. The original prompt:" + prompt},
                    {"type": "text", "text": "3. The candidate report generated based on the original prompt:" + generated_report},
                    {"type": "text", "text": "4. Another large language model evaluation score and reasons based on the reference report:" + score_reason},
                    {"type": "text", "text": """ 
                            5. Reporting Your Assessment: Follow this specific format for your output:
                            [New prompt]:
                            <new prompt>
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
            if file.endswith('.png'):
                nii_files.append(os.path.join(root, file))
    return nii_files

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
    
    
L_image_url_npz = read_nii_files("./DRR_preprocessed_GPT_train/")
L_image_url_npz.sort()
prompt = default_prompt
from tqdm import tqdm
print("gpt_gqa")
conversation = []
for i in tqdm(range(3)):
    image_url_npz = L_image_url_npz[i]
    with open(image_url_npz, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    reference_path = './CT_RATE/validation_reports.csv'
    reference_report = pd.read_csv(reference_path)
    name = image_url_npz.split("/")[4].split(".")[0] + ".nii.gz"
    filtered_data = reference_report[reference_report['VolumeName'] == name]
    original_report = filtered_data['Findings_EN'].tolist()[0]
    
    temp = [{"role": "system", "content": "You are a knowledgeable medical assistant, helping users generate a paragraph of medical report based on the provided image."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                {"type": "text", "text": prompt}
            ]}]
    conversation.extend(temp)
    
    generated_report = gpt4o_infer(encoded_image, prompt)
    print(generated_report)
    
    temp = [{"role": "assistant", "content": generated_report}]
    conversation.extend(temp)
    
    temp = [{"role": "system", "content": "You are a medical professional."},
            {"role": "user","content": [
                {"type": "text", "text": "The following is a medical report. Could you please tell me 15 parts and organs are generally included in this chest medical report?"},
                {"type": "text", "text": "Next is the report:" + original_report},
                {"type": "text", "text": "The answer should follow this format: Part 1:..., Part 2:... Do not return anything except the 15 parts."}]}
            ]
    conversation.extend(temp)
    
    question_list = []
    while len(question_list) != 15:
        question_txt = ask_gpt_generated_q(original_report)
        question_list = question_txt.split("Part")[1:]
    print(question_txt)
    temp = [{"role": "assistant", "content": question_txt}]
    conversation.extend(temp)
    
    temp = [{"role": "system", "content": "You are a medical evaluator to evaluate clinical accuracy for AI-generated reports."},
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
                    If there is a corresponding description in both reports, score 1. If not, score 0. 
                    Furthermore, if the description has a consistent tendency (both believe that the disease exists or both believe that the disease does not exist), score 2. 
                    If the description of the disease content is similar, score 3.
                    For each question, the score should be a number among 0, 1, 2, and 3.
                    We have a total of 15 questions, and you need to output the sum of the scores in the end.
                    """},
                {"type": "text", "text": "2. The common questions list:" + question_txt},
                {"type": "text", "text": "3. The reference report:" + original_report},
                {"type": "text", "text": "4. The candidate report:" + generated_report},
                {"type": "text", "text": """ 
                        5. Reporting Your Assessment: Follow this specific format for your output:
                        [Reasons and Score for Each Qustion]: <reason for each qustion> <score for each qustion> (detailed reasons for giving such score for each question).
                        [Total Score]: <sum of scores for 15 questions> 
                        """}]}
            ]
    conversation.extend(temp)
    score_reason = gpt4o_eval(generated_report, original_report, question_txt)
    print(score_reason)
    temp = [{"role": "assistant", "content": score_reason}]
    conversation.extend(temp)

    if i == i:
        L_image_url = read_nii_files("./DRR_preprocessed_GPT_test/")
        L_image_url.sort()
        for j in range(30):
            image_url = L_image_url[j]
            with open(image_url, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
            conversation2 = conversation.copy()
            temp2 = [{"role": "system", "content": "You are a knowledgeable medical assistant, helping users generate a paragraph of medical report based on the provided image."},
                     {"role": "user", "content": [{"type": "text", "text": "The above is one or several examples, which show how the large language model generates a medical report based on the medical image and the whole process of evaluating it. Now I will give you a new image. Can you generate a report based on the new picture and the above example to make the report score higher in the evaluation? Please note that different images have different reference reports, and you cannot directly copy the report in the example."},
                                                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}]} 
                    ]
            conversation2.extend(temp2)
            answer = gpt4o(conversation2)
            
            save_folder = "./Baseline_result_gqa_gpt_combine2/llama_infer"+str(i+1)+"/" 
            path = image_url
            folder_path_new = os.path.join(save_folder + path.split("/")[2] + '/' + path.split("/")[3])
            os.makedirs(folder_path_new, exist_ok=True)
            file_name = path.split("/")[4].split('.')[0]+'.txt'
            save_path = os.path.join(folder_path_new, file_name)
            with open(save_path, "w", encoding="utf-8") as file:
                file.write(answer)
            import time
            time.sleep(30)
    

    