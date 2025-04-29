import openai
import os
import base64
import pandas as pd
import pdb
import numpy as np
from scipy.stats import spearmanr

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY #'您的 OpenAI API 密钥'

def ask_gpt4o_mini(generated_report, original_report):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
            messages=[
            {"role": "system", "content": "You are a medical evaluator to Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists."},
            {"role": "user","content": [
                    {"type": "text", "text": """
                        Process Overview: 
                        You will be presented with: 
                            1. The criteria for making a judgment.
                            2. The reference radiology report.
                            3. The candidate radiology report.
                            4. The desired format for your assessment
                        """},
                    {"type": "text", "text":"""
                        1. Criteria for Judgment:For each candidate report,determine :- The count of clinically significant errors.The count of clinically insignificant errors.
                        Errors can fall into one of these categories:a)False report of a finding in the candidate.b)Missing a finding present in the reference.C)Misidentification of a finding's anatomic location/position.9)Misassessment of the severity of a finding.e)Mentioning a comparison that isn't in the reference.Omitting a comparison detailing a change from a prior study.)
                        Note: Concentrate on the clinical findings rather than the report's writing styleEvaluate only the findings that appear in both reports.
                        """},
                    {"type": "text", "text": "2. Reference Report:" + original_report},
                    {"type": "text", "text": "3. Candidate Report:" + generated_report},
                    {"type": "text", "text": """ 
                            4. Reporting Your Assessment:Follow this specific format for your output, even if no errors are found:
                            Explanation 
                            <Explanation>
                            
                            [Clinically Significant Errors]:
                            (a) <Error Type>: <The total number of errors>. <Error 1>; <Error 2>;...;<Error n>
                            ...
                            (f) <Error Type>: <The total number of errors>. <Error 1>; <Error 2>; ...; <Error n>
                            
                            [Clinically Insignificant Errors]:
                            (a) <Error Type>: <The total number of errors>. <Error 1>; <Error 2>;...; <Error n>
                            ...
                            (f)'<Error Type>: <The total number of errors>. <Error 1>; <Error 2>; ...; <Error n>
                            [Matched Findings]:
                            <The total number of matched findings>. <Finding 1>; <Finding 2>;...; <Finding n> 
                            [Final score]: score = <The total number of matched findings>/ (<The total number of matched findings> + <The total number of clinically significant errors>)
                            [Result]: score = <score> (accurate to 0.001)
                            """},
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
"./GeneratedReport2/20",
"./GeneratedReport2/30",
'./GeneratedReport2/40',
"./GeneratedReport2/50",
"./GeneratedReport2/60",
"./GeneratedReport2/80"
]
ans_score = []
ans_std = []
print("GREEN_GPT_2")
for evaluate in evaluate_l:
    scores_ = []
    urls = []
    s_std = []
    L_image_url = read_nii_files(evaluate)
    pdb.set_trace()
    L_image_url.sort()
    for i in range(len(L_image_url)):
        image_url = L_image_url[i]
        with open(image_url, 'r', encoding='utf-8') as file:
            generated_report = file.read()
        reference_report = pd.read_csv(reference_path)
        name = image_url.split("/")[5].split(".")[0] + ".nii.gz"
        filtered_data = reference_report[reference_report['VolumeName'] == name]
        original_report = filtered_data['Findings_EN'].tolist()
        
        one_case_score = [0]
        
        for _ in range(3):
            if len(one_case_score) == 2:
                break
            out = ask_gpt4o_mini(generated_report, original_report[0])
            try:
                answer = out.split("[Result]: ")[1].split("score = ")[1][:5]
                score = float(answer)
                one_case_score.append(score)
            except:
                continue
        
        s_std.append(0)
        scores_.append(one_case_score[-1])
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

