import nltk
from nltk.translate.meteor_score import meteor_score
import pandas as pd
import pdb
import os
import numpy as np
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
ans = []
for evaluate in evaluate_l:
    print(evaluate)
    L_image_url = read_nii_files(evaluate)
    L_image_url.sort()
    scores_ = []
    urls = []
    for i in range(len(L_image_url)):
        image_url = L_image_url[i]
        with open(image_url, 'r', encoding='utf-8') as file:
                generated_report = file.read()
        reference_report = pd.read_csv(reference_path)
        name = image_url.split("/")[5].split(".")[0] + ".nii.gz"
        filtered_data = reference_report[reference_report['VolumeName'] == name]
        original_report = filtered_data['Findings_EN'].tolist()

        reference = original_report[0]
        candidate = generated_report

        score = meteor_score([reference.split()], candidate.split())
        scores_.append(score)
        urls.append(name.split("_")[1:3])
        ans.append(score)
        
print("*****************")
from scipy.stats import spearmanr
x = np.array(ans)
y_2 = []
for i in range(6, 0, -1):
    y_2 = y_2 + [i for _ in range(47)] 
y = np.array(y_2)
spearman_corr, p_value = spearmanr(x, y)
print(spearman_corr)
print(p_value)
