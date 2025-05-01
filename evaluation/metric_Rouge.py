import nltk
from nltk.translate.bleu_score import sentence_bleu
import sacrebleu
import pandas as pd
import pdb
import os
import numpy as np
# reference_report = "Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. Several lymph nodes are observed in the mediastinum. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; there is a finding consistent with a bulla measuring up to 36 mm in which patchy ground glass densities are observed around the subpleural area in the superior segment of the right lung lower lobe. The findings were evaluated in terms of early viral pneumonia (Covid-19). Clinical laboratory correlation is recommended. No nodular lesions were detected in both lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved."
# generate_report = """
#                     The provided CT scan image presents a detailed view of the thoracic cavity, showcasing the lungs, heart, and surrounding structures. The lungs appear as two dark, rounded structures on either side of the thoracic cavity, with the right lung slightly larger than the left. The heart is visible in the center of the image, positioned between the lungs, and appears as a lighter gray structure.
#                     Upon closer inspection, several abnormalities are noticeable in the image. The right lung exhibits a large, irregularly shaped mass, which appears to be located in the upper lobe. This mass is characterized by its irregular borders and a mix of solid and cystic components. Additionally, the left lung displays a smaller, rounded mass in the lower lobe, which appears to be a solid nodule. The presence of these masses suggests potential abnormalities within the lungs, which may warrant further investigation to determine their nature and significance.
#                     The image also reveals some normal anatomical structures, including the trachea, which is visible as a narrow tube extending from the top of the image towards the heart. The diaphragm is also visible, forming the dome-shaped structure at the base of the lungs. The presence of these structures provides a clear visual representation of the thoracic cavity's anatomy.
#                     In conclusion, the CT scan image reveals several abnormalities in the lungs, including a large, irregularly shaped mass in the right lung and a smaller, solid nodule in the left lung. These findings suggest the need for further evaluation to determine.
#                     """
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
rouge_type = 'rouge2'
print(rouge_type)
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

        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        scores = scorer.score(reference, candidate)
        score = scores[rouge_type]
        scores_.append(score.precision)
        urls.append(name.split("_")[1:3])
        ans.append(score.precision)

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