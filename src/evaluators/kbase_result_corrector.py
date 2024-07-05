"""
Date: 05/09/2024



"""

import os
import json
from itertools import chain
from evaluator import *


result_dict = {
    "falcon-7b (original)": "/home/ac.gpark/BioIE-LLM-WIP/result/Falcon/falcon-7b/kbase/entity_type/entity_type_result_2024-02-14 23:36:24.886072.txt",
    "falcon-7b (finetued)": "/home/ac.gpark/BioIE-LLM-WIP/result/Falcon/falcon-7b/kbase/entity_type/entity_type_result_2024-04-17 01:16:49.546694.txt",
    
    "falcon-40b (original)": "/home/ac.gpark/BioIE-LLM-WIP/result/Falcon/falcon-40b/kbase/entity_type/entity_type_result_2024-02-14 23:52:05.283876.txt",
    "falcon-40b (finetued)": "/home/ac.gpark/BioIE-LLM-WIP/result/Falcon/falcon-40b/kbase/entity_type/entity_type_result_2024-04-17 22:20:10.325288.txt",
    
    "mpt-7b-chat (original)": "/home/ac.gpark/BioIE-LLM-WIP/result/MPT/mpt-7b-chat/kbase/entity_type/entity_type_result_2024-02-14 22:07:58.733506.txt",
    "mpt-7b-chat (finetued)": "/home/ac.gpark/BioIE-LLM-WIP/result/MPT/mpt-7b-chat/kbase/entity_type/entity_type_result_2024-04-15 19:30:36.362328.txt",
    
    "mpt-30b-chat (original)": "/home/ac.gpark/BioIE-LLM-WIP/result/MPT/mpt-30b-chat/kbase/entity_type/entity_type_result_2024-02-14 22:19:32.110251.txt",
    "mpt-30b-chat (finetued)": "/home/ac.gpark/BioIE-LLM-WIP/result/MPT/mpt-30b-chat/kbase/entity_type/entity_type_result_2024-04-16 16:11:31.485251.txt",
    
    "Llama-2-7b-chat (original)": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-2/Llama-2-7b-chat-hf/kbase/entity_type/entity_type_result_2024-04-12 15:21:19.328007.txt",
    "Llama-2-7b-chat (finetued)": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-2/Llama-2-7b-chat-hf/kbase/entity_type/entity_type_result_2024-04-15 20:11:33.867255.txt",
    
    "Llama-2-70b-chat (original)": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-2/Llama-2-70b-chat-hf/kbase/entity_type/entity_type_result_2024-02-14 22:37:27.033524.txt",
    "Llama-2-70b-chat (finetued)": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-2/Llama-2-70b-chat-hf/kbase/entity_type/entity_type_result_2024-04-16 11:38:09.748233.txt",
    
    "SOLAR-10.7B-Instruct (original)": "/home/ac.gpark/BioIE-LLM-WIP/result/Solar/SOLAR-10.7B-Instruct-v1.0/kbase/entity_type/entity_type_result_2024-02-14 23:20:56.087891.txt",
    "SOLAR-10.7B-Instruct (finetued)": "/home/ac.gpark/BioIE-LLM-WIP/result/Solar/SOLAR-10.7B-Instruct-v1.0/kbase/entity_type/entity_type_result_2024-04-16 21:13:19.742759.txt",
    
    "Mistral-7B-Instruct (original)": "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mistral-7B-Instruct-v0.2/kbase/entity_type/entity_type_result_2024-02-14 19:53:00.278952.txt",
    "Mistral-7B-Instruct (finetued)": "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mistral-7B-Instruct-v0.2/kbase/entity_type/entity_type_result_2024-04-15 19:51:53.671412.txt",
    
    "Mixtral-8x7B-Instruct (original)": "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mixtral-8x7B-Instruct-v0.1/kbase/entity_type/entity_type_result_2024-02-14 20:13:14.734474.txt",
    "Mixtral-8x7B-Instruct (finetued)": "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mixtral-8x7B-Instruct-v0.1/kbase/entity_type/entity_type_result_2024-04-14 01:01:37.942934.txt",
    
    "Llama-3-8b (original)": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-3/Meta-Llama-3-8B/kbase/entity_type/entity_type_result_2024-04-19 19:48:50.323099.txt",
    "Llama-3-8b (finetued)": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-3/Meta-Llama-3-8B/kbase/entity_type/entity_type_result_2024-04-19 20:37:58.036883.txt",
    
    "Llama-3-70b (original)": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-3/Meta-Llama-3-70B/kbase/entity_type/entity_type_result_2024-04-19 20:06:50.474479.txt",
    "Llama-3-70b (finetued)": "/home/ac.gpark/BioIE-LLM-WIP/result/LLaMA-3/Meta-Llama-3-70B/kbase/entity_type/entity_type_result_2024-04-20 11:51:44.111695.txt",
}

corrected_results = {}

labels = ["plasmid", "organism host", "promoter", "genome engineering", "cloning method", "reporter", "regulator", 
          "antibiotic marker", "genetic screen", "rbs", "counter selection", "terminator", "dna transfer", "operator", "none"]
# labels = np.array(labels)
    
for model_name, result_file in result_dict.items():
    pred_list, true_list = [], []
    
    with open(result_file) as fin:
        lines = fin.readlines()
        delimiter_idx = lines.index("********************************************************************\n")
        
        # stop at the original text if exists.
        if "####################################################################\n" in lines:
            end_idx = lines.index("####################################################################\n")
        else:
            end_idx = -1

        for idx, line in enumerate(lines[delimiter_idx+1:], delimiter_idx+1):
            
            if idx == end_idx:
                break

            num, pred, true = line.split(', ', 2)

            num = num.strip()
            pred = pred.strip()
            true = true.strip()
            
            pred = pred.lower()
            true = true.lower()

            pred_list.append(pred)
            true_list.append(true)

    scores = compute_metrics(pred_list, true_list)
    rounded_scores = {key: round(value, 4) for key, value in scores.items()}

    corrected_results[model_name] = rounded_scores

    cm = confusion_matrix(pred_list, true_list, labels=labels)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    if len(labels) > 3:
        dpi = 500
        xticks_rotation = 90
    else:
        dpi = 300
        xticks_rotation = "horizontal"
    
    cm_disp.plot(xticks_rotation=xticks_rotation)
    
    plt.tight_layout()
    cm_disp.figure_.savefig(model_name + "_confusion_matrix.png", dpi=dpi, pad_inches=5)
        
with open('kbase_correct_results.json', 'w') as fout:
    json.dump(corrected_results, fout)
    
