import openpyxl
import json

from evaluator import *

file_path = "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mixtral-8x7B-Instruct-v0.1/kbase/entity_and_entity_type_annot/Mixtral prediction.xlsx"
sheet_name = "organism_hosts_genetic_tools"
output_dir = "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mixtral-8x7B-Instruct-v0.1/kbase"
task = "entity_and_entity_type_annot"
data_name = "kbase"

# Load the workbook
wb = openpyxl.load_workbook(file_path)

# Select the worksheet
sheet = wb[sheet_name]

pred = [] # Used for metrics.
true = [] # Used for metrics.

labels = ["plasmid", "organism host", "promoter", "genome engineering", "cloning method", "reporter", "regulator", 
          "antibiotic marker", "genetic screen", "rbs", "counter selection", "terminator", "dna transfer", "operator", "none"]

def parse_data(data):
    parsed_data = {}
    for entity_label, entity_list in data.items():
        
        # ignore non relevant labels in the annotation data.
        if entity_label == 'not relevant':
            continue
        
        parsed_data[entity_label] = []
        for entity_item in entity_list:
            entity_names = entity_item.split(',')
            entity_names = [x.strip() for x in entity_names]
            entity_names = [x for x in entity_names if len(x) != 0]

            for idx, x in enumerate(entity_names):
                # synonyms of E. coli
                if x.lower() in ['e. coli', 'e. coli.', 'e.coli', 'coli', 'escherichia coli', 'escherichia coli', 'escherichia coli (e. coli)'] or \
                   x in ["The paper uses Escherichia coli as the organism host for the optogenetic toolkit and antibiotic resistance genes.",
                         "The paper mentions the use of Escherichia coli as the host organism for cloning and transformation.", 
                         "The paper uses Escherichia coli as the organism host for expressing the genetic circuits.",
                         "The paper uses E. coli as the host organism for the phage Mu transposition experiments.",
                         "The paper mentions the use of E. coli as an organism host for the genetic tools and techniques described. E. coli is a common bacterium that is often used as a host for genetic engineering due to its ability to easily take up and replicate plasmids.",
                         "The paper mentions the use of E. coli as a host organism for genetic engineering and plasmid cloning.",
                         "Cre recombinase is expressed in E. coli cells."
                         ]: 
                    entity_names[idx] = 'Escherichia coli'
                
                # synonyms of E. coli BL21
                if x.lower() in ['e. coli bl21', 'escherichia coli bl21'] or \
                   x in ["The organism host used in this study is Escherichia coli (E. coli) strain BL21.",
                         ]: 
                    entity_names[idx] = 'Escherichia coli BL21'				
                
                # synonyms of A. ferrooxidans
                if x.lower() in ["acidithiobacillus ferrooxidans", "a. ferrooxidans"]: 
                    entity_names[idx] = 'Acidithiobacillus ferrooxidans'
                
                # synonyms of B. bacillus ("acillus subtilis" appears to be a typographical error.)		
                if x.lower() in ["bacillus subtilis", "b. subtilis", "bacillus", "acillus subtilis"] or \
                    x in ["The paper uses B. subtilis as an organism host for the riboswitch reporter system and for constructing strains heterologously expressing P. megaterium metE and metH."]: 
                    entity_names[idx] = 'Bacillus subtilis'
                
                # synonyms of Saccharomyces cerevisiae				
                if x.lower() in ["saccharomyces cerevisiae", "s. cerevisiae"] or \
                    x in ["The authors used the yeast Saccharomyces cerevisiae as the organism host to express the protein components."]: 
                    entity_names[idx] = 'Saccharomyces cerevisiae'

                # synonyms of Kluyveromyces lactis	
                if x.lower() in ["kluyveromyces lactis"] or \
                    x in ["The paper mentions that the yeast Kluyveromyces lactis contains killer DNA plasmids."]: 
                    entity_names[idx] = 'Kluyveromyces lactis'
    
                
            entity_names = list(set(entity_names)) # remove duplicates
            # entity_names = [x.lower() for x in entity_names] # lowercase names. Don't lowercase to keep the original words for JSON file to be used in the web app.
            # entity_label = entity_label.lower()
            
            if entity_label in parsed_data:
                parsed_data[entity_label] += entity_names
            else:
                parsed_data[entity_label] = entity_names
            
            # print(title)
            # print(parsed_data[entity_label])
            # input('enterr.')
            # for name in entity_names: 
                # print(name)
                
    return parsed_data


# the data format to send to Chris.
correct_info = []
def update_correct_info(organism_host, label, entity, title, uid):
    
    for item in correct_info:
        if organism_host == item["species"]:
            item["tools"].append({
                "type": label, 
                "name": entity, 
                "references": [title, uid]
            })
            return
        
    correct_info.append({
        "species": organism_host,
        "tools": [{
            "type": label, 
            "name": entity, 
            "references": [title, uid]
        }]
    })
    return


# debug
paper_with_single_organism = []


# Iterate through all rows (starting from row 2, skip header)
for row in sheet.iter_rows(min_row=2):
    title_cell = row[0]
    uid_cell = row[1]
    anno_cell = row[2]
    pred_cell = row[3]
    true_cell = row[4]
    wrong_cell = row[5]
    
    paper_with_single_organism_anno_flag = False
    paper_with_single_organism_pred_flag = False
    is_complete_wrong = False
    
    if title_cell.fill.start_color.index != '00000000':
        is_complete_wrong = True

    title = title_cell.value
    uid = uid_cell.value
    anno_data = json.loads(anno_cell.value)
    pred_data = json.loads(pred_cell.value)
    true_data = json.loads(true_cell.value) # Not used.
    
    anno_data = parse_data(anno_data)
    pred_data = parse_data(pred_data)
    
    organism_host_anno = None
    if "organism host" in anno_data:
        if len(anno_data["organism host"]) == 1:
            paper_with_single_organism.append(title)
            paper_with_single_organism_anno_flag = True
            organism_host_anno = anno_data["organism host"][0]
    
    organism_host_pred = None
    if "organism host" in pred_data:
        if len(pred_data["organism host"]) == 1:
            paper_with_single_organism.append(title)
            paper_with_single_organism_pred_flag = True
            organism_host_pred = pred_data["organism host"][0]

    for label, entity_list in anno_data.items():
        for entity in entity_list:
            pred.append(label)
        
            if organism_host_anno is not None:
                # the data format to send to Chris's web application.
                update_correct_info(organism_host_anno, label, entity, title, uid)

            # if label in pred_data and entity in pred_data[label]:
            if label in pred_data and entity.lower() in [x.lower() for x in pred_data[label]]:

                if is_complete_wrong:
                    true.append("none")
                else:
                    true.append(label)
                
                # print(title)
                # print(entity)
                # print(pred_data[label])
                
                # pred_data[label].remove(entity) # remove the entity to avoid duplicate check below.
                pred_data[label] = [x for x in pred_data[label] if x.lower() != entity] # remove the entity to avoid duplicate check below. use this code to keep the original words for JSON file to be used in the web app.
            else:
                true.append("none")

    if is_complete_wrong is False:
        if wrong_cell.value is not None:
            wrong_data = wrong_cell.value
            wrong_data = wrong_data.strip()
            wrong_data = wrong_data.rstrip(',')
            wrong_data = '{' + wrong_data + '}'
            wrong_data = json.loads(wrong_data)
            wrong_data = parse_data(wrong_data)
            
    for label, entity_list in pred_data.items():
        for entity in entity_list:
            pred.append(label)
            # if label in wrong_data and entity in wrong_data[label]:
            if label in wrong_data and entity.lower() in [x.lower() for x in wrong_data[label]]:
                # print(title)
                # print(entity)
                # print(pred_data[label])
                # input('enter..')
                true.append("none")
            else:
                if is_complete_wrong:
                    true.append("none")
                else:
                    
                    if organism_host_pred is not None:
                        # the data format to send to Chris's web application.
                        update_correct_info(organism_host_pred, label, entity, title, uid)
                
                    true.append(label)

# debug
paper_with_single_organism = set(paper_with_single_organism)
for num, x in enumerate(paper_with_single_organism, 1):
    print(num, x)


pred = [x.lower() for x in pred]
true = [x.lower() for x in true]

# debug
'''
def count_items_in_list(lst):
    # Create an empty dictionary to store counts
    item_counts = {}

    # Iterate through the list and count occurrences of each item
    for item in lst:
        if item in item_counts:
            item_counts[item] += 1
        else:
            item_counts[item] = 1

    return item_counts
    
def count_matches(list1, list2):
    # Ensure both lists have the same length
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")

    # Create a dictionary to store counts of matches at each index
    match_counts = {}

    # Iterate through the lists and count matches at each index
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            if list1[i] in match_counts:
                match_counts[list1[i]] += 1
            else:
                match_counts[list1[i]] = 1
        # else:
            # print('mismatch!!')
            # input('enter..')

    # Sort the match counts dictionary by value in descending order
    sorted_matches = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)
    
    item_counts = count_items_in_list(list1)
    
    # Print the sorted match counts
    for item in sorted_matches:
        print(f"Match '{item[0]}' count: {item[1]} total: {item_counts[item[0]]} accuracy: {round(item[1]/item_counts[item[0]], 4)}")

count_matches(pred, true)
input('enter..')
'''

scores = compute_metrics(pred, true)

save_results(
    scores=scores, 
    pred=pred, 
    true=true,
    task=task, 
    labels=labels, 
    output_dir=output_dir,
    task_prompt="",
    data_name=data_name,
)

with open(os.path.join(output_dir, "entity_and_entity_type_annot/hosts_tools.json"), "w") as file:
    json.dump(correct_info, file)