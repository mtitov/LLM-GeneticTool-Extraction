import os
import json
import re
import csv


# from 1 to 200 papers
result_file = "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mixtral-8x7B-Instruct-v0.1/kbase/entity_and_entity_type/entity_and_entity_type_result_2024-03-14 06:39:22.587743.txt"

with open(result_file) as fin:
    result_txt = fin.read()

json_data1 = json.loads(result_txt)

# from 201 to 389 papers
result_file = "/home/ac.gpark/BioIE-LLM-WIP/result/Mistral/Mixtral-8x7B-Instruct-v0.1/kbase/entity_and_entity_type/entity_and_entity_type_result_2024-03-14 19:42:54.136581.txt"

with open(result_file) as fin:
    result_txt = fin.read()

json_data2 = json.loads(result_txt)

json_data = json_data1 + json_data2

print(len(json_data))

labels = ["plasmid", "organism host", "promoter", "genome engineering", "cloning method", "reporter", "regulator", 
          "antibiotic marker", "genetic screen", "rbs", "counter selection", "terminator", "dna transfer", "operator"]

synonym_list = [
    ['E. coli', 'Escherichia coli', 'coli'],
]

filtered_num_of_papers_cnt = 0 # number of papers having labels besides not_relevant labels.


output_list = []

'''
"title": "Genome-wide phenotypic analysis of growth, cell morphogenesis and cell cycle events in Escherichia coli\n",
"doi": "https://doi.org/10.1101/101832",
"annotations": [{
        "id": 2064,
        "label": "not relevant",
        "start_offset": 1004,
        "end_offset": 1012,
        "entity": "nucleoid",
        "which_text": "abstract",
        "text": "Cell size, cell growth and the cell cycle are necessarily intertwined to achieve robust bacterial replication. Yet, a comprehensive and integrated view of these fundamental processes is lacking. Here, we describe an image-based quantitative screen of the single-gene knockout collection of Escherichia coli, and identify many new genes involved in cell morphogenesis, population growth, nucleoid (bulk chromosome) dynamics and cell division. Functional analyses, together with high-dimensional classification, unveil new associations of morphological and cell cycle phenotypes with specific functions and pathways. Additionally, correlation analysis across ~4,000 genetic perturbations shows that growth rate is surprisingly not predictive of cell size. Growth rate was also uncorrelated with the relative timings of nucleoid separation and cell constriction. Rather, our analysis identifies scaling relationships between cell size and nucleoid size and between nucleoid size and the relative timings of nucleoid separation and cell division. These connections suggest that the nucleoid links cell morphogenesis to the cell cycle."
    }
],
"text": "Title: Genome-wide phenotypic analysis of growth, cell morphogenesis and cell cycle events in Escherichia coli. Full-Text: Cell size, 
'''
for item in json_data:
    # print(item['annotations'])
    # print(item['pred'])
    # input('enter..')
    
    
    '''
    if item['title'] == '':
        print(item)
        input('enter..')
    '''
    
    
    anno = {}
    
    are_all_labels_not_relevant = True
    
    '''
    Some entities belong to more than one label.
    E.g.,
        >> entity with multiple labels: CRISPRi ['genome engineering', 'genetic screen']
        >> entity with multiple labels: pCola ['reporter', 'plasmid']
        >> entity with multiple labels: BEVA ['genome engineering', 'plasmid']
        >> entity with multiple labels: pyrF ['genome engineering', 'genetic screen']
    '''
    for i in item['annotations']:
        label = i['label']
        entity = i['entity'].strip()
        
        ## TODO: check if this is an annotation error. 
        if len(entity) == 0:
            continue
        
        # code to find entities that belong to more than one label.
        '''
        if entity in anno:
            if label not in anno[entity]:
                anno[entity].append(label)
                # debug
                print('entity with multiple labels:', entity, anno[entity])
        else:
            anno[entity] = [label]
        '''
        if label in anno:
            anno[label].append(entity)
        else:
            anno[label] = [entity]
            
        if label != 'not relevant':
            are_all_labels_not_relevant = False
            
    anno = {k: list(set(v)) for k, v in anno.items()}
    
    # currently, 'not relevant' labels are ignored.
    if are_all_labels_not_relevant == True:
        continue
    
    filtered_num_of_papers_cnt += 1
    
    model_pred = item['pred']

    # clean the generated text.
    '''
    (annotation <-> model prediction)
    e.g., 
        - 'label': 'organism host', 'entity': 'E. coli ' <-> Organism host: Escherichia coli K12 W3110 wildtype
        - 'label': 'promoter', 'entity': 'ParaBAD' <-> Promoter: ParaBAD promoter (cf. Plasmid: pBbB8k (4480 bp) plasmid with the ParaBAD::GFP)
        - 'label': 'reporter', 'entity': 'GFP' <->  Reporter: GFP (cf. Plasmid: pBbB8k (4480 bp) plasmid with the ParaBAD::GFP)
        - 'label': 'not relevant', 'entity': 'Assembly' <-> Promoter: The paper mentions the use of promoters in the construction of synthetic promoters using Loop assembly.
        - 'Synechocystis': ['organism host'] <->  Organism host: Synechocystis sp. PCC 6803, 
        - 'BBBa_J23106': ['promoter'] <-> Promoter: BBBa\_J23106 (Pj106)
        - 'ParaBAD': ['promoter'] <-> Promoter: ParaBAD promoter
        - 'Golden Gate': ['cloning method’] <->  Cloning method: Golden Gate assembly 
    '''
    
    # print(anno)
    # print('==========================')
    # print(model_pred)
    # input('enter..')
    
    pred = {}
    
    for line in model_pred.splitlines():
        line = line.strip()
        
        if len(line) == 0:
            continue
        
        # Remove any numerical value or asterisks at the beginning of the string
        cleaned_line = re.sub(r"^(?:[\d*]+)*", "", line)
        
        # Remove any dots and any spaces if they exist at the beginning of the string
        cleaned_line = re.sub(r"^[\s.-]*", "", cleaned_line)
        
        # Remove any double quotation marks in the string
        # E.g., 
        #	* "plasmid": pTpsMAG1,
        #   * "plasmid" (used for library construction)
        cleaned_line = cleaned_line.replace('"', '')
        
        # debug
        # print('>> line:', line)
        # print('>> cleaned_line:', cleaned_line)
        # continue
        
        label_entity_line_flag = False
        for label in labels:
            lowercased_cleaned_line = cleaned_line.lower()
            if lowercased_cleaned_line.startswith(label):
            # if lowercased_cleaned_line.startswith(label + ': '): # E.g., RBS (ribosome binding site):, counter selection marker:
                label_entity_line_flag = True
                break
        
        if label_entity_line_flag == True:
            if ': ' in cleaned_line:
                pred_label, pred_entity_txt = cleaned_line.split(':', 1)
                pred_label = pred_label.strip()
                pred_entity_txt = pred_entity_txt.strip()
                
                pred_label_lowercased = pred_label.lower()
                pred_entity_txt_lowercased = pred_entity_txt.lower()

                # convert plural to singular. e.g., plasmids, organism hosts, promoters, antibiotic markers
                if pred_label_lowercased.endswith('s') and pred_label_lowercased != 'rbs':
                    pred_label_lowercased = pred_label_lowercased[:-1]
                
                if pred_label_lowercased == "rbs (ribosome binding site)":
                    pred_label_lowercased = 'rbs'
                
                ## TODO: double check if "counter selection", "counter selection marker" are the same thing.
                if pred_label_lowercased == "counter selection marker":
                    pred_label_lowercased = 'counter selection'
                
                # print(pred_label)
                # input('enter..')
                
                if pred_label_lowercased not in labels:
                    # debug
                    # print('>> Unknown label:', pred_label)
                    # print('>> cleaned_line:', cleaned_line)
                    # input('enter..')
                    continue
                    
                if pred_entity_txt_lowercased.startswith('not specified') or \
                   pred_entity_txt_lowercased.startswith('not mentioned') or \
                   pred_entity_txt_lowercased.startswith('none mentioned') or \
                   pred_entity_txt_lowercased.startswith('not explicitly mentioned') or \
                   pred_entity_txt_lowercased.startswith('the paper does not mention') or \
                   pred_entity_txt_lowercased.startswith('the paper does not explicitly mention') or \
                   pred_entity_txt_lowercased.startswith('the paper does not specify') or \
                   pred_entity_txt_lowercased.startswith('the authors did not mention'):
                    continue

                # print(pred_entity_txt)
                # input('enter..')
                
                if pred_label_lowercased in pred:
                    if pred_entity_txt not in pred[pred_label_lowercased]: # ignore duplicates.
                        pred[pred_label_lowercased].append(pred_entity_txt)
                else:
                    pred[pred_label_lowercased] = [pred_entity_txt]
                
            # else:
                # debug
                # print(cleaned_line)
                # print(item['title'])
                # print(item['doi'])
                
                # the following two cases are manually added to the csv file.
                '''
                plasmid (used for library construction)
                organism host (Saccharomyces cerevisiae)
                promoter (synthetic promoter taken from a published promoter library)
                genome engineering (Cre-Lox based method for library integration into the yeast genome)
                cloning method (NEBuilder HiFi DNA Assembly)
                reporter (not explicitly mentioned, but the library is designed to measure splicing efficiency)
                regulator (not explicitly mentioned)
                antibiotic marker (KanMX-1, NAT)
                genetic screen (not explicitly mentioned)
                RBS (not explicitly mentioned)
                counter selection (not explicitly mentioned)
                terminator (ADH1 terminator sequence)
                DNA transfer (electroporation)
                operator (Lox71 site)
                
                plasmid is mentioned as a genetic tool.
                organism host is mentioned as E. coli.
                promoter is mentioned as a type of genetic entity.
                genome engineering is mentioned as a type of genetic tool.
                cloning method is mentioned as Golden Gate assembly.
                reporter is not mentioned.
                regulator is not mentioned.
                antibiotic marker is mentioned as kanamycin and ampicillin.
                genetic screen is mentioned as a method.
                RBS is mentioned as a type of genetic entity.
                counter selection is not mentioned.
                terminator is not mentioned.
                DNA transfer is mentioned as electroporation.
                operator is not mentioned.
                '''
    
        # else:
            # debug
            # print(line)
            '''
            The biological entities and genetic tools mentioned in this paper are: "plasmid", "organism host", "promoter", "genome engineering", "cloning method", "reporter", "regulator", "antibiotic marker", "genetic screen", "RBS", "counter selection", "terminator", "DNA transfer", and "operator". The organism host is Pseudomonas species, and the plasmids used are pAblo·pCasso. The promoter used is Pm, and the regulator is XylS. The antibiotic marker is Str, and the reporter is msfGFP. The cloning method used is USER cloning. The RBS used is the native 5'-UTR region preceding repA and the Pm promoter. The terminator used is the native terminator of the gene of interest. The DNA transfer method used is electroporation. The operator used is the cognate guide, formed as a complex of a transactivating crRNA (tracrRNA) and a small RNA molecule (crRNA), transcribed from the CRISPR array.
        
            The paper describes the development of arabinose-inducible artificial transcription factors (ATFs) using CRISPR/dCas9 and Arabidopsis-derived DNA-binding proteins to control gene expression in E. coli and Salmonella. The authors demonstrate the utility of these ATFs by engineering a Salmonella biosensor strain that responds to the presence of alkaloid drugs with quantifiable fluorescent output and by controlling β-carotene biosynthesis in E. coli. The ATFs are derived from widely different DNA-binding domains (DBDs) and are expressed using the arabinose-inducible araBAD promoter (PBAD). The ATFs target their binding sites (BSs) located upstream of a weak synthetic promoter, which results in the expression of the target gene. The authors also optimize the arabinose induction system in Salmonella by genetically modifying the L-arabinose (arabinose) catabolic pathway and defining an optimal 'arabinose induction window' that results in high heterologous gene expression with minimal effects on bacterial growth. The authors further establish an arabinose-inducible CRISPR/dCas9-derived ATF library in Salmonella and evaluate a novel class of ATFs with heterologous DBDs based on plant-specific transcription factors (TFs) from Arabidopsis thaliana combined with the SoxS AD. The authors also characterize the dose-dependency of arabinose-based gene expression from plasmid and chromosome in S. Typhimurium LT2 and perform growth curve analyses to determine the optimal conditions for arabinose-inducible gene expression. The authors also demonstrate the utility of the ATFs by engineering a Salmonella strain to function as a sensitive biosensor for alkaloid drugs and an E. coli strain as a microbial cell factory to produce β-carotene.
            
            The biological entities and genetic tools mentioned in this paper are: "plasmid" (pCA24, pET28, pBAD33 topA_strepII G116S M320V, pCA24 14kDa CTD, pCA24 topA, pCA24 GFP), "organism host" (E. coli DY330, E. coli DY330 topA-SPA, E. coli DY330 rpoC-TAP, E. coli DH5α, E. coli BW25113, E. coli W3110, E. coli W3110 MuSGS), "promoter" (T5-lac promoter), "genome engineering" (recombineering approach using Lambda Red system on pKD46 plasmid), "cloning method" (PCR amplification, overlap extension PCR, restriction enzymes digestion, ligation), "reporter" (Cy5-labeled oligonucleotides), "regulator" (RNAP, EcTopoI), "antibiotic marker" (chloramphenicol, kanamycin, ampicillin), "genetic screen" (SOS-response), "RBS" (ribosome binding site), "counter selection" (proteolysis and de-crosslinking after washing procedure), "terminator" (T7 terminator), "DNA transfer" (transformation, transduction, conjugation), "operator" (lac operator).
            
            The biological entities and genetic tools mentioned in this paper are: "plasmid", "organism host", "promoter", "genome engineering", "cloning method", "reporter", "regulator", "antibiotic marker", "genetic screen", "RBS", "counter selection", "terminator", "DNA transfer", and "operator".
            '''
    
    # debug
    '''
    if all(not value for value in pred.values()):
        print(item['title'], item['doi'])
        for k, v in pred.items():
            print(k, v)
        print(model_pred)
        print('-------------------------------------------------------------\n')
    '''	
    # the following three cases are manually added to the csv file.
    '''
    Unraveling the functional role of DNA methylation using targeted DNA demethylation by steric blockage of DNA methyltransferase with CRISPR/dCas9 https://doi.org/10.1101/2020.03.28.012518
    The biological entities and genetic tools mentioned in this paper are: "plasmid" (Il33-pCpGl, SV40-pCpGl), "organism host" (NIH-3T3 cells, HEK293 cells, MDA-MB-231 cells, Fragile X syndrome patient primary fibroblasts), "promoter" (Il33-002 promoter, SERPINB5 promoter, Tnf promoter), "genome engineering" (dCas9, Cre recombinase), "cloning method" (lentiviral transfer plasmids, TOPO-TA cloning), "reporter" (luciferase), "regulator" (DNMT1, DNMT3A, TET1, TET2, OGT), "antibiotic marker" (blasticidin, puromycin), "genetic screen" (Cas9-mediated gene depletion studies), "RBS" (ribosome binding site), "counter selection" (KRuO4 oxidation of DNA followed by bisulfite-pyrosequencing), "terminator" (transcription terminator), and "DNA transfer" (lentiviral transduction).
    -------------------------------------------------------------

    The pAblo·pCasso self-curing vector toolset for unconstrained cytidine and adenine base-editing in Pseudomonas species https://doi.org/10.1101/2023.04.16.537106
    The biological entities and genetic tools mentioned in this paper are: "plasmid", "organism host", "promoter", "genome engineering", "cloning method", "reporter", "regulator", "antibiotic marker", "genetic screen", "RBS", "counter selection", "terminator", "DNA transfer", and "operator". The organism host is Pseudomonas species, and the plasmids used are pAblo·pCasso. The promoter used is Pm, and the regulator is XylS. The antibiotic marker is Str, and the reporter is msfGFP. The cloning method used is USER cloning. The RBS used is the native 5'-UTR region preceding repA and the Pm promoter. The terminator used is the native terminator of the gene of interest. The DNA transfer method used is electroporation. The operator used is the cognate guide, formed as a complex of a transactivating crRNA (tracrRNA) and a small RNA molecule (crRNA), transcribed from the CRISPR array.

    -------------------------------------------------------------
    Interaction Between Transcribing RNA Polymerase and Topoisomerase I Prevents R-loop Formation in E. coli https://doi.org/10.1101/2021.10.26.465782
    The biological entities and genetic tools mentioned in this paper are: "plasmid" (pCA24, pET28, pBAD33 topA_strepII G116S M320V, pCA24 14kDa CTD, pCA24 topA, pCA24 GFP), "organism host" (E. coli DY330, E. coli DY330 topA-SPA, E. coli DY330 rpoC-TAP, E. coli DH5α, E. coli BW25113, E. coli W3110, E. coli W3110 MuSGS), "promoter" (T5-lac promoter), "genome engineering" (recombineering approach using Lambda Red system on pKD46 plasmid), "cloning method" (PCR amplification, overlap extension PCR, restriction enzymes digestion, ligation), "reporter" (Cy5-labeled oligonucleotides), "regulator" (RNAP, EcTopoI), "antibiotic marker" (chloramphenicol, kanamycin, ampicillin), "genetic screen" (SOS-response), "RBS" (ribosome binding site), "counter selection" (proteolysis and de-crosslinking after washing procedure), "terminator" (T7 terminator), "DNA transfer" (transformation, transduction, conjugation), "operator" (lac operator).
    '''


    

    true = {k: [] for k in pred}

    for label, true_entity_list in anno.items():
        if label in pred:
            pred_entity_txt_list = pred[label]
            
            for true_ent in true_entity_list:
                for pred_ent_txt in pred_entity_txt_list:
                    if true_ent in pred_ent_txt:
                        # print('>> label:', label)
                        # print('>> true_ent:', true_ent)
                        # print('>> pred_ent_txt:', pred_ent_txt)
                        # input('enter..')
                        
                        true[label].append(true_ent)
    
    '''
    "title": "Genome-wide phenotypic analysis of growth, cell morphogenesis and cell cycle events in Escherichia coli\n",
    "doi": "https://doi.org/10.1101/101832",
    "annotations": [{
            "id": 2064,
            "label": "not relevant",
            "start_offset": 1004,
            "end_offset": 1012,
            "entity": "nucleoid",
            "which_text": "abstract",
            "text": "Cell size, cell growth and the cell cycle are necessarily intertwined to achieve robust bacterial replication. Yet, a comprehensive and integrated view of these fundamental processes is lacking. Here, we describe an image-based quantitative screen of the single-gene knockout collection of Escherichia coli, and identify many new genes involved in cell morphogenesis, population growth, nucleoid (bulk chromosome) dynamics and cell division. Functional analyses, together with high-dimensional classification, unveil new associations of morphological and cell cycle phenotypes with specific functions and pathways. Additionally, correlation analysis across ~4,000 genetic perturbations shows that growth rate is surprisingly not predictive of cell size. Growth rate was also uncorrelated with the relative timings of nucleoid separation and cell constriction. Rather, our analysis identifies scaling relationships between cell size and nucleoid size and between nucleoid size and the relative timings of nucleoid separation and cell division. These connections suggest that the nucleoid links cell morphogenesis to the cell cycle."
        }
    ],
    "text": "Title: Genome-wide phenotypic analysis of growth, cell morphogenesis and cell cycle events in Escherichia coli. Full-Text: Cell size, 
    '''
    
    
    # Sort the dictionary by keys
    anno = dict(sorted(anno.items()))
    pred = dict(sorted(pred.items()))
    true = dict(sorted(true.items()))

    output_list.append({'Title': item['title'], 
                        'UID': item['doi'], 
                        'Annotation': json.dumps(anno, indent=4),
                        'Model Prediction': json.dumps(pred, indent=4),
                        'True': json.dumps(true, indent=4)})

    
    
print(filtered_num_of_papers_cnt)


# CSV file name
csv_file = "organism_hosts_genetic_tools.csv"

# Writing the list of dictionaries to a CSV file
with open(csv_file, 'w', newline='') as csvfile:
    fieldnames = output_list[0].keys()  # Get the field names from the first dictionary
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write header
    writer.writeheader()
    
    # Write rows
    for row in output_list:
        writer.writerow(row)
        
        