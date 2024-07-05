import os
import json
import re
import pickle


class KBaseReader:
    
    def __init__(
        self, 
        path, 
        task,
        num_of_kbase_classes
    ):
        path = os.path.expanduser(path)
        
        if task == 'entity_type': # annotated data file.
            data_file = os.path.join(path, "KBase/prompt_records.json")
            self.data = json.load(open(data_file))
            self.data = [x for x in self.data if x["label"] != "not relevant"]

        elif task == 'entity_and_entity_type':
            # articles used for the annotation task.
            file = open(os.path.join(path, "KBase/all_annot_data.pkl"), 'rb')
            pkl_data = pickle.load(file)
            file_data = []
            for d in pkl_data:
                title = d['title'].strip()
                full_text = d['full_text'].strip()

                if len(title) > 0:
                    input_txt = 'Title: ' + title + '. Full-Text: ' + full_text
                else:
                    input_txt = 'Full-Text: ' + full_text
                    
                file_data.append({'title': d['title'], 'doi': d['uid'], 'annotations': d['annotations'], 'text': input_txt})
            
            self.data = file_data
            
        
        # get entity types
        ent_type_file = os.path.join(path, "KBase/entity_types.json")
        ent_type_data = json.load(open(ent_type_file))
        
        self.ent_types = [x for x in ent_type_data]
        self.ent_types = self.ent_types[:num_of_kbase_classes]
