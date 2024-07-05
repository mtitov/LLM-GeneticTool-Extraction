import sys
import time
import string
import random
import re
import itertools

from sklearn.model_selection import train_test_split
from datasets import Dataset
from datetime import timedelta

# setting path
sys.path.append('../data_readers')
from data_readers import KBaseReader
from .base_processor import BaseProcessor

random.seed(42)


class KBaseProcessor(BaseProcessor):
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv)

        self.num_of_kbase_classes = kwargs['num_of_kbase_classes']
        self.data_reader = KBaseReader(self.data_repo_path, self.task, self.num_of_kbase_classes)
    

    def generate_datasets(
        self, 
        n_shots, 
        is_training
    ):
        task = self.task
        data = self.data_reader.data
        
        self.task_prompt[task] = ""
        
        train_dataset = None
        val_dataset = None
        test_dataset = None
        
        ent_types_included = {x: 0 for x in self.data_reader.ent_types}

        assert len(string.ascii_uppercase) >= len(self.data_reader.ent_types)
        
        ent_type_multiple_choices_dict = {x: y for x, y in zip(self.data_reader.ent_types, string.ascii_uppercase)}
        
        self.ent_type_multiple_choices_str = ", ".join(['"' + x + '"' for x in self.data_reader.ent_types])
        
        random.shuffle(data)
        
        ## TODO: get the size of training/val data as an argument.
        train_dataset, test_dataset = train_test_split(data, test_size=0.1, shuffle=False)
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=len(test_dataset), shuffle=False)
        
        if n_shots > 0:
            for item in train_dataset:
                ent_type = item['label']

                if all(value == n_shots for value in ent_types_included.values()):
                    break

                if ent_type in ent_types_included:
                    if ent_types_included[ent_type] < n_shots:
                        self.shot_samples.append(item)
                        ent_types_included[ent_type] += 1
            
            random.shuffle(self.shot_samples)
            
            for sample in self.shot_samples:
                text = sample['text']
                entity = sample['entity']
                label = sample['label']

                self.task_prompt[task] += f"{self.model_prompt['entity_type_q'](entity, text, self.ent_type_multiple_choices_str)}{self.model_prompt['entity_type_a'](label)}"

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        ## TODO: fix this later.
        if self.task == "entity_and_entity_type":
            self.test_dataset = data # testing all data


    def format_dataset(
        self, 
        dataset, 
        data_type
    ):
        task = self.task
        
        if task == "entity_type":
            if data_type == 'train':
                formatted_dataset = [
                                        {
                                            "text": f"{self.model_prompt['entity_type_q'](i['entity'], i['text'], self.ent_type_multiple_choices_str)}{self.model_prompt['entity_type_a'](i['label'])}",
                                            "answer": f"{self.model_prompt['entity_type_q'](i['entity'], i['text'], self.ent_type_multiple_choices_str)}{self.model_prompt['entity_type_a'](i['label'])}",
                                        }
                                        for i in dataset
                                    ]
            elif data_type in ['validation', 'test']:
                formatted_dataset = [
                                        {
                                            "entity": i['entity'],
                                            "text": f"{self.task_prompt[task]}{self.model_prompt['entity_type_q'](i['entity'], i['text'], self.ent_type_multiple_choices_str)}",
                                            "answer": i['label']
                                        }
                                        for i in dataset
                                    ]
        
        formatted_dataset = Dataset.from_list(formatted_dataset)

        return formatted_dataset
    
    
    def update_results(
        self,
        decoded_entity, 
        decoded_pred, 
        decoded_gold
    ):
        task = self.task

        for item, pred, true in zip(decoded_entity, decoded_pred, decoded_gold):			
            pred = pred.strip()
            true = true.strip()

            pred = self.clean_response(pred, true=true, entity=item)
            
            pred = pred.lower()
            true = true.lower()

            if len(self.results[task]['preprocessed']) != 0:
                self.results[task]['preprocessed'][0].append(pred)
                self.results[task]['preprocessed'][1].append(true)
            else:
                self.results[task]['preprocessed'] = [[pred], [true]]
            
        
    def infer(
        self,
        model, 
        generation_config,
        batch_size: int = 1,
    ):
        test_data = self.test_dataset
        task = self.task
        results = self.results

        shots_keys = [x['id'] for x in self.shot_samples]
        
        if task == "entity_type":
            test_sample_size = self.test_sample_size
        
        start = 0
        stop = start + batch_size

        while True:
            batch_data = itertools.islice(test_data, start, stop)
            
            ## TODO: make it cleaner later.
            if task == "entity_type":
                batch_items = [] # debug
                batch_input_texts = []
                true_list = []
                
                for item in batch_data:
                    batch_items.append(item)
                    text = item['text']
                    entity = item['entity']
                    ent_type = item['label']
                    
                    true_list.append(ent_type.lower())
                    
                    entity_type_prompt_with_test_sample = self.task_prompt[task]

                    entity_type_prompt_with_test_sample += self.model_prompt['entity_type_q'](entity, text, self.ent_type_multiple_choices_str)

                    batch_input_texts.append(entity_type_prompt_with_test_sample)

                pred_list = self.get_response(model, generation_config, batch_input_texts)

                for item, pred, true in zip(batch_items, pred_list, true_list):
                    orig_pred = pred # debug
                    
                    true = true.lower()
                    text = item['text']
                    entity = item['entity']
                    
                    pred = self.clean_response(pred, true=true, entity=entity)

                    if len(results[task]) != 0:
                        results[task][0].append(pred)
                        results[task][1].append(true)
                    else:
                        results[task] = [[pred], [true]]

            elif task == "entity_and_entity_type":
                batch_items = [] # debug
                batch_input_texts = []
                
                for item in batch_data:
                    batch_items.append(item)
                    text = item['text']
                    
                    entity_type_prompt_with_test_sample = self.task_prompt[task]
                    entity_type_prompt_with_test_sample += self.model_prompt['entyty_and_entity_type_q'](text, self.ent_type_multiple_choices_str)
                    
                    batch_input_texts.append(entity_type_prompt_with_test_sample)
                    
                pred_list = self.get_response(model, generation_config, batch_input_texts)
                
                for item, pred in zip(batch_items, pred_list):
                    item['pred'] = pred
                    results[task]['preprocessed'].append(item)

            print(f">> batch processed - len(test_data): {len(test_data)}, start: {start}, stop: {stop}")

            if stop >= len(test_data):
                break

            start = stop
            stop = start + batch_size

        return results
