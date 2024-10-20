
import os
import sys
import time

from datetime import datetime, timedelta

import radical.utils as ru

from ..evaluators import compute_metrics, save_results

from ..run_model import load_model, get_data_processor, get_args, get_rank
from ..run_model import torch, dist
from ..run_model import PeftModel, BitsAndBytesConfig
from ..run_model import Accelerator, DistributedType

# environment
os.environ['HF_HOME'] = '/eagle/RECUP/matitov/.cache/huggingface'

DEFAULT_CONFIG_FILE = 'service.json'


class ModelService(ru.zmq.Server):

    def __init__(self, path=None, args=None):
        ru.zmq.Server.__init__(self, path=path)

        cfg_file = os.path.join(path or '', DEFAULT_CONFIG_FILE)
        if not os.path.exists(cfg_file):
            raise FileNotFoundError('Model configuration file not found')

        self._args = ru.TypedDict(from_dict=ru.read_json(cfg_file))
        if args:
            self._args.update(vars(args))
        self.configure()

        self._model = None
        self.load_model()

        self._data_processor = None
        self.get_data_processor()

        self._generate_time  = 0
        self._tune_time      = 0
        self._inference_time = 0

        self.register_request('processor', self._processor)

    def configure(self):

        dirs_list = ['data_repo_path', 'output_dir']
        if self._args.lora_finetune:
            dirs_list.append('lora_output_dir')
        for k in dirs_list:
            self._args[k] = os.path.expanduser(self._args[k])

        mtype = self._args.model_type or ''
        mtype = mtype.rsplit('/', 1)[1] if '/' in mtype else mtype
        self._args.output_dir = os.path.join(self._args.output_dir,
                                             self._args.model_name,
                                             mtype,
                                             self._args.data_name)
        if get_rank() == 0:
            if not os.path.exists(self._args.output_dir):
                os.makedirs(self._args.output_dir)

        if self._args.lora_finetune:
            self._args.lora_output_dir = os.path.join(self._args.lora_output_dir,
                                                      self._args.model_type,
                                                      self._args.data_name,
                                                      self._args.task)
            if get_rank() == 0:
                if not os.path.exists(self._args.lora_output_dir):
                    os.makedirs(self._args.lora_output_dir)

        # Sanity check: ensure that a task for a dataset is correct
        if self._args.data_name == 'kbase':
            assert self._args.task in ['entity_type', 'entity_and_entity_type']

        if self._args.max_new_tokens == 0:
            if self._args.task == 'entity_type':  # data: kbase
                self._args.max_new_tokens = 40
            elif self._args.task == 'entity_and_entity_type':  # data: kbase
                self._args.max_new_tokens = 2000  # 500, 2000
            else:
                raise ValueError('Invalid task: ' + self._args.task)

        if (self._args.model_name in ['Falcon'] or
                self._args.model_type == 'mosaicml/mpt-7b-chat'):
            self._args.model_max_len = 2048
        elif self._args.model_name in ['LLaMA-2', 'Solar']:
            self._args.model_max_len = 4096
        elif (self._args.model_name in ['LLaMA-3'] or
              self._args.model_type in ['mosaicml/mpt-30b-chat',
                                        'mistralai/Mistral-7B-Instruct-v0.2']):
            self._args.model_max_len = 8192
        elif self._args.model_type == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
            self._args.model_max_len = 32768  # 8192 * 4

        # ------------------------------
        accelerator = Accelerator()
        self._args.state = accelerator.state
        if self._args.state.distributed_type == DistributedType.NO:
            self._args.device_map = 'auto'  # otherwise it will be None
        # ------------------------------

        if self._args.use_quantization:
            self._args.data_type = torch.bfloat16
            self._args.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                # load_in_8bit=True,
                bnb_4bit_compute_dtype=self._args.data_type,
                # bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"  # defaults to "fp4"
            )
            self._args.load_in_4bit_flag = True
        else:
            self._args.data_type = 'auto'
            self._args.load_in_4bit_flag = False

        self._args.max_memory = {i: '80GiB'
                                 for i in range(torch.cuda.device_count())}

        self._args.load_in_8bit_flag = False
        self._args.low_cpu_mem_usage = True

    def load_model(self):
        model_info = load_model(self, **{k: self._args[k]
                                         for k in ['model_name',
                                                   'model_type',
                                                   'max_new_tokens',
                                                   'data_type',
                                                   'load_in_4bit_flag',
                                                   'load_in_8bit_flag',
                                                   'max_memory',
                                                   'low_cpu_mem_usage',
                                                   'quantization_config',
                                                   'device_map',
                                                   'lora_finetune']})
        self._model                  = model_info[0]
        self._args.tokenizer         = model_info[1]
        self._args.generation_config = model_info[2]

    def get_data_processor(self):
        self._data_processor = get_data_processor(
            *[self._args[k] for k in ['data_name',
                                      'data_repo_path',
                                      'task',
                                      'test_sample_size',
                                      'model_name',
                                      'tokenizer',
                                      'model_max_len',
                                      'generation_config']],
            **{k: self._args[k] for k in ['num_of_kbase_classes']})

    def _processor(self, arg):

        if arg['action'] == 'generate':

            # measure the processing time
            if get_rank() == 0:
                start_time = time.time()

            self._data_processor.generate_datasets(self._args.n_shots,
                                                   self._args.lora_finetune)

            if get_rank() == 0:
                self._generate_time = time.time() - start_time

        elif arg['action'] == 'tune':

            # measure the processing time
            if get_rank() == 0:
                start_time = time.time()

            if self._args.lora_finetune:
                if (self._args.model_name in ['Falcon', 'Solar'] or
                        self._args.model_type in [
                            'meta-llama/Llama-2-70b-chat-hf',
                            'meta-llama/Meta-Llama-3-70B',
                            'meta-llama/Meta-Llama-3-70B-Instruct']):
                    self._data_processor.finetune(
                        self._model,
                        *[self._args[k] for k in ['model_type',
                                                  'train_batch_size',
                                                  'validation_batch_size',
                                                  'lora_output_dir']])
                    self._args.lora_output_dir = os.path.join(
                        self._args.lora_output_dir, 'final_checkpoint')

                else:
                    self._data_processor.finetune_by_accelerator(
                        self._model,
                        *[self._args[k] for k in ['model_type',
                                                  'train_batch_size',
                                                  'validation_batch_size',
                                                  'lora_output_dir']])

                _model = load_model(self, **{k: self._args[k]
                                             for k in ['model_name',
                                                       'model_type',
                                                       'max_new_tokens',
                                                       'data_type',
                                                       'load_in_4bit_flag',
                                                       'load_in_8bit_flag',
                                                       'max_memory',
                                                       'low_cpu_mem_usage',
                                                       'quantization_config',
                                                       'device_map',
                                                       'lora_finetune']})[0]

                if self._args.model_name == 'MPT':
                    # MPT is trained with right padding, so change it to
                    # left padding for inference.
                    self._args.tokenizer.padding_side = 'left'

                # Merge Model with Adapter
                self._model = PeftModel.from_pretrained(
                    _model,  # the base model with full precision
                    self._args.lora_output_dir,  # path to the finetuned adapter
                )

            elif self._args.lora_weights:
                self._model = PeftModel.from_pretrained(
                    self._model,
                    self._args.lora_weights,
                )

            self._model.config.use_cache = True  # enable for inference!
            self._model.eval()

            if torch.__version__ >= '2' and sys.platform != 'win32':
                self._model = torch.compile(self._model)

            if get_rank() == 0:
                self._tune_time = time.time() - start_time

        elif arg['action'] == 'inference':

            # measure the processing time
            if get_rank() == 0:
                start_time = time.time()

            if self._args.task == 'entity_and_entity_type':
                self._data_processor.infer(
                    self._model,
                    self._args.generation_config,
                    self._args.test_batch_size)
            else:
                self._data_processor.infer_by_accelerator(
                    self._model,
                    'test',
                    self._args.test_batch_size)

            if get_rank() == 0:
                self._inference_time = time.time() - start_time

        return f'{arg["action"].title()} completed'

    def save_results(self):
        results = self._data_processor.results
        preprocessed = results[self._args.task]['preprocessed']

        if get_rank() == 0:

            if self._args.task == 'entity_and_entity_type':
                task_output_dir = os.path.join(self._args.output_dir,
                                               self._args.task)
                if not os.path.exists(task_output_dir):
                    os.makedirs(task_output_dir)

                # get current date and time
                current_datetime = str(datetime.now())
                results_file = os.path.join(
                    task_output_dir,
                    f'{self._args.task}_result_{current_datetime}.txt')
                ru.write_json(preprocessed, results_file)

            else:
                if len(preprocessed) == 3:
                    src, pred, true = preprocessed
                else:
                    src = None
                    pred, true = preprocessed

                if hasattr(self._data_processor.data_reader, 'rel_types'):
                    labels = self._data_processor.data_reader.rel_types
                elif hasattr(self._data_processor.data_reader, 'ent_types'):
                    labels = self._data_processor.data_reader.ent_types
                elif self._args.task in ['relation', 'entity_relation']:
                    labels = self._data_processor.relation_query_answers
                else:
                    labels = None

                processing_time = timedelta(self._generate_time +
                                            self._tune_time +
                                            self._inference_time)

                scores = compute_metrics(pred, true)
                save_results(
                    scores=scores,
                    src=src,
                    orig=results[self._args.task]['original'],
                    pred=pred,
                    true=true,
                    task=self._args.task,
                    labels=labels,
                    output_dir=self._args.output_dir,
                    num_processes=self._args.state.num_processes,
                    batch_size=self._args.test_batch_size,
                    n_shots=self._args.n_shots,
                    test_sample_size=len(self._data_processor.test_dataset),
                    model_config=self._model.config,
                    generation_config=self._args.generation_config,
                    task_prompt=self._data_processor.task_prompt[self._args.task],
                    data_name=self._args.data_name,
                    num_of_kbase_classes=self._args.num_of_kbase_classes,
                    exec_time=str(processing_time),
                )

    def stop(self) -> None:
        self.save_results()
        super().stop()


if __name__ == '__main__':

    service = ModelService(args=get_args())
    service.start()

    # consider to use ru.zmq.Registry
    ru.write_json({'service_addr': service.addr}, 'model_service_reg.json')
    # service.wait()

    # --- FOR TEST PURPOSES ---
    client = ru.zmq.Client(url=service.addr)
    prompt_response = client.request('processor', {'action': 'generate'})
    print(prompt_response)
    prompt_response = client.request('processor', {'action': 'tune'})
    print(prompt_response)
    prompt_response = client.request('processor', {'action': 'inference'})
    print(prompt_response)
    client.close()
    # --- ^^^^^^^^^^^^^^^^^ ---

    service.stop()

