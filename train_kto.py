import sys
sys.path.append('/mnt/hdd/alignment/trl')
# sys.path.append('/mnt/hdd/alignment/trl/trl')
# sys.path.append('/mnt/hdd/alignment/trl/trl/trainer')

from trl.trainer.kto_trainer import KTOTrainer
from trl.trainer.kto_config import KTOConfig

from datasets import load_dataset, Dataset, concatenate_datasets
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from argparse import ArgumentParser
import wandb
import os
import torch


parser = ArgumentParser()
parser.add_argument('--lr', type=float)
args = parser.parse_args()

MAX_LENGTH         = 1024
NUM_PROCESSES      = 16
SEED               = 42
TEST_SIZE          = 0.1
LR                 = args.lr
OUTPUT_DIR         = f'models/kto_model_{LR}'
BATCH_SIZE         = 1
MODEL_NAME         = 'models/sft_model'
ACCUMULATION_STEPS = 32

BETA               = 0.1
MAX_SEQ_LENGTH     = 1024
MAX_PROMPT_LENGHT  = 128
LOGGING_STEPS      = 10
NUM_EPOCHS         = 1
WARMUP             = 100
MAX_GRAD_NORM      = 1.0
RUN_NAME           = 'test'
SAVE_STEPS         = 1000
REPORT_TO          = 'wandb'

wandb.init(project="Alignment", entity="sashnevskiy")
os.environ["WANDB_PROJECT"]="Alignment"
os.environ["WANDB_LOG_MODEL"]="false"
os.environ["WANDB_WATCH"]="false"

chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.chat_template = chat_template
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map = 'cuda:0')

def filter_dataset(dataset):
    # drop all samples that have prompt + max(chosen, rejected) > max_seq_length
    _apply_template = lambda x: tokenizer.apply_chat_template(x, tokenize=True, add_generation_prompt=True)
    
    dataset = dataset.map(
        lambda x: {
            'length_chosen': len(_apply_template(x['chosen'])),
            'length_rejected': len(_apply_template(x['rejected'])),
            'length_prompt': len(_apply_template(x['prompt']))
        },
        num_proc=NUM_PROCESSES
    )
    dataset = dataset.filter(
        lambda x: max(x['length_chosen'],x['length_rejected']) < MAX_LENGTH - x['length_prompt']
    )
    dataset = dataset.remove_columns(['length_chosen', 'length_rejected', 'length_prompt'])
    return dataset


def apply_chat_template(example, tokenizer):
    prompt_messages = example["chosen"][:-1]
    chosen_messages = example["chosen"][-1:]
    rejected_messages = example["rejected"][-1:]
    
    res = {}
    res["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
    res["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    res["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
    return res

def convert_dataset(dataset):
    # preprocess dataset and convert to like/dislike dataset
    dataset = filter_dataset(dataset)
    _apply_chat_template = partial(apply_chat_template, tokenizer=tokenizer)
    dataset = dataset.map(_apply_chat_template, num_proc=NUM_PROCESSES)
    
    prompts, pos_labels, neg_labels = (
        dataset['text_prompt'], 
        [True]*len(dataset['text_prompt']), 
        [False]*len(dataset['text_prompt'])
    )
    
    _get_dataset = lambda x, y, z: Dataset.from_dict({'prompt': x, 'completion': y, 'label': z})
    pos_dataset = _get_dataset(prompts, dataset['text_chosen'], pos_labels)
    neg_dataset = _get_dataset(prompts, dataset['text_rejected'], neg_labels)
    
    concatenate_dataset = concatenate_datasets([pos_dataset, neg_dataset])
    return concatenate_dataset

dataset = load_dataset('HuggingFaceH4/ultrafeedback_binarized')
split_dataset = dataset['train_prefs'].train_test_split(test_size=TEST_SIZE, seed=SEED)

train_dataset = convert_dataset(split_dataset['train'])
train_dataset = train_dataset.shuffle(seed=SEED)

val_dataset = convert_dataset(split_dataset['test'])
val_dataset = val_dataset.shuffle(seed=SEED)

training_args = KTOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACCUMULATION_STEPS,
    learning_rate=LR,
    logging_steps=LOGGING_STEPS,
    # evaluation_strategy="steps",
    gradient_checkpointing=False,
    # eval_steps=EVAL_STEPS,
    # per_device_eval_batch_size=BATCH_SIZE,
    # do_eval=True,
    # max_grad_norm=MAX_GRAD_NORM,
    max_length=MAX_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGHT,
    beta=BETA,
    do_train=True,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=WARMUP,
    run_name=RUN_NAME,
    save_steps=SAVE_STEPS,
    save_only_model=False,
    bf16=True,
    report_to=REPORT_TO,
)

trainer = KTOTrainer(
    model,
    ref_model=None,
    args=training_args,
    # peft_config=peft_config,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    # eval_dataset=val_dataset,
)

trainer.train()
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)