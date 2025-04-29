from transformers import Gemma3ForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig, get_scheduler
from bitsandbytes.optim import Adam8bit,PagedAdam32bit
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from peft import prepare_model_for_kbit_training
import torch
from IPython.display import  clear_output
import numpy as np
import time
import gc
import os,json
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from evaluation_utlis import rouge_bleu_score,get_prediction_per_sentence
from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DEFAULT_MODEL = "google/gemma-3-4b-it"#"meta-llama/Llama-3.2-3B-Instruct","/home/nas/buffer/mohan.dash/llama_3_2_3B"
LORA_ADAPTER_DIR = 'runs/lora_adapter'
OPTIMIZER_CKPT_DIR = 'runs'
MAX_LENGTH = 256
BATCH_SIZE = 1
GRAD_ACCUMULATION_STEPS = 4
EVAL_STEPS=10
MAX_STEPS=5000
CONTINUE_FROM_CHECKPOINT = False

os.makedirs(LORA_ADAPTER_DIR, exist_ok=True)


def flush():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
def print_gpu_utilization():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"GPU Memory Usage>>>> Allocated: {allocated:.2f} MB |||||  Reserved:  {reserved:.2f} MB:")   
 



bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_storage=torch.bfloat16,
                                )


model = Gemma3ForConditionalGeneration.from_pretrained(
    DEFAULT_MODEL,
    quantization_config=bnb_config,
    attn_implementation='eager',
    device_map={'':torch.cuda.current_device()},
    torch_dtype=torch.bfloat16
    
)

print_gpu_utilization()

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, use_safetensors=True)
tokenizer.pad_token_id = tokenizer.eos_token_id

   
dataset = load_dataset('OdiaGenAI/odia_domain_context_train_v1')

class GemmaDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question=sample['instruction']
        answer = sample['output']
        prompt = f'''<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n'''
        full_text = prompt+f'''{answer}<end_of_turn>'''

        tokenized = tokenizer(full_text,add_special_tokens=False,truncation=True,max_length=250)

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Tokenize just the prompt to get the split point
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_start = len(prompt_ids)

        # Mask everything before answer_start
        labels = [-100] * answer_start + input_ids[answer_start:]
        # Mask out padding as well
        labels = [
            label if token != tokenizer.pad_token_id else -100
            for label, token in zip(labels, input_ids)
        ]
    
        return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels)
    }
        

def gemma_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad sequences to the max length in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }
        
from torch.utils.data import random_split, DataLoader

# Assume dataset['train'] is the full dataset you want to split
full_dataset = dataset['train']


# Split the dataset
train_data, test_data = random_split(full_dataset, [0.99, 0.01])

# Wrap in your custom Dataset class
train_dataset = GemmaDataset(train_data)
test_dataset = GemmaDataset(test_data)

# DataLoaders with custom collate_fn
train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=gemma_collate_fn
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=gemma_collate_fn
)



config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
    use_rslora=True,
    init_lora_weights="gaussian",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
print_gpu_utilization()


def generate_eval(model,idx=5):
    
    model.config.use_cache = True
    sample=dataset['train'][idx]
    question=sample['instruction']
    answer = sample['output']
    chat_template = f'''<|begin_of_text|> <|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
    inputs = tokenizer(chat_template , return_tensors="pt").to(device)
 
    model.eval()

    output = model.generate(
    **inputs,
    do_sample=True,
    max_new_tokens=MAX_LENGTH,
    repetition_penalty=1.3,
    temperature=0.7,         # Optional: smooth randomness
    top_k=50,                # Optional: top-k sampling
    top_p=0.9                # Optional: nucleus sampling
    )

    processed_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    model.train()

    return processed_text

def get_validation_loss(model,test_dataloader):
    model.eval()
    total_val_loss = 0
    count = 0
    with torch.inference_mode():
        pbar = tqdm(test_dataloader, total=len(test_dataloader), leave=False)
        for val_batch in pbar:
            val_outputs = model(
                input_ids=val_batch['input_ids'].to('cuda'),
                attention_mask=val_batch['attention_mask'].to('cuda'),
                labels=val_batch['labels'].to('cuda')
            )
            total_val_loss += val_outputs.loss.item()
            count += 1
            pbar.set_postfix({'val_loss': f'{(total_val_loss / count):.4f}'})
    avg_val_loss = total_val_loss / count
    return avg_val_loss

def save_loss_plot(train_losses, val_losses, path):
    plt.figure(figsize=(10, 6))
    plt.plot(
        [i * EVAL_STEPS for i in range(len(train_losses))],
        train_losses, label="Train Loss", color='blue')
    plt.plot(
        [i * EVAL_STEPS for i in range(len(val_losses))],
        val_losses,
        label="Validation Loss",
        color='orange'
    )
    plt.xlabel("Global Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
model.config.use_cache = False
model.config.pretraining_tp = 1

max_loss = 1e9
global_step= 0

train_losses=[]
val_losses=[]
loss_buffer=[]


# Define optimizer
optimizer = PagedAdam32bit(model.parameters(), lr=2e-4)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * MAX_STEPS),
    num_training_steps=MAX_STEPS,
)

if CONTINUE_FROM_CHECKPOINT:
    # Load checkpoint
    checkpoint = torch.load(f'{OPTIMIZER_CKPT_DIR}/model_checkpoint.pt', map_location=device) # Biggest chnage in this script

    # Restore model, optimizer, scheduler, and step
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    global_step = checkpoint['global_step']



    print('>'*30,f"Checkpoint loaded from {OPTIMIZER_CKPT_DIR} at step {global_step}")


# Training loop
model.train()

while global_step< MAX_STEPS:
    for step,batch in enumerate(train_dataloader):
        
        model.config.use_cache = False
        model.train()
        
        # Forward pass
        outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))
        loss = outputs.loss
        loss_buffer.append(loss.item())
        loss = loss / GRAD_ACCUMULATION_STEPS  # Normalize loss
        loss.backward()
        
        if (step + 1) % GRAD_ACCUMULATION_STEPS == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        global_step += 1
        if global_step >= MAX_STEPS:
            break
        
        if global_step % EVAL_STEPS == 0:
            
            flush()
            
            val_losses.append(get_validation_loss(model,test_dataloader))
            train_losses.append(np.mean(loss_buffer[-EVAL_STEPS:]))
            
            llm_score = rouge_bleu_score(model,tokenizer,dataset['train'],
                                         current_step=global_step,
                                         saving_path=OPTIMIZER_CKPT_DIR,
                                         loss=loss.item(),
                                         max_new_tokens=200,
                                         device=device)
            
            save_path = f"{OPTIMIZER_CKPT_DIR}/model_checkpoint.pt"
            
            model.save_pretrained(LORA_ADAPTER_DIR)
            torch.save({
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'global_step': global_step
            }, save_path)
            
            with open(f"{OPTIMIZER_CKPT_DIR}/loss_log.json", "w") as f:
                json.dump({
                    "train_losses": train_losses,
                    "val_losses": val_losses
                }, f)
                
            save_loss_plot(train_losses, val_losses, path=f"{OPTIMIZER_CKPT_DIR}/loss_plot.png")
            
            
            
            # torch.save(model.model.model.embed_tokens.state_dict(), f"{OPTIMIZER_CKPT_DIR}/embedding_weights.pt")
            # torch.save(model.lm_head.state_dict(), f"{OPTIMIZER_CKPT_DIR}/lm_head_weights.pt")
            
            
            
        
        print(f"Epoch {global_step + 1}/{MAX_STEPS}, Loss: {loss_buffer[-1]:.4f}")
        print_gpu_utilization()