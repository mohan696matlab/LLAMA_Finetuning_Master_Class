from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_scheduler
from bitsandbytes.optim import Adam8bit,PagedAdam32bit
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from peft import prepare_model_for_kbit_training
import torch
from IPython.display import  clear_output
import time
import gc
import os
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from evaluation_utlis import rouge_bleu_score,get_prediction_per_sentence

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DEFAULT_MODEL = "/home/nas/buffer/mohan.dash/llama_3_2_3B"#"meta-llama/Llama-3.2-3B-Instruct","/home/nas/buffer/mohan.dash/llama_3_2_3B"
TOKENIZER_PATH = "llama_odia_tokenizer"
LORA_ADAPTER_DIR = '/home/nas/buffer/mohan.dash/llama_3_finetuned/adapter'
OPTIMIZER_CKPT_DIR = '/home/nas/buffer/mohan.dash/llama_3_finetuned'
MAX_LENGTH = 256
BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 4
EVAL_STEPS=10
MAX_STEPS=5000
CONTINUE_FROM_CHECKPOINT = False


bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )


model = AutoModelForCausalLM.from_pretrained(
    DEFAULT_MODEL,
    quantization_config=bnb_config,
    use_safetensors=True,
    device_map=device,
)

print(model.get_memory_footprint()/(1024*1024)) 

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_safetensors=True)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Resize the model's token embeddings to match the tokenizer's vocab size
model.resize_token_embeddings(len(tokenizer))
    
dataset = load_dataset('OdiaGenAI/odia_domain_context_train_v1')

class LlamaDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question=sample['instruction']
        answer = sample['output']
        prompt = f'''<|begin_of_text|> <|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
        full_text = prompt+f'''{answer}।<|eot_id|>'''

        tokenized = tokenizer(full_text, truncation=True, add_special_tokens=False, padding="max_length", max_length=MAX_LENGTH)

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
        
train_dataset = LlamaDataset(dataset['train'])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = prepare_model_for_kbit_training(model)

if CONTINUE_FROM_CHECKPOINT:
    embedding_state_dict = torch.load(f"{OPTIMIZER_CKPT_DIR}/embedding_weights.pt", map_location=device)
    lm_head_state_dict = torch.load(f"{OPTIMIZER_CKPT_DIR}/lm_head_weights.pt", map_location=device)
    # Load the trained embeddings and LM head
    model.model.embed_tokens.load_state_dict(embedding_state_dict)
    model.lm_head.load_state_dict(lm_head_state_dict)
    
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_DIR, is_trainable=True) # Biggest change in this script
    
else:
    config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
        use_rslora=True,
        init_lora_weights="gaussian",
    )

    model = get_peft_model(model, config)

# Keep the embedding and the Llama head trainable
model.lm_head.weight.requires_grad = True
model.model.model.embed_tokens.weight.requires_grad = True

def generate_eval(model,idx=5,disable_lora=False):
    
    model.config.use_cache = True
    sample=dataset['train'][idx]
    question=sample['instruction']
    answer = sample['output']
    chat_template = f'''<|begin_of_text|> <|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
    inputs = tokenizer(chat_template , return_tensors="pt").to(device)
    # print(prompt)

    model.eval()


    if disable_lora:
        with model.disable_adapter():
            output = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=MAX_LENGTH,
                repetition_penalty=1.3,
                temperature=0.7,         # Optional: smooth randomness
                top_k=50,                # Optional: top-k sampling
                top_p=0.9                # Optional: nucleus sampling
            )
    else:
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
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
model.config.use_cache = False
model.config.pretraining_tp = 1

max_loss = 1e9
global_step= 0


# Define optimizer
optimizer = PagedAdam32bit(model.parameters(), lr=2e-4)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
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
            llm_score = rouge_bleu_score(model,tokenizer,dataset['train'],
                                         current_step=global_step,
                                         saving_path=OPTIMIZER_CKPT_DIR,
                                         max_new_tokens=100,
                                         device=device)
            
            
        if loss.item() < max_loss:
            max_loss = loss.item()
            save_path = f"{OPTIMIZER_CKPT_DIR}/model_checkpoint.pt"
            
            model.save_pretrained(LORA_ADAPTER_DIR)
            # torch.save({
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            #     'global_step': global_step
            # }, save_path)
            
            # torch.save(model.model.model.embed_tokens.state_dict(), f"{OPTIMIZER_CKPT_DIR}/embedding_weights.pt")
            # torch.save(model.lm_head.state_dict(), f"{OPTIMIZER_CKPT_DIR}/lm_head_weights.pt")
            
            
            
        
        print(f"Epoch {global_step + 1}/{MAX_STEPS}, Loss: {loss.item():.4f}")