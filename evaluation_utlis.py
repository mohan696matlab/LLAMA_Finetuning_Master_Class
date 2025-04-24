import evaluate
import numpy as np

def get_prediction_per_sentence(model,tokenizer,sample,max_new_tokens,device='cuda'):
    
    model.config.use_cache = True
    question=sample['instruction']
    true_text = sample['output']
    chat_template = f'''<|begin_of_text|> <|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
    inputs = tokenizer(chat_template , return_tensors="pt", add_special_tokens=False).to(device)

    model.eval()

    output = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=max_new_tokens,
    repetition_penalty=1.3,
    temperature=0.7,         # Optional: smooth randomness
    top_k=50,                # Optional: top-k sampling
    top_p=0.9                # Optional: nucleus sampling
    )

    pred_text = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=False)

    return pred_text,true_text

def rouge_bleu_score(model,tokenizer,dataset,max_new_tokens=100,device='cuda'):
    # 1. Load the ROUGE metric and BLEU
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    pred_texts,true_texts=[],[]
    for idx in [11,111,222,333,444]:
        sample=dataset[idx]
        pred_text,true_text = get_prediction_per_sentence(model,tokenizer,sample,max_new_tokens,device=device)
        pred_texts.append(pred_text)
        true_texts.append(true_text)
        
    result_rouge = rouge.compute(predictions=pred_texts, 
                                 references=true_texts,
                                 tokenizer=lambda x: x.split(),  # Tokenize by whitespace only
                                 )
    result_bleu = bleu.compute(predictions=pred_texts, 
                               references=true_texts,
                               tokenizer=lambda x: x.split(),  # Tokenize by whitespace only
                               )
    return {'rouge_score':np.mean(list(result_rouge.values())), 'bleu_score':result_bleu['bleu']}