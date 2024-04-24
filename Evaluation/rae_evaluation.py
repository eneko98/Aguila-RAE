import os
import re
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_metric

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
checkpoint_path = "Training/results/aguila-rae/checkpoint-5500"

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)

rouge_metric = load_metric("rouge")
bertscore_metric = load_metric("bertscore")

def generate_definition(model, tokenizer, word, pos, max_length=150, top_p=0.9, temperature=0.9, num_beams=3, do_sample=True):
    prompt = f"[BOS] {word} (POS: {pos}) <definition>"

    encoding = tokenizer.encode_plus(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=do_sample,
            top_k=50,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    definition = tokenizer.decode(output[0], skip_special_tokens=False)

    definition = definition.split('<definition>')[1] if '<definition>' in definition else definition
    
    patterns_to_remove = [
        r"\[C\.\s*Rica\..*$",
        r"\[Cen\.\s*y\s*Am\..*$",
    ]

    for pattern in patterns_to_remove:
        definition = re.sub(pattern, '', definition, flags=re.IGNORECASE | re.DOTALL)
        
    definition = re.sub(r"\[BOS\]\s?", "", definition)  # Removes [BOS] and any following space
    definition = re.sub(r"\s?\[EOS\]", "", definition)  # Removes [EOS] and any preceding space
    definition = re.sub(r"\.\.", ".", definition)  # Replace two dots with one dot
    definition = definition.strip(", ")

    return definition

csv_file_path = "Evaluation/evaluation_results.csv"

with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    if file.tell() == 0:
        csv_writer.writerow(["Word", "POS", "Generated Definition", "Expected Definition", "ROUGE-L F1", "BERTScore F1"])
    
    try:
        while True:
            word = input("Enter the word (or 'quit' to stop): ")
            if word.lower() == 'quit':
                break
            pos = input("Enter the part of speech: ")

            generated_definition = generate_definition(model, tokenizer, word, pos)
            print(f"Generated Definition: {generated_definition}\n")

            expected_definition = input("Enter the expected definition for metric calculation: ")

            rouge_score = rouge_metric.compute(predictions=[generated_definition], references=[expected_definition])
            bertscore_result = bertscore_metric.compute(predictions=[generated_definition], references=[expected_definition], lang="es")

            print(f"ROUGE-L F1: {rouge_score['rougeL'].mid.fmeasure}")
            print(f"BERTScore F1: {bertscore_result['f1'][0]}")
            print("\n---\n")

            csv_writer.writerow([
                word,
                pos,
                generated_definition,
                expected_definition,
                rouge_score['rougeL'].mid.fmeasure,
                bertscore_result['f1'][0]
            ])
            file.flush()
    except Exception as e:
        print(f"An error occurred: {e}")

print(f"Results saved to {csv_file_path}")
