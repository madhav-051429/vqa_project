from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
import os

def setup_llama_model(token=None):

    if token is None:
        token = os.environ.get("HF_TOKEN")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        quantization_config=bnb_config, 
        device_map="auto",
        token=token  
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        token=token  
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    model = prepare_model_for_kbit_training(base_model)
    
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM", 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    lora_model = get_peft_model(model, lora_config)
    return lora_model, tokenizer

class LlamaVQADataset(torch.utils.data.Dataset):
    def __init__(self, questions_file, annotations_file, tokenizer, max_length=512, max_samples=None):
        import json
        with open(questions_file, 'r') as f:
            self.questions = json.load(f)['questions']
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)['annotations']
        if max_samples:
            self.questions = self.questions[:max_samples]
            self.annotations = self.annotations[:max_samples]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        annotation = self.annotations[idx]
        question_text = question['question']
        answer = annotation['multiple_choice_answer']
        prompt = f"<s>### Question: {question_text}\n### Answer: {answer}</s>"
        encodings = self.tokenizer(prompt, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        encodings['labels'] = encodings['input_ids'].clone()
        for k in encodings:
            encodings[k] = encodings[k].squeeze(0)
        return encodings
