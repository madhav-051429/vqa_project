from src.models.llama_model import setup_llama_model, LlamaVQADataset
from transformers import Trainer, TrainingArguments
import os

token = os.getenv("HF_TOKEN")

llama_model, llama_tokenizer = setup_llama_model(token=token)
llama_dataset = LlamaVQADataset(
    questions_file='/teamspace/studios/this_studio/vqa_project/data/v2_OpenEnded_mscoco_train2014_questions.json',
    annotations_file='/teamspace/studios/this_studio/vqa_project/data/v2_mscoco_train2014_annotations.json',
    tokenizer=llama_tokenizer,
    max_samples=10000
)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    eval_strategy="no",
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="steps",
    save_steps=500,
    learning_rate=2e-4,
    fp16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="none"
)
trainer = Trainer(model=llama_model, args=training_args, train_dataset=llama_dataset)
trainer.train()
llama_model.save_pretrained("models/fine_tuned_llama_lora")
llama_tokenizer.save_pretrained("models/fine_tuned_llama_lora")
