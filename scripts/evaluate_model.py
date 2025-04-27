from src.evaluation.metrics import calculate_vqa_metrics
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image
import json

clip_model = CLIPModel.from_pretrained("/teamspace/studios/this_studio/vqa_project/scripts/models/fine_tuned_clip")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
llama_model = AutoModelForCausalLM.from_pretrained("/teamspace/studios/this_studio/vqa_project/scripts/models/fine_tuned_llama_lora")
llama_tokenizer = AutoTokenizer.from_pretrained("/teamspace/studios/this_studio/vqa_project/scripts/models/fine_tuned_llama_lora")

with open('/teamspace/studios/this_studio/vqa_project/data/v2_OpenEnded_mscoco_val2014_questions.json', 'r') as f:
    val_questions = json.load(f)['questions']
with open('/teamspace/studios/this_studio/vqa_project/data/v2_mscoco_val2014_annotations.json', 'r') as f:
    val_annotations = json.load(f)['annotations']
question_id_to_annotation = {a['question_id']: a for a in val_annotations}
predictions, ground_truth = [], []
for question in val_questions[:100]:
    image_id = question['image_id']
    question_text = question['question']
    annotation = question_id_to_annotation[question['question_id']]
    answer = annotation['multiple_choice_answer']
    image_path = f"data/val2014/COCO_val2014_{image_id:012d}.jpg"
    image = Image.open(image_path).convert('RGB')
    inputs = clip_processor(text=question_text, images=image, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    with torch.no_grad():
        clip_outputs = clip_model(**inputs)
        image_features = clip_outputs.image_embeds
        text_features = clip_outputs.text_embeds
    similarity = torch.nn.functional.cosine_similarity(image_features, text_features).item()
    prompt = f"<s>### Image and Question Analysis\nSimilarity Score: {similarity:.4f}\n### Question: {question_text}\n### Answer:"
    llama_inputs = llama_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = llama_model.generate(**llama_inputs, max_new_tokens=50)
    generated_text = llama_tokenizer.decode(output[0], skip_special_tokens=True)
    predicted_answer = generated_text.split("### Answer:")[-1].strip()
    predictions.append(predicted_answer.lower())
    ground_truth.append(answer.lower())
metrics = calculate_vqa_metrics(predictions, ground_truth)
print(metrics)
