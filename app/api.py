from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM

clip_model = CLIPModel.from_pretrained("/teamspace/studios/this_studio/vqa_project/scripts/models/fine_tuned_clip")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
llama_model = AutoModelForCausalLM.from_pretrained("/teamspace/studios/this_studio/vqa_project/scripts/models/fine_tuned_llama_lora")
llama_tokenizer = AutoTokenizer.from_pretrained("/teamspace/studios/this_studio/vqa_project/scripts/models/fine_tuned_llama_lora")

app = FastAPI(title="Visual Question Answering API")

@app.post("/process_vqa")
async def process_vqa(image: UploadFile = File(...), question: str = Form(...)):
    image_content = await image.read()
    pil_image = Image.open(io.BytesIO(image_content)).convert("RGB")
    inputs = clip_processor(text=question, images=pil_image, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    with torch.no_grad():
        clip_outputs = clip_model(**inputs)
        image_features = clip_outputs.image_embeds
        text_features = clip_outputs.text_embeds
    similarity = torch.nn.functional.cosine_similarity(image_features, text_features).item()
    prompt = f"<s>### Image and Question Analysis\nSimilarity Score: {similarity:.4f}\n### Question: {question}\n### Answer:"
    llama_inputs = llama_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = llama_model.generate(**llama_inputs, max_new_tokens=50)
    generated_text = llama_tokenizer.decode(output[0], skip_special_tokens=True)
    answer = generated_text.split("### Answer:")[-1].strip()
    return {"question": question, "answer": answer, "clip_similarity": similarity}
