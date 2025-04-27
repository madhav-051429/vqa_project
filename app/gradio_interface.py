import gradio as gr
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM

clip_model = CLIPModel.from_pretrained("/teamspace/studios/this_studio/vqa_project/scripts/models/fine_tuned_clip")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
llama_model = AutoModelForCausalLM.from_pretrained("/teamspace/studios/this_studio/vqa_project/scripts/models/fine_tuned_llama_lora")
llama_tokenizer = AutoTokenizer.from_pretrained("/teamspace/studios/this_studio/vqa_project/scripts/models/fine_tuned_llama_lora")

def vqa_function(image, question):
    inputs = clip_processor(text=question, images=image, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
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
    return answer, f"CLIP Similarity: {similarity:.4f}"

def create_gradio_interface():
    interface = gr.Interface(
        fn=vqa_function,
        inputs=[gr.Image(type="pil", label="Upload Image"), gr.Textbox(label="Enter your question about the image")],
        outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Diagnostic Information")],
        title="Visual Question Answering System",
        description="Upload an image and ask a question about it. The system will analyze the image and provide an answer."
    )
    return interface.queue()  # Adding queue() helps with multiple users

