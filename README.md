# Visual Question Answering (VQA) Multimodal AI System

A state-of-the-art Visual Question Answering system that combines computer vision and natural language processing to answer questions about images. Built with **Lightning AI platform** for scalable training and deployment, this project integrates CLIP for image understanding and Mistral-7B for language processing, utilizing LoRA for efficient fine-tuning.

## 🚀 Features

- **Lightning AI Orchestration**: Complete ML pipeline management with `LightningApp` and `LightningFlow`
- **Multimodal Architecture**: CLIP-ViT-Base-Patch32 + Mistral-7B-v0.1 integration
- **Efficient Fine-tuning**: LoRA (Low-Rank Adaptation) for parameter-efficient training
- **Production Ready**: FastAPI backend with RESTful API endpoints
- **Interactive Interface**: Gradio web demo for real-time user interaction
- **Memory Optimized**: BitsAndBytes quantization for reduced VRAM usage
- **Structured Training**: PyTorch Lightning modules for robust training loops

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Image   │    │   Text Question │    │   Lightning AI  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
    ┌─────────┐              ┌─────────┐          ┌─────────┐
    │  CLIP   │              │Mistral-7B│          │Workflow │
    │Encoder  │              │   LLM   │          │Manager  │
    └─────┬───┘              └─────┬───┘          └─────────┘
          │                        │
          ▼                        ▼
    ┌─────────────────────────────────┐
    │    Multimodal Fusion Layer      │
    │         (LoRA Adapted)          │
    └─────────────┬───────────────────┘
                  ▼
            ┌─────────┐
            │ Answer  │
            └─────────┘
```

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- Lightning AI account (for platform features)

## 🛠️ Installation

1. **Clone the Repository**
   ```
   git clone https://github.com/madhav-051429/vqa_project.git
   cd vqa_project
   ```

2. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Set Up Lightning AI** (Optional for local use)
   ```
   lightning login
   ```

## 📁 Project Structure

```
vqa_project/
├── app/
│   ├── api.py                 # FastAPI backend
│   └── gradio_interface.py    # Gradio web interface
├── scripts/
│   ├── train_clip.py         # CLIP training script
│   └── train_llama.py        # LLaMA training script
├── src/
│   ├── models/               # Model definitions
│   ├── data/                 # Data loading utilities
│   └── utils/                # Helper functions
├── lightning_app.py          # Lightning AI orchestrator
├── main.py                   # Main application entry
└── requirements.txt          # Dependencies
```

## 🚀 Usage

### Local Development

1. **Run the FastAPI Backend**
   ```
   uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Launch Gradio Interface**
   ```
   python app/gradio_interface.py
   ```

3. **Access the Application**
   - API Documentation: `http://localhost:8000/docs`
   - Gradio Demo: `http://localhost:7860`

### Lightning AI Platform

1. **Run the Lightning App**
   ```
   lightning run app lightning_app.py
   ```

2. **Monitor Training**
   ```
   lightning run app lightning_app.py --cloud
   ```

### Training

1. **Train CLIP Model**
   ```
   python scripts/train_clip.py --epochs 10 --batch_size 32
   ```

2. **Fine-tune Mistral with LoRA**
   ```
   python scripts/train_llama.py --lora_rank 16 --learning_rate 5e-5
   ```

## 💡 API Usage

### Predict Endpoint

```
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "image_url": "https://example.com/image.jpg",
       "question": "What is in this image?"
     }'
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "image_url": "path/to/image.jpg",
        "question": "How many people are in the image?"
    }
)
print(response.json())
```

## 📊 Model Performance

- **Dataset**: VQA v2.0
- **Architecture**: CLIP + Mistral-7B with LoRA
- **Training**: Parameter-efficient fine-tuning
- **Inference**: Optimized for real-time performance

## 🔧 Configuration

Key configuration parameters in `config.py`:

```python
MODEL_CONFIG = {
    "clip_model": "openai/clip-vit-base-patch32",
    "llm_model": "mistralai/Mistral-7B-v0.1",
    "lora_rank": 16,
    "quantization": "4bit",
    "max_length": 512
}
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use smaller LoRA rank

2. **Model Loading Errors**
   - Check internet connection for model downloads
   - Verify transformer library version compatibility
   - Ensure sufficient disk space

3. **Lightning AI Connection Issues**
   - Verify Lightning AI credentials
   - Check network connectivity
   - Update Lightning CLI

### Memory Requirements

- **Training**: 16GB+ VRAM recommended
- **Inference**: 8GB+ VRAM minimum
- **CPU Inference**: 32GB+ RAM (slower performance)

## 🧪 Testing

Run the test suite:
```
python -m pytest tests/ -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [VQA v2 Dataset](https://visualqa.org/)
- [Lightning AI Documentation](https://lightning.ai/docs/)


---

⭐ **Star this repository if you found it helpful!**
