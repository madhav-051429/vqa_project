from transformers import CLIPProcessor
from src.models.clip_model import CLIPFineTuner
from src.dataa.dataset import VQADataset
import pytorch_lightning as pl

clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
train_dataset = VQADataset(
    questions_file='/teamspace/studios/this_studio/vqa_project/data/v2_OpenEnded_mscoco_train2014_questions.json',
    annotations_file='/teamspace/studios/this_studio/vqa_project/data/v2_mscoco_train2014_annotations.json',
    image_dir='/teamspace/studios/this_studio/vqa_project/data/train2014',
    processor=clip_processor,
    max_samples=10000
)
train_loader = pl.LightningDataModule.from_datasets(train_dataset)
clip_model = CLIPFineTuner()
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=3, precision="16-mixed")
trainer.fit(clip_model, train_loader)
clip_model.model.save_pretrained("models/fine_tuned_clip")
