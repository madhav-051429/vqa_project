import os
import json
from PIL import Image
from torch.utils.data import Dataset

class VQADataset(Dataset):
    def __init__(self, questions_file, annotations_file, image_dir, processor, max_samples=None):
        with open(questions_file, 'r') as f:
            self.questions = json.load(f)['questions']
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)['annotations']
        if max_samples:
            self.questions = self.questions[:max_samples]
            self.annotations = self.annotations[:max_samples]
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        annotation = self.annotations[idx]
        image_id = question['image_id']
        image_path = os.path.join(self.image_dir, f'COCO_train2014_{image_id:012d}.jpg')
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        question_text = question['question']
        answer = annotation['multiple_choice_answer']
        inputs = self.processor(text=[question_text, answer], images=image, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
        for k in inputs:
            if hasattr(inputs[k], 'squeeze'):
                inputs[k] = inputs[k].squeeze(0)
        return inputs
