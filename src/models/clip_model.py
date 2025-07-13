import torch
import torch.nn.functional as F
from transformers import CLIPModel
import pytorch_lightning as pl

class CLIPFineTuner(pl.LightningModule):
    def __init__(self, learning_rate=5e-5, weight_decay=0.01):
        super().__init__()
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, input_ids, attention_mask, pixel_values):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pixel_values = batch['pixel_values']
        
       
        question_input_ids = input_ids[:, 0, :]
        question_attention_mask = attention_mask[:, 0, :]
        answer_input_ids = input_ids[:, 1, :]
        answer_attention_mask = attention_mask[:, 1, :]
        
        
        question_outputs = self.model(
            input_ids=question_input_ids, 
            attention_mask=question_attention_mask, 
            pixel_values=pixel_values
        )
        
        
        answer_embeds = self.model.get_text_features(
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask
        )
        
        
        image_embeds = question_outputs.image_embeds
        question_embeds = question_outputs.text_embeds
        
       
        image_question_similarity = F.cosine_similarity(image_embeds, question_embeds)
        image_answer_similarity = F.cosine_similarity(image_embeds, answer_embeds)
        
        
        loss = -torch.mean(image_answer_similarity - image_question_similarity + 0.1)
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pixel_values = batch['pixel_values']
        
        
        question_input_ids = input_ids[:, 0, :]
        question_attention_mask = attention_mask[:, 0, :]
        answer_input_ids = input_ids[:, 1, :]
        answer_attention_mask = attention_mask[:, 1, :]
        
        
        question_outputs = self.model(
            input_ids=question_input_ids, 
            attention_mask=question_attention_mask, 
            pixel_values=pixel_values
        )
        
        
        answer_embeds = self.model.get_text_features(
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask
        )
        
        
        image_embeds = question_outputs.image_embeds
        question_embeds = question_outputs.text_embeds
        
        
        image_question_similarity = F.cosine_similarity(image_embeds, question_embeds)
        image_answer_similarity = F.cosine_similarity(image_embeds, answer_embeds)
        
        
        val_loss = -torch.mean(image_answer_similarity - image_question_similarity + 0.1)
        self.log('val_loss', val_loss, prog_bar=True)
        
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer
