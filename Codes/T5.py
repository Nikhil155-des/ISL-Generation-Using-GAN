import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from typing import List
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class TextToSOVConverter:
    def __init__(self, model_name: str = "t5-small"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess input text by adding task prefix"""
        return f"convert to SOV: {text}"
    
    def convert_to_sov(self, sentences: List[str], max_length: int = 128) -> List[str]:
        """Convert input sentences to SOV format"""
        processed_texts = [self.preprocess_text(sent) for sent in sentences]
        
        inputs = self.tokenizer(
            processed_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        
        sov_sentences = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return sov_sentences

class SOVDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        svo_sentence = self.data.iloc[idx]['English Sentence (SVO)']
        sov_sentence = self.data.iloc[idx]['SOV for ISL']
        
        # Prepare input
        input_text = f"convert to SOV: {svo_sentence}"
        
        # Tokenize input and target
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            sov_sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': input_encoding.input_ids.squeeze(),
            'attention_mask': input_encoding.attention_mask.squeeze(),
            'labels': target_encoding.input_ids.squeeze()
        }

if __name__ == "__main__":
    # Initialize converter
    converter = TextToSOVConverter()
    
    # Load the dataset
    try:
        data = pd.read_csv('enhanced_svo_to_sov_dataset.csv')  # Make sure this CSV is in the correct path
        print(f"Loaded {len(data)} sentence pairs")
        
        # Split the dataset into train and test sets (80-20 split)
        train_data, test_data = train_test_split(
            data, 
            test_size=0.2, 
            random_state=42
        )
        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")
        
        # Create training dataset
        train_dataset = SOVDataset(train_data, converter.tokenizer)
        test_dataset = SOVDataset(test_data, converter.tokenizer)
        
        # Update training arguments to include evaluation
        training_args = TrainingArguments(
            output_dir="./sov_converter",
            num_train_epochs=20,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )
        
        # Initialize trainer with both train and eval datasets
        trainer = Trainer(
            model=converter.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the trained model and tokenizer
        model_save_path = "./trained_sov_converter"
        converter.model.save_pretrained(model_save_path)
        converter.tokenizer.save_pretrained(model_save_path)
        print(f"Model training completed and saved at {model_save_path}!")
        
        # Test the model with examples from test set
        print("\nTesting the trained model with examples from test set:")
        test_examples = test_data.head(10)  # Take first 5 examples from test set
        
        for _, row in test_examples.iterrows():
            original = row['English Sentence (SVO)']
            expected = row['SOV for ISL']
            predicted = converter.convert_to_sov([original])[0]
            
            print(f"Original: {original}")
            print(f"Expected SOV: {expected}")
            print(f"Predicted SOV: {predicted}")
            print()
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")