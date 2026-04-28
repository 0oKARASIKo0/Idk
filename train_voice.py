"""
Training script for the Advanced Voice Analyzer.
Records voice samples and trains the emotion/speaker detection model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import speech_recognition as sr
import os
import json
import time
import logging
from pathlib import Path
from qwen_agent.advanced_voice import AdvancedVoiceAnalyzer, VoiceClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceDataset(Dataset):
    """Dataset for voice samples."""
    
    def __init__(self, audio_files: list, labels: list, sample_rate: int = 16000):
        self.audio_files = audio_files
        self.labels = labels
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio file
        audio_path = self.audio_files[idx]
        audio_data = np.load(audio_path)
        
        # Convert to tensor
        waveform = torch.FloatTensor(audio_data)
        
        # Pad or truncate to fixed length (5 seconds)
        target_length = self.sample_rate * 5
        if waveform.size(0) < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.size(0)))
        else:
            waveform = waveform[:target_length]
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return waveform, label


class VoiceTrainer:
    """Trainer for voice analysis models."""
    
    def __init__(self, model: VoiceClassifier, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # Loss functions
        self.emotion_criterion = nn.CrossEntropyLoss()
        self.speaker_criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_emotion_loss = 0
        total_speaker_loss = 0
        correct_emotion = 0
        correct_speaker = 0
        total = 0
        
        for batch_idx, (waveforms, labels) in enumerate(dataloader):
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(waveforms)
            
            # Compute losses
            emotion_loss = self.emotion_criterion(outputs['emotion_logits'], labels)
            speaker_loss = self.speaker_criterion(outputs['speaker_logits'], labels)
            total_loss = emotion_loss + speaker_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_emotion_loss += emotion_loss.item()
            total_speaker_loss += speaker_loss.item()
            
            _, predicted_emotion = torch.max(outputs['emotion_logits'], 1)
            _, predicted_speaker = torch.max(outputs['speaker_logits'], 1)
            total += labels.size(0)
            correct_emotion += (predicted_emotion == labels).sum().item()
            correct_speaker += (predicted_speaker == labels).sum().item()
        
        self.scheduler.step()
        
        return {
            'emotion_loss': total_emotion_loss / len(dataloader),
            'speaker_loss': total_speaker_loss / len(dataloader),
            'emotion_acc': correct_emotion / total,
            'speaker_acc': correct_speaker / total
        }
    
    def validate(self, dataloader) -> dict:
        """Validate the model."""
        self.model.eval()
        total_emotion_loss = 0
        total_speaker_loss = 0
        correct_emotion = 0
        correct_speaker = 0
        total = 0
        
        with torch.no_grad():
            for waveforms, labels in dataloader:
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(waveforms)
                
                emotion_loss = self.emotion_criterion(outputs['emotion_logits'], labels)
                speaker_loss = self.speaker_criterion(outputs['speaker_logits'], labels)
                
                total_emotion_loss += emotion_loss.item()
                total_speaker_loss += speaker_loss.item()
                
                _, predicted_emotion = torch.max(outputs['emotion_logits'], 1)
                _, predicted_speaker = torch.max(outputs['speaker_logits'], 1)
                total += labels.size(0)
                correct_emotion += (predicted_emotion == labels).sum().item()
                correct_speaker += (predicted_speaker == labels).sum().item()
        
        return {
            'emotion_loss': total_emotion_loss / len(dataloader),
            'speaker_loss': total_speaker_loss / len(dataloader),
            'emotion_acc': correct_emotion / total,
            'speaker_acc': correct_speaker / total
        }
    
    def train(self, train_loader, val_loader, epochs: int = 50, save_path: str = None):
        """Full training loop."""
        logger.info(f"Starting training for {epochs} epochs")
        
        best_acc = 0
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train: E_loss={train_metrics['emotion_loss']:.3f} E_acc={train_metrics['emotion_acc']:.3f} | "
                f"Val: E_loss={val_metrics['emotion_loss']:.3f} E_acc={val_metrics['emotion_acc']:.3f}"
            )
            
            # Save best model
            if val_metrics['emotion_acc'] > best_acc:
                best_acc = val_metrics['emotion_acc']
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"Saved best model with acc={best_acc:.3f}")
        
        logger.info(f"Training complete. Best accuracy: {best_acc:.3f}")
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")


def record_voice_samples(output_dir: str, num_samples: int = 10, duration: int = 3):
    """Record voice samples for training."""
    os.makedirs(output_dir, exist_ok=True)
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    print(f"\n{'='*60}")
    print(f"Запись {num_samples} образцов голоса")
    print(f"{'='*60}\n")
    
    for i in range(num_samples):
        input(f"Нажмите Enter для записи образца #{i+1}...")
        
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print(f"Запись {duration} секунд...")
            audio = recognizer.record(source, duration=duration)
        
        # Save raw audio
        audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        filepath = os.path.join(output_dir, f"sample_{i+1}.npy")
        np.save(filepath, audio_float)
        print(f"Сохранено: {filepath}\n")
    
    print("Запись завершена!")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train voice analysis model")
    parser.add_argument('--mode', choices=['record', 'train'], default='train')
    parser.add_argument('--data-dir', type=str, default='voice_data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--save-path', type=str, default='voice_model.pth')
    parser.add_argument('--num-samples', type=int, default=10)
    
    args = parser.parse_args()
    
    if args.mode == 'record':
        record_voice_samples(args.data_dir, args.num_samples)
        return
    
    elif args.mode == 'train':
        # Check if data exists
        if not os.path.exists(args.data_dir):
            print(f"Директория {args.data_dir} не найдена!")
            print("Сначала запишите данные: python train_voice.py --mode record")
            return
        
        # Create dataset
        audio_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.npy')]
        if len(audio_files) == 0:
            print("Нет данных для обучения!")
            return
        
        # Dummy labels (in practice, you'd have proper labels)
        labels = [i % 8 for i in range(len(audio_files))]  # 8 emotions
        
        # Split train/val
        split_idx = int(len(audio_files) * 0.8)
        train_files = audio_files[:split_idx]
        val_files = audio_files[split_idx:]
        train_labels = labels[:split_idx]
        val_labels = labels[split_idx:]
        
        train_dataset = VoiceDataset(train_files, train_labels)
        val_dataset = VoiceDataset(val_files, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = VoiceClassifier(num_speakers=5, num_emotions=8)
        
        # Train
        trainer = VoiceTrainer(model, device)
        trainer.train(train_loader, val_loader, epochs=args.epochs, save_path=args.save_path)


if __name__ == "__main__":
    main()
