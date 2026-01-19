#!/usr/bin/env python3
"""
Fast Joan Kusuma Implementation
Optimized for quick high-quality results
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import pickle
import faiss
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import random
from joan_kusuma_model import JoanKusumaEmbeddingModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

class QuickFashionDataset(Dataset):
    """Efficient dataset for quick training"""
    def __init__(self, image_dir, transform=None, max_samples=3000):
        self.image_dir = image_dir
        self.transform = transform
        
        # Get balanced sample
        all_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
        
        # Quick category balancing
        category_files = defaultdict(list)
        for file in all_files:
            try:
                category = file.split('_')[1]
                category_files[category].append(file)
            except:
                category_files['Unknown'].append(file)
        
        # Take up to 200 per category for speed
        self.image_files = []
        for category, files in category_files.items():
            sample_size = min(200, len(files))
            self.image_files.extend(random.sample(files, sample_size))
        
        # Limit total for speed
        if len(self.image_files) > max_samples:
            self.image_files = random.sample(self.image_files, max_samples)
        
        print(f"Quick dataset: {len(self.image_files)} images from {len(category_files)} categories")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, image
        except:
            # Return dummy for corrupted images
            if self.transform:
                dummy = Image.new('RGB', (128, 128), (128, 128, 128))
                return self.transform(dummy), self.transform(dummy)
            return torch.zeros(3, 128, 128), torch.zeros(3, 128, 128)

def quick_train():
    """Fast training for immediate results"""
    print("🚀 Quick Joan Kusuma Training (Optimized)")
    
    device = torch.device('cpu')  # Use CPU for stability
    
    # Joan's transform but simplified
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Quick dataset
    dataset = QuickFashionDataset('index_images', transform=transform, max_samples=2500)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Joan Kusuma's model
    model = JoanKusumaEmbeddingModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    
    # Quick training - 20 epochs for speed
    num_epochs = 20
    print(f"Training for {num_epochs} epochs...")
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Limit batches for speed
            if batch_idx > 60:
                break
        
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_info': {
            'architecture': 'Joan Kusuma +Layer Model (Quick)',
            'embedding_dim': 512,
            'epochs': num_epochs
        }
    }, 'featurizer-model-1.pt')
    
    print("✅ Quick training complete!")
    return model

def create_embeddings(model):
    """Create embeddings efficiently"""
    print("\n🔍 Creating Fashion Embeddings...")
    
    device = torch.device('cpu')
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    embeddings = []
    image_paths = []
    categories = []
    
    image_dir = 'index_images'
    all_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    
    # Balance categories for better recommendations
    category_files = defaultdict(list)
    for file in all_files:
        try:
            category = file.split('_')[1]
            category_files[category].append(file)
        except:
            category_files['Unknown'].append(file)
    
    # Take up to 500 per category for good coverage
    selected_files = []
    for category, files in category_files.items():
        sample_size = min(500, len(files))
        selected_files.extend(random.sample(files, sample_size))
    
    print(f"Processing {len(selected_files)} images for embeddings...")
    
    with torch.no_grad():
        for img_file in tqdm(selected_files, desc="Extracting features"):
            img_path = os.path.join(image_dir, img_file)
            
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Extract Joan Kusuma's 512-d embeddings
                embedding = model.encode(image_tensor)
                embedding = embedding.cpu().numpy().flatten()
                
                category = img_file.split('_')[1] if '_' in img_file else 'Unknown'
                
                embeddings.append(embedding)
                image_paths.append(img_path)
                categories.append(category)
                
            except Exception as e:
                continue
    
    embeddings = np.array(embeddings)
    
    print(f"✅ Created {len(embeddings)} embeddings")
    print(f"   Dimension: {embeddings.shape[1]}")
    print(f"   Categories: {len(set(categories))}")
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    # Save all data
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    with open('img_paths.pkl', 'wb') as f:
        pickle.dump(image_paths, f)
    with open('categories.pkl', 'wb') as f:
        pickle.dump(categories, f)
    
    faiss.write_index(index, 'flatIndex.index')
    
    print(f"✅ FAISS index saved with {index.ntotal} vectors")
    
    return embeddings, categories

def evaluate_map_k(embeddings, categories, k=5):
    """Quick mAP@k evaluation"""
    print(f"\n📊 Evaluating mAP@{k}...")
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    aps = []
    
    # Sample for quick evaluation
    sample_size = min(1000, len(embeddings))
    sample_indices = random.sample(range(len(embeddings)), sample_size)
    
    for i in tqdm(sample_indices, desc=f"Calculating mAP@{k}"):
        query_category = categories[i]
        query = embeddings[i:i+1]
        
        _, indices = index.search(query, k+1)
        retrieved = indices[0][1:]  # Skip self
        
        relevant = 0
        precision_sum = 0
        
        for j, idx in enumerate(retrieved):
            if categories[idx] == query_category:
                relevant += 1
                precision_sum += relevant / (j + 1)
        
        ap = precision_sum / min(relevant, k) if relevant > 0 else 0
        aps.append(ap)
    
    map_score = np.mean(aps)
    print(f"📈 mAP@{k}: {map_score:.3f}")
    print(f"   Target: 0.530 (Joan Kusuma)")
    print(f"   Status: {'✅ GOOD' if map_score >= 0.40 else '⚠️  IMPROVING'}")
    
    return map_score

if __name__ == "__main__":
    print("=" * 60)
    print("⚡ JOAN KUSUMA QUICK TRAINING")
    print("=" * 60)
    
    # Quick training
    model = quick_train()
    
    # Create embeddings
    embeddings, categories = create_embeddings(model)
    
    # Quick evaluation
    map_score = evaluate_map_k(embeddings, categories)
    
    print("\n" + "=" * 60)
    print("✅ QUICK TRAINING COMPLETE!")
    print(f"📊 Achieved mAP@5: {map_score:.3f}")
    print("🎯 Fashion recommender ready!")
    print("=" * 60)