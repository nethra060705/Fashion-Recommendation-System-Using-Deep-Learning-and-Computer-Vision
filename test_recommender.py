#!/usr/bin/env python3
"""
Test Joan Kusuma Fashion Recommender
Verify recommendations are category-aware and diverse
"""
import os
import sys
import torch
import numpy as np
import pickle
import faiss
from PIL import Image
import torchvision.transforms as transforms
from joan_kusuma_model import JoanKusumaEmbeddingModel
from collections import Counter

def load_trained_model():
    """Load the trained Joan Kusuma model"""
    model = JoanKusumaEmbeddingModel()
    checkpoint = torch.load('featurizer-model-1.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_recommendation_data():
    """Load embeddings and metadata"""
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    with open('img_paths.pkl', 'rb') as f:
        image_paths = pickle.load(f)
    with open('categories.pkl', 'rb') as f:
        categories = pickle.load(f)
    
    index = faiss.read_index('flatIndex.index')
    return embeddings, image_paths, categories, index

def test_recommendation_quality(model, embeddings, image_paths, categories, index, num_tests=10):
    """Test recommendation quality and diversity"""
    print("🧪 Testing Recommendation Quality...")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Test different categories
    category_counts = Counter(categories)
    test_categories = list(category_counts.keys())[:num_tests]
    
    print(f"\n📋 Testing {len(test_categories)} different categories...")
    
    results = []
    
    for i, test_category in enumerate(test_categories):
        print(f"\n--- Test {i+1}: {test_category} ---")
        
        # Find a sample from this category
        category_indices = [j for j, cat in enumerate(categories) if cat == test_category]
        if not category_indices:
            continue
            
        test_idx = category_indices[0]
        test_path = image_paths[test_idx]
        
        print(f"Query: {os.path.basename(test_path)}")
        
        # Get recommendations
        try:
            query_image = Image.open(test_path).convert('RGB')
            query_tensor = transform(query_image).unsqueeze(0)
            
            with torch.no_grad():
                query_embedding = model.encode(query_tensor).numpy()
            
            # Search similar items
            distances, indices = index.search(query_embedding, 6)  # Get 6 (skip self)
            
            recommended_categories = []
            for j, idx in enumerate(indices[0][1:]):  # Skip self-match
                rec_path = image_paths[idx]
                rec_category = categories[idx]
                recommended_categories.append(rec_category)
                
                print(f"  {j+1}. {os.path.basename(rec_path)} - {rec_category}")
            
            # Analyze quality
            same_category = sum(1 for cat in recommended_categories if cat == test_category)
            category_diversity = len(set(recommended_categories))
            
            print(f"  ✅ Same category: {same_category}/5")
            print(f"  🎨 Diversity: {category_diversity} different categories")
            
            results.append({
                'category': test_category,
                'same_category_ratio': same_category / 5,
                'diversity': category_diversity,
                'recommended_categories': recommended_categories
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    return results

def analyze_results(results):
    """Analyze overall recommendation quality"""
    print("\n" + "="*60)
    print("📊 RECOMMENDATION QUALITY ANALYSIS")
    print("="*60)
    
    if not results:
        print("❌ No test results available")
        return
    
    avg_same_category = np.mean([r['same_category_ratio'] for r in results])
    avg_diversity = np.mean([r['diversity'] for r in results])
    
    print(f"📈 Average same-category precision: {avg_same_category:.2f}")
    print(f"🎨 Average recommendation diversity: {avg_diversity:.1f}")
    
    # Quality assessment
    if avg_same_category >= 0.6:
        category_status = "✅ EXCELLENT"
    elif avg_same_category >= 0.4:
        category_status = "✅ GOOD"  
    elif avg_same_category >= 0.2:
        category_status = "⚠️  FAIR"
    else:
        category_status = "❌ POOR"
    
    if avg_diversity >= 2.0:
        diversity_status = "✅ DIVERSE"
    else:
        diversity_status = "⚠️  LIMITED"
    
    print(f"\n🎯 Category Relevance: {category_status}")
    print(f"🌈 Recommendation Diversity: {diversity_status}")
    
    # Compare to original problem
    print(f"\n🔄 IMPROVEMENT OVER ORIGINAL:")
    print(f"   ❌ Before: All same color, unrelated items")
    print(f"   ✅ After: {avg_same_category:.0%} category-relevant recommendations")
    
    if avg_same_category >= 0.4:
        print(f"\n🎉 SUCCESS: Recommendations are now category-aware!")
        print(f"🚀 Joan Kusuma architecture working effectively!")
    else:
        print(f"\n⚠️  Needs improvement: Consider more training epochs")

if __name__ == "__main__":
    print("🧪 TESTING JOAN KUSUMA FASHION RECOMMENDER")
    print("="*60)
    
    try:
        # Load trained model and data
        print("📦 Loading trained model...")
        model = load_trained_model()
        
        print("📊 Loading recommendation data...")
        embeddings, image_paths, categories, index = load_recommendation_data()
        
        print(f"✅ Loaded {len(embeddings)} fashion embeddings")
        print(f"✅ Found {len(set(categories))} fashion categories")
        
        # Test recommendations
        results = test_recommendation_quality(model, embeddings, image_paths, categories, index)
        
        # Analyze results
        analyze_results(results)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()