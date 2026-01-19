#!/usr/bin/env python3
"""
Fashion Recommender Evaluation Metrics
Comprehensive evaluation suite for Joan Kusuma +Layer CNN model
"""
import os
import numpy as np
import torch
import pickle
import faiss
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter, defaultdict
from tqdm import tqdm
from joan_kusuma_model import JoanKusumaEmbeddingModel
import warnings
warnings.filterwarnings('ignore')

# Optional imports for visualization (install with: pip install matplotlib seaborn scikit-learn)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

class FashionRecommenderEvaluator:
    """Comprehensive evaluation metrics for fashion recommendation system"""
    
    def __init__(self, model_path='featurizer-model-1.pt'):
        """Initialize evaluator with trained model"""
        self.model = self.load_model(model_path)
        self.embeddings, self.image_paths, self.categories, self.index = self.load_data()
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"✅ Evaluator initialized with {len(self.embeddings)} fashion items")
        print(f"📊 Categories: {len(set(self.categories))} types")
    
    def load_model(self, model_path):
        """Load trained Joan Kusuma model"""
        model = JoanKusumaEmbeddingModel()
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def load_data(self):
        """Load embeddings and metadata"""
        with open('embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        with open('img_paths.pkl', 'rb') as f:
            image_paths = pickle.load(f)
        with open('categories.pkl', 'rb') as f:
            categories = pickle.load(f)
        
        index = faiss.read_index('flatIndex.index')
        return embeddings, image_paths, categories, index
    
    def calculate_map_at_k(self, k_values=[1, 3, 5, 10], sample_size=1000):
        """
        Calculate Mean Average Precision at K (mAP@K)
        Joan Kusuma's target: mAP@5 = 0.53
        """
        print(f"\n📊 Calculating mAP@K for k={k_values}...")
        
        results = {}
        
        # Sample queries for evaluation
        query_indices = np.random.choice(len(self.embeddings), 
                                       min(sample_size, len(self.embeddings)), 
                                       replace=False)
        
        for k in k_values:
            aps = []
            
            for query_idx in tqdm(query_indices, desc=f"mAP@{k}", leave=False):
                query_category = self.categories[query_idx]
                query_embedding = self.embeddings[query_idx:query_idx+1]
                
                # Get k+1 nearest neighbors (including self)
                distances, indices = self.index.search(query_embedding.astype('float32'), k+1)
                retrieved_indices = indices[0][1:]  # Exclude self
                
                # Calculate AP for this query
                relevant_count = 0
                precision_sum = 0.0
                
                for i, retrieved_idx in enumerate(retrieved_indices):
                    if self.categories[retrieved_idx] == query_category:
                        relevant_count += 1
                        precision_at_i = relevant_count / (i + 1)
                        precision_sum += precision_at_i
                
                # Average Precision for this query
                total_relevant = sum(1 for cat in self.categories if cat == query_category) - 1  # Exclude self
                ap = precision_sum / min(relevant_count, k) if relevant_count > 0 else 0.0
                aps.append(ap)
            
            results[f'mAP@{k}'] = np.mean(aps)
        
        return results
    
    def calculate_precision_recall_at_k(self, k_values=[1, 3, 5, 10], sample_size=1000):
        """Calculate Precision and Recall at K"""
        print(f"\n🎯 Calculating Precision/Recall@K...")
        
        results = {}
        query_indices = np.random.choice(len(self.embeddings), 
                                       min(sample_size, len(self.embeddings)), 
                                       replace=False)
        
        for k in k_values:
            precisions = []
            recalls = []
            
            for query_idx in tqdm(query_indices, desc=f"P/R@{k}", leave=False):
                query_category = self.categories[query_idx]
                query_embedding = self.embeddings[query_idx:query_idx+1]
                
                # Get k+1 nearest neighbors
                distances, indices = self.index.search(query_embedding.astype('float32'), k+1)
                retrieved_indices = indices[0][1:]  # Exclude self
                
                # Count relevant items in top-k
                relevant_in_topk = sum(1 for idx in retrieved_indices 
                                     if self.categories[idx] == query_category)
                
                # Total relevant items in dataset (excluding self)
                total_relevant = sum(1 for cat in self.categories if cat == query_category) - 1
                
                # Calculate metrics
                precision = relevant_in_topk / k if k > 0 else 0
                recall = relevant_in_topk / total_relevant if total_relevant > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
            
            results[f'Precision@{k}'] = np.mean(precisions)
            results[f'Recall@{k}'] = np.mean(recalls)
            results[f'F1@{k}'] = 2 * results[f'Precision@{k}'] * results[f'Recall@{k}'] / \
                                 (results[f'Precision@{k}'] + results[f'Recall@{k}']) \
                                 if (results[f'Precision@{k}'] + results[f'Recall@{k}']) > 0 else 0
        
        return results
    
    def calculate_category_performance(self, k=5, sample_size=100):
        """Calculate performance metrics per category"""
        print(f"\n🏷️  Calculating per-category performance...")
        
        category_stats = defaultdict(list)
        categories_list = list(set(self.categories))
        
        for category in tqdm(categories_list, desc="Categories"):
            # Get indices for this category
            category_indices = [i for i, cat in enumerate(self.categories) if cat == category]
            
            if len(category_indices) < 2:  # Need at least 2 items for evaluation
                continue
            
            # Sample queries from this category
            sample_indices = np.random.choice(category_indices, 
                                            min(sample_size, len(category_indices)), 
                                            replace=False)
            
            for query_idx in sample_indices:
                query_embedding = self.embeddings[query_idx:query_idx+1]
                
                # Get recommendations
                distances, indices = self.index.search(query_embedding.astype('float32'), k+1)
                retrieved_indices = indices[0][1:]  # Exclude self
                
                # Count relevant recommendations
                relevant_count = sum(1 for idx in retrieved_indices 
                                   if self.categories[idx] == category)
                
                precision = relevant_count / k
                category_stats[category].append(precision)
        
        # Calculate average performance per category
        category_performance = {}
        for category, precisions in category_stats.items():
            if precisions:
                category_performance[category] = {
                    'precision': np.mean(precisions),
                    'std': np.std(precisions),
                    'count': len(precisions)
                }
        
        return category_performance
    
    def calculate_diversity_metrics(self, k=5, sample_size=1000):
        """Calculate recommendation diversity metrics"""
        print(f"\n🌈 Calculating diversity metrics...")
        
        query_indices = np.random.choice(len(self.embeddings), 
                                       min(sample_size, len(self.embeddings)), 
                                       replace=False)
        
        intra_list_diversity = []  # Diversity within recommendation lists
        category_coverage = []     # Number of different categories in recommendations
        
        for query_idx in tqdm(query_indices, desc="Diversity", leave=False):
            query_embedding = self.embeddings[query_idx:query_idx+1]
            
            # Get recommendations
            distances, indices = self.index.search(query_embedding.astype('float32'), k+1)
            retrieved_indices = indices[0][1:]  # Exclude self
            
            # Calculate category diversity
            retrieved_categories = [self.categories[idx] for idx in retrieved_indices]
            unique_categories = len(set(retrieved_categories))
            category_coverage.append(unique_categories)
            
            # Calculate embedding diversity (average pairwise distance)
            if len(retrieved_indices) > 1:
                retrieved_embeddings = self.embeddings[retrieved_indices]
                pairwise_distances = []
                
                for i in range(len(retrieved_embeddings)):
                    for j in range(i+1, len(retrieved_embeddings)):
                        dist = np.linalg.norm(retrieved_embeddings[i] - retrieved_embeddings[j])
                        pairwise_distances.append(dist)
                
                avg_diversity = np.mean(pairwise_distances) if pairwise_distances else 0
                intra_list_diversity.append(avg_diversity)
        
        return {
            'avg_category_diversity': np.mean(category_coverage),
            'std_category_diversity': np.std(category_coverage),
            'avg_embedding_diversity': np.mean(intra_list_diversity),
            'std_embedding_diversity': np.std(intra_list_diversity)
        }
    
    def calculate_embedding_quality_metrics(self):
        """Evaluate embedding quality"""
        print(f"\n🔍 Analyzing embedding quality...")
        
        # Calculate inter/intra category distances
        categories_list = list(set(self.categories))
        intra_category_distances = []
        inter_category_distances = []
        
        for category in tqdm(categories_list, desc="Embedding analysis"):
            category_indices = [i for i, cat in enumerate(self.categories) if cat == category]
            
            if len(category_indices) < 2:
                continue
            
            # Intra-category distances
            category_embeddings = self.embeddings[category_indices]
            for i in range(len(category_embeddings)):
                for j in range(i+1, len(category_embeddings)):
                    dist = np.linalg.norm(category_embeddings[i] - category_embeddings[j])
                    intra_category_distances.append(dist)
            
            # Inter-category distances (sample other categories)
            other_indices = [i for i, cat in enumerate(self.categories) if cat != category]
            if other_indices:
                sample_other = np.random.choice(other_indices, 
                                              min(100, len(other_indices)), 
                                              replace=False)
                
                for cat_idx in category_indices[:10]:  # Sample from current category
                    for other_idx in sample_other[:10]:  # Sample from other categories
                        dist = np.linalg.norm(self.embeddings[cat_idx] - self.embeddings[other_idx])
                        inter_category_distances.append(dist)
        
        return {
            'avg_intra_category_distance': np.mean(intra_category_distances),
            'std_intra_category_distance': np.std(intra_category_distances),
            'avg_inter_category_distance': np.mean(inter_category_distances),
            'std_inter_category_distance': np.std(inter_category_distances),
            'separation_ratio': np.mean(inter_category_distances) / np.mean(intra_category_distances)
        }
    
    def generate_evaluation_report(self, save_path='evaluation_report.txt'):
        """Generate comprehensive evaluation report"""
        print(f"\n📝 Generating comprehensive evaluation report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FASHION RECOMMENDER EVALUATION REPORT")
        report_lines.append("Joan Kusuma +Layer CNN Architecture")
        report_lines.append("=" * 80)
        
        # 1. mAP@K metrics
        map_results = self.calculate_map_at_k()
        report_lines.append("\n🎯 MEAN AVERAGE PRECISION (mAP@K)")
        report_lines.append("-" * 40)
        for metric, value in map_results.items():
            status = "✅ EXCELLENT" if value >= 0.5 else "✅ GOOD" if value >= 0.3 else "⚠️  FAIR"
            report_lines.append(f"{metric:>12}: {value:.4f} {status}")
        
        # Joan Kusuma comparison
        joan_target = 0.53
        current_map5 = map_results.get('mAP@5', 0)
        comparison = f"({current_map5/joan_target:.1%} of Joan's target)" if joan_target > 0 else ""
        report_lines.append(f"{'Joan Target':>12}: {joan_target:.4f} (mAP@5)")
        report_lines.append(f"{'Performance':>12}: {comparison}")
        
        # 2. Precision/Recall metrics
        pr_results = self.calculate_precision_recall_at_k()
        report_lines.append("\n📊 PRECISION & RECALL METRICS")
        report_lines.append("-" * 40)
        for k in [1, 3, 5, 10]:
            if f'Precision@{k}' in pr_results:
                p = pr_results[f'Precision@{k}']
                r = pr_results[f'Recall@{k}']
                f1 = pr_results[f'F1@{k}']
                report_lines.append(f"k={k:>2} | P:{p:.3f} R:{r:.3f} F1:{f1:.3f}")
        
        # 3. Category performance
        category_perf = self.calculate_category_performance()
        report_lines.append("\n🏷️  CATEGORY PERFORMANCE (Precision@5)")
        report_lines.append("-" * 40)
        
        # Sort categories by performance
        sorted_categories = sorted(category_perf.items(), 
                                 key=lambda x: x[1]['precision'], 
                                 reverse=True)
        
        for category, stats in sorted_categories[:10]:  # Top 10
            precision = stats['precision']
            count = stats['count']
            status = "🥇" if precision >= 0.7 else "🥈" if precision >= 0.5 else "🥉" if precision >= 0.3 else "📊"
            report_lines.append(f"{status} {category:<20}: {precision:.3f} ({count} samples)")
        
        # 4. Diversity metrics
        diversity = self.calculate_diversity_metrics()
        report_lines.append("\n🌈 DIVERSITY ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append(f"Avg Categories per Rec: {diversity['avg_category_diversity']:.2f}")
        report_lines.append(f"Embedding Diversity:    {diversity['avg_embedding_diversity']:.4f}")
        
        # 5. Embedding quality
        embedding_quality = self.calculate_embedding_quality_metrics()
        report_lines.append("\n🔍 EMBEDDING QUALITY")
        report_lines.append("-" * 40)
        report_lines.append(f"Intra-category dist:  {embedding_quality['avg_intra_category_distance']:.4f}")
        report_lines.append(f"Inter-category dist:  {embedding_quality['avg_inter_category_distance']:.4f}")
        report_lines.append(f"Separation ratio:     {embedding_quality['separation_ratio']:.4f}")
        
        # 6. System info
        report_lines.append("\n⚙️  SYSTEM INFORMATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Total items:          {len(self.embeddings)}")
        report_lines.append(f"Categories:           {len(set(self.categories))}")
        report_lines.append(f"Embedding dimension:  {self.embeddings.shape[1]}")
        
        # 7. Overall assessment
        report_lines.append("\n🎯 OVERALL ASSESSMENT")
        report_lines.append("-" * 40)
        
        overall_score = (current_map5 + pr_results.get('Precision@5', 0) + 
                        diversity['avg_category_diversity']/5) / 3
        
        if overall_score >= 0.6:
            assessment = "🏆 EXCELLENT - Production ready!"
        elif overall_score >= 0.4:
            assessment = "✅ GOOD - Meeting requirements"
        elif overall_score >= 0.25:
            assessment = "⚠️  FAIR - Needs improvement"
        else:
            assessment = "❌ POOR - Requires retraining"
        
        report_lines.append(f"Overall Score:        {overall_score:.3f}")
        report_lines.append(f"Status:               {assessment}")
        
        report_lines.append("\n" + "=" * 80)
        
        # Save report
        report_text = "\n".join(report_lines)
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Report saved to {save_path}")
        print(report_text)
        
        return report_text
    
    def plot_performance_charts(self, save_dir='evaluation_plots'):
        """Generate visualization plots"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
            return
            
        print(f"\n📈 Generating performance charts...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. mAP@K plot
        map_results = self.calculate_map_at_k([1, 2, 3, 4, 5, 10, 20])
        
        plt.figure(figsize=(10, 6))
        k_values = [int(k.split('@')[1]) for k in map_results.keys()]
        map_values = list(map_results.values())
        
        plt.plot(k_values, map_values, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=0.53, color='r', linestyle='--', label='Joan Kusuma Target (0.53)')
        plt.xlabel('K (Number of Recommendations)')
        plt.ylabel('Mean Average Precision (mAP)')
        plt.title('mAP@K Performance - Joan Kusuma Fashion Recommender')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f'{save_dir}/map_at_k.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Category performance heatmap
        category_perf = self.calculate_category_performance()
        
        if category_perf:
            categories = list(category_perf.keys())
            precisions = [category_perf[cat]['precision'] for cat in categories]
            
            plt.figure(figsize=(12, 8))
            
            # Create color map
            colors = ['red' if p < 0.3 else 'orange' if p < 0.5 else 'green' for p in precisions]
            
            bars = plt.barh(categories, precisions, color=colors, alpha=0.7)
            plt.xlabel('Precision@5')
            plt.title('Category Performance - Fashion Recommendation')
            plt.axvline(x=0.5, color='blue', linestyle='--', alpha=0.5, label='Good Threshold')
            
            # Add value labels
            for i, (bar, precision) in enumerate(zip(bars, precisions)):
                plt.text(precision + 0.01, i, f'{precision:.3f}', 
                        va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/category_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Charts saved to {save_dir}/")

def main():
    """Run complete evaluation suite"""
    print("🧪 FASHION RECOMMENDER EVALUATION SUITE")
    print("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = FashionRecommenderEvaluator()
        
        # Generate comprehensive report
        evaluator.generate_evaluation_report()
        
        # Generate visualization plots
        evaluator.plot_performance_charts()
        
        print("\n🎉 Evaluation complete!")
        print("📁 Check evaluation_report.txt for detailed results")
        print("📊 Check evaluation_plots/ for visualizations")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()