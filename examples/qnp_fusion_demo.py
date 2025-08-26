#!/usr/bin/env python3
"""
Quantum-Neuromorphic-Photonic (QNP) Fusion Demo
Demonstrates the breakthrough QNP fusion system for sentiment analysis.

This demo showcases:
- Quantum coherence-based feature extraction
- Neuromorphic spike-based processing
- Photonic acceleration for ultra-fast inference
- Comprehensive performance benchmarking
"""

import asyncio
import json
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantum_neuromorphic_photonic_fusion import (
    create_qnp_fusion_engine,
    batch_analyze_sentiment,
    QNPFusionEngine
)

async def demo_basic_analysis():
    """Demonstrate basic QNP sentiment analysis"""
    print("🔬 Basic QNP Fusion Analysis Demo")
    print("=" * 50)
    
    engine = create_qnp_fusion_engine()
    
    test_texts = [
        "I absolutely love this amazing product!",
        "This is terrible and disappointing.",
        "The weather is okay today.",
        "Quantum computing combined with neuromorphic processing creates revolutionary AI capabilities!",
        "The photonic acceleration provides unprecedented speed and efficiency."
    ]
    
    for text in test_texts:
        print(f"\n📝 Text: {text}")
        result = await engine.analyze_sentiment_async(text)
        
        print(f"   Sentiment: {max(result, key=lambda k: result[k] if k in ['positive', 'negative', 'neutral'] else 0)}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   QNP Fusion Score: {result['qnp_fusion_score']:.3f}")
        print(f"   Processing Time: {result['processing_time_ms']:.2f}ms")

async def demo_batch_processing():
    """Demonstrate batch processing capabilities"""
    print("\n\n🚀 Batch Processing Demo")
    print("=" * 50)
    
    batch_texts = [
        "Excellent service and fast delivery!",
        "Poor quality, not worth the money.",
        "Average product, nothing special.",
        "Revolutionary breakthrough in AI technology!",
        "Disappointing results from the experiment.",
        "Neutral opinion about the new policy.",
        "Outstanding performance and reliability!",
        "Worst experience I've ever had.",
    ]
    
    print(f"Processing {len(batch_texts)} texts in parallel...")
    
    start_time = time.time()
    results = await batch_analyze_sentiment(batch_texts)
    batch_time = time.time() - start_time
    
    print(f"\n⏱️  Total batch processing time: {batch_time:.2f}s")
    print(f"📊 Average per-text processing time: {batch_time/len(batch_texts)*1000:.2f}ms")
    
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    total_fusion_score = 0
    
    for i, result in enumerate(results):
        if not result.get("fallback", False):
            dominant_sentiment = max(["positive", "negative", "neutral"], 
                                   key=lambda k: result[k])
            sentiment_counts[dominant_sentiment] += 1
            total_fusion_score += result["qnp_fusion_score"]
            
            print(f"Text {i+1}: {dominant_sentiment.upper()} (score: {result['qnp_fusion_score']:.3f})")
    
    print(f"\n📈 Batch Statistics:")
    print(f"   Positive: {sentiment_counts['positive']}")
    print(f"   Negative: {sentiment_counts['negative']}")
    print(f"   Neutral: {sentiment_counts['neutral']}")
    print(f"   Average QNP Fusion Score: {total_fusion_score/len(results):.3f}")

def demo_performance_comparison():
    """Compare QNP fusion vs traditional approaches"""
    print("\n\n⚡ Performance Comparison Demo")
    print("=" * 50)
    
    test_text = "This revolutionary quantum-photonic system delivers exceptional performance with unprecedented speed and accuracy!"
    
    # QNP Fusion Engine
    print("🔬 QNP Fusion Analysis:")
    qnp_engine = create_qnp_fusion_engine()
    
    start_time = time.time()
    qnp_result = qnp_engine.analyze_sentiment(test_text)
    qnp_time = time.time() - start_time
    
    print(f"   Processing Time: {qnp_time*1000:.2f}ms")
    print(f"   QNP Fusion Score: {qnp_result['qnp_fusion_score']:.3f}")
    print(f"   Quantum States: {qnp_result['quantum_states_processed']}")
    print(f"   Neuromorphic Spikes: {qnp_result['neuromorphic_spikes']}")
    print(f"   Confidence: {qnp_result['confidence']:.3f}")
    
    # Fallback comparison
    print("\n🔄 Classical Fallback Analysis:")
    fallback_result = qnp_engine.fallback_analysis(test_text)
    
    print(f"   Processing Time: {fallback_result['processing_time_ms']:.2f}ms")
    print(f"   Confidence: {fallback_result['confidence']:.3f}")
    print(f"   Method: Lexicon-based")
    
    print(f"\n📊 Performance Improvement:")
    if not qnp_result.get("fallback", False):
        confidence_improvement = (qnp_result['confidence'] - fallback_result['confidence']) / fallback_result['confidence'] * 100
        print(f"   Confidence Improvement: +{confidence_improvement:.1f}%")
        print(f"   Advanced Features: Quantum coherence, Neuromorphic processing, Photonic acceleration")
    else:
        print("   QNP system used fallback - performance equivalent")

async def demo_advanced_features():
    """Demonstrate advanced QNP features"""
    print("\n\n🧪 Advanced QNP Features Demo")
    print("=" * 50)
    
    # Custom configuration
    advanced_config = {
        "num_qubits": 12,
        "num_neurons": 128,
        "num_photonic_channels": 32,
        "processing_timeout": 2.0
    }
    
    print("🔧 Initializing advanced QNP configuration...")
    print(f"   Qubits: {advanced_config['num_qubits']}")
    print(f"   Neurons: {advanced_config['num_neurons']}")
    print(f"   Photonic Channels: {advanced_config['num_photonic_channels']}")
    
    engine = create_qnp_fusion_engine(advanced_config)
    
    complex_text = """
    The integration of quantum computing principles with neuromorphic architectures 
    represents a paradigm shift in artificial intelligence. By leveraging photonic 
    acceleration, we achieve unprecedented performance in natural language processing 
    tasks while maintaining quantum coherence and neuromorphic efficiency.
    """
    
    print(f"\n📝 Analyzing complex technical text...")
    result = await engine.analyze_sentiment_async(complex_text)
    
    print(f"   Sentiment Distribution:")
    print(f"     Positive: {result['positive']:.3f}")
    print(f"     Negative: {result['negative']:.3f}")
    print(f"     Neutral: {result['neutral']:.3f}")
    print(f"   QNP Metrics:")
    print(f"     Fusion Score: {result['qnp_fusion_score']:.3f}")
    print(f"     Processing Time: {result['processing_time_ms']:.2f}ms")
    print(f"     Quantum States: {result['quantum_states_processed']}")
    print(f"     Neuromorphic Spikes: {result['neuromorphic_spikes']}")
    
    # Performance report
    print(f"\n📈 System Performance Report:")
    report = engine.get_performance_report()
    print(json.dumps(report, indent=2))

def demo_research_benchmarks():
    """Demonstrate research-quality benchmarking"""
    print("\n\n📚 Research Benchmarks Demo")
    print("=" * 50)
    
    # Research-grade test suite
    research_texts = [
        # Technical/Scientific
        "The quantum entanglement properties exhibited remarkable coherence stability.",
        "Photonic interference patterns demonstrated optimal signal processing capabilities.",
        "Neuromorphic spike timing showed significant temporal precision improvements.",
        
        # Emotional Spectrum
        "I am absolutely thrilled with these groundbreaking results!",
        "This methodology shows promising potential for future applications.",
        "The experimental data reveals concerning inconsistencies.",
        
        # Complex Sentiment
        "While the initial results were disappointing, the refined approach yielded excellent outcomes.",
        "The system performance exceeded expectations despite challenging operational conditions.",
        "Mixed feelings about the implementation - great concept but execution needs improvement."
    ]
    
    print("🔬 Running research-quality benchmark suite...")
    
    engine = create_qnp_fusion_engine({
        "num_qubits": 16,
        "num_neurons": 256, 
        "num_photonic_channels": 64
    })
    
    benchmark_results = []
    
    for i, text in enumerate(research_texts, 1):
        print(f"\n📊 Test Case {i}:")
        print(f"   Text: {text[:60]}{'...' if len(text) > 60 else ''}")
        
        result = engine.analyze_sentiment(text)
        benchmark_results.append(result)
        
        print(f"   Sentiment: {max(['positive', 'negative', 'neutral'], key=lambda k: result[k])}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   QNP Score: {result['qnp_fusion_score']:.3f}")
        print(f"   Process Time: {result['processing_time_ms']:.2f}ms")
    
    # Statistical analysis
    avg_processing_time = sum(r['processing_time_ms'] for r in benchmark_results) / len(benchmark_results)
    avg_fusion_score = sum(r['qnp_fusion_score'] for r in benchmark_results) / len(benchmark_results)
    avg_confidence = sum(r['confidence'] for r in benchmark_results) / len(benchmark_results)
    
    print(f"\n📈 Benchmark Statistics:")
    print(f"   Average Processing Time: {avg_processing_time:.2f}ms")
    print(f"   Average QNP Fusion Score: {avg_fusion_score:.3f}")
    print(f"   Average Confidence: {avg_confidence:.3f}")
    print(f"   Throughput: {1000/avg_processing_time:.0f} texts/second")

async def main():
    """Main demo execution"""
    print("🌟 Quantum-Neuromorphic-Photonic Fusion Demo")
    print("=" * 60)
    print("🚀 Demonstrating breakthrough QNP sentiment analysis")
    print("🔬 Combining Quantum + Neuromorphic + Photonic processing")
    print("⚡ Ultra-high performance with novel AI architecture")
    print("=" * 60)
    
    try:
        # Run all demos
        await demo_basic_analysis()
        await demo_batch_processing()
        demo_performance_comparison()
        await demo_advanced_features()
        demo_research_benchmarks()
        
        print("\n\n✅ QNP Fusion Demo Complete!")
        print("🎯 Successfully demonstrated:")
        print("   ✓ Quantum coherence-based feature extraction")
        print("   ✓ Neuromorphic spike-based processing")
        print("   ✓ Photonic acceleration capabilities")
        print("   ✓ Batch processing performance")
        print("   ✓ Advanced configuration options")
        print("   ✓ Research-quality benchmarking")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())