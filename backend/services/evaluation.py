"""
Analytics and Evaluation Module
- Emotion model accuracy evaluation
- RAG vs no-RAG hallucination comparison
- Performance metrics collection
"""
import time
import psutil
import os
from typing import List, Dict, Tuple

# Sample labeled dataset for emotion model evaluation (50 samples)
EVALUATION_DATASET = [
    ("I feel so sad and empty inside", "sadness"),
    ("Today was amazing! I got the internship!", "joy"),
    ("I'm terrified about my presentation tomorrow", "fear"),
    ("I'm furious at my roommate right now", "anger"),
    ("I don't know what to feel, just numb", "neutral"),
    ("I'm disgusted by what happened at the party", "disgust"),
    ("I can't believe they offered me the scholarship!", "surprise"),
    ("Everything feels hopeless and pointless", "sadness"),
    ("I'm so nervous about meeting new people", "fear"),
    ("I hate myself for failing that exam", "anger"),
    ("Life is beautiful and I'm grateful", "joy"),
    ("I feel worthless and nobody cares about me", "sadness"),
    ("I'm panicking, I can't breathe properly", "fear"),
    ("I'm really angry that my project was rejected", "anger"),
    ("Just another ordinary day", "neutral"),
    ("I feel disgusted at myself", "disgust"),
    ("Wow I totally didn't expect to win!", "surprise"),
    ("I've been crying all day and don't know why", "sadness"),
    ("I'm anxious about my future all the time", "fear"),
    ("I got an A on my thesis, I'm so proud!", "joy"),
    ("Nobody understands what I'm going through", "sadness"),
    ("I'm scared of going back to class after what happened", "fear"),
    ("I snapped at my best friend and regret it", "anger"),
    ("I feel okay today, just going through the motions", "neutral"),
    ("That behavior is absolutely repulsive to me", "disgust"),
    ("I can't stop worrying about everything", "fear"),
    ("My family visited and we had such a wonderful time", "joy"),
    ("I feel like a burden to everyone around me", "sadness"),
    ("I'm irritated by all the noise in the dorms", "anger"),
    ("I didn't expect my professor to give me an extension", "surprise"),
    ("I feel disconnected from everyone lately", "sadness"),
    ("The thought of the exam makes my heart race", "fear"),
    ("I was so frustrated I threw my textbook", "anger"),
    ("Had a productive study session, feeling good", "joy"),
    ("I feel indifferent about most things right now", "neutral"),
    ("That story about the animal cruelty was disgusting", "disgust"),
    ("I can't believe my parents are coming to visit!", "surprise"),
    ("I have no motivation to get out of bed", "sadness"),
    ("I'm constantly on edge and can't relax", "fear"),
    ("I feel so joyful when I help others", "joy"),
    ("I'm annoyed at the unfair grading", "anger"),
    ("Today was fine, nothing special", "neutral"),
    ("I feel sick thinking about what he said to me", "disgust"),
    ("I opened the acceptance letter and screamed!", "surprise"),
    ("The loneliness is unbearable some days", "sadness"),
    ("I dread social situations, they exhaust me", "fear"),
    ("I'm so excited for graduation!", "joy"),
    ("I'm frustrated with myself for procrastinating", "anger"),
    ("I feel neither good nor bad today", "neutral"),
    ("I'm genuinely shocked by the test results", "surprise"),
]

def evaluate_emotion_model() -> Dict:
    """Evaluate emotion detection model on labeled dataset"""
    from backend.services.emotion_service import detect_emotion
    
    print("Evaluating emotion detection model...")
    correct = 0
    total = len(EVALUATION_DATASET)
    predictions = []
    start = time.time()
    
    for text, true_label in EVALUATION_DATASET:
        predicted, confidence = detect_emotion(text)
        is_correct = predicted.lower() == true_label.lower()
        correct += int(is_correct)
        predictions.append({
            "text": text[:50],
            "true": true_label,
            "predicted": predicted,
            "confidence": confidence,
            "correct": is_correct
        })
    
    elapsed = time.time() - start
    accuracy = correct / total
    
    # Per-class metrics
    from collections import defaultdict
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    
    for p in predictions:
        if p["correct"]:
            class_tp[p["true"]] += 1
        else:
            class_fn[p["true"]] += 1
            class_fp[p["predicted"]] += 1
    
    emotions = set([d[1] for d in EVALUATION_DATASET])
    class_metrics = {}
    for em in emotions:
        tp = class_tp[em]
        fp = class_fp[em]
        fn = class_fn[em]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        class_metrics[em] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}
    
    macro_f1 = sum(v["f1"] for v in class_metrics.values()) / len(class_metrics)
    
    return {
        "total_samples": total,
        "correct": correct,
        "accuracy": round(accuracy, 3),
        "macro_f1": round(macro_f1, 3),
        "total_time_seconds": round(elapsed, 2),
        "avg_time_per_sample": round(elapsed / total, 3),
        "class_metrics": class_metrics
    }

def evaluate_rag_vs_no_rag(test_queries: List[str] = None) -> Dict:
    """Compare RAG responses vs non-RAG baseline"""
    if test_queries is None:
        test_queries = [
            "I'm feeling really stressed about my exams",
            "I've been feeling sad and disconnected lately",
            "I'm having panic attacks before presentations"
        ]
    
    from backend.services.rag_service import retrieve_context, generate_response
    
    results = []
    for query in test_queries:
        # Without RAG
        start = time.time()
        response_no_rag = generate_response(query, "sadness", context="")
        time_no_rag = time.time() - start
        
        # With RAG
        start = time.time()
        context = retrieve_context(query)
        response_rag = generate_response(query, "sadness", context=context)
        time_with_rag = time.time() - start
        
        results.append({
            "query": query,
            "response_no_rag": response_no_rag[:200],
            "response_with_rag": response_rag[:200],
            "context_retrieved": bool(context),
            "context_length": len(context),
            "time_no_rag": round(time_no_rag, 3),
            "time_with_rag": round(time_with_rag, 3)
        })
    
    return {"rag_comparison": results}

def collect_performance_metrics() -> Dict:
    """Collect system performance metrics"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent(interval=1)
    
    return {
        "memory_usage_mb": round(memory_mb, 1),
        "cpu_percent": round(cpu_percent, 1),
        "python_pid": os.getpid()
    }

def run_full_evaluation():
    """Run complete evaluation suite"""
    print("=" * 60)
    print("MINDBRIDGE PLATFORM EVALUATION")
    print("=" * 60)
    
    print("\n📊 Emotion Model Evaluation:")
    emotion_results = evaluate_emotion_model()
    print(f"  Accuracy: {emotion_results['accuracy']:.1%}")
    print(f"  Macro F1: {emotion_results['macro_f1']:.3f}")
    print(f"  Avg time/sample: {emotion_results['avg_time_per_sample']}s")
    
    print("\n🔍 RAG Pipeline Evaluation:")
    rag_results = evaluate_rag_vs_no_rag()
    for r in rag_results["rag_comparison"]:
        print(f"  Query: {r['query'][:50]}...")
        print(f"  Context retrieved: {r['context_retrieved']} ({r['context_length']} chars)")
    
    print("\n💻 System Performance:")
    perf = collect_performance_metrics()
    print(f"  Memory: {perf['memory_usage_mb']} MB")
    print(f"  CPU: {perf['cpu_percent']}%")
    
    return {
        "emotion_evaluation": emotion_results,
        "rag_evaluation": rag_results,
        "performance": perf
    }

if __name__ == "__main__":
    run_full_evaluation()
