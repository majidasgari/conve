#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
محاسبه Effective Performance: ConvE + BGE Pipeline
"""

# نتایج از coverage analysis
coverage_top10 = 0.2874  # 28.74% of answers are in top-10

# نتایج ConvE (baseline)
conve_hits_at_10 = 0.3087

# نتایج BGE Re-ranking (within top-10)
bge_rerank_hits_at_1 = 0.1008
bge_rerank_hits_at_3 = 0.2785
bge_rerank_hits_at_5 = 0.4723
bge_rerank_hits_at_10 = 0.9355

print("=" * 70)
print("Pipeline Performance: ConvE → BGE Re-ranking")
print("=" * 70)

print("\n1️⃣ ConvE Baseline (top-10 از همه entities):")
print(f"   Hits@10 = {conve_hits_at_10:.4f} ({conve_hits_at_10*100:.2f}%)")

print("\n2️⃣ Coverage Analysis:")
print(f"   چند درصد از answers در top-10 ConvE هستند؟")
print(f"   Coverage = {coverage_top10:.4f} ({coverage_top10*100:.2f}%)")

print("\n3️⃣ BGE Re-ranking Performance (فقط روی top-10):")
print(f"   از مواردی که در top-10 ConvE هستند:")
print(f"   - Hits@1 (within top-10) = {bge_rerank_hits_at_1:.4f}")
print(f"   - Hits@3 (within top-10) = {bge_rerank_hits_at_3:.4f}")
print(f"   - Hits@5 (within top-10) = {bge_rerank_hits_at_5:.4f}")
print(f"   - Hits@10 (within top-10) = {bge_rerank_hits_at_10:.4f}")

print("\n4️⃣ Effective Performance (کل pipeline):")
print(f"   این نشون میده که اگه ConvE + BGE رو با هم بزنی،")
print(f"   چند درصد از کل test set رو درست پیش‌بینی می‌کنی:")

effective_hits_1 = coverage_top10 * bge_rerank_hits_at_1
effective_hits_3 = coverage_top10 * bge_rerank_hits_at_3
effective_hits_5 = coverage_top10 * bge_rerank_hits_at_5
effective_hits_10 = coverage_top10 * bge_rerank_hits_at_10

print(f"\n   Effective Hits@1  = {coverage_top10:.4f} × {bge_rerank_hits_at_1:.4f} = {effective_hits_1:.4f} ({effective_hits_1*100:.2f}%)")
print(f"   Effective Hits@3  = {coverage_top10:.4f} × {bge_rerank_hits_at_3:.4f} = {effective_hits_3:.4f} ({effective_hits_3*100:.2f}%)")
print(f"   Effective Hits@5  = {coverage_top10:.4f} × {bge_rerank_hits_at_5:.4f} = {effective_hits_5:.4f} ({effective_hits_5*100:.2f}%)")
print(f"   Effective Hits@10 = {coverage_top10:.4f} × {bge_rerank_hits_at_10:.4f} = {effective_hits_10:.4f} ({effective_hits_10*100:.2f}%)")

print("\n" + "=" * 70)
print("نتیجه‌گیری:")
print("=" * 70)

if effective_hits_10 < conve_hits_at_10:
    diff = (conve_hits_at_10 - effective_hits_10) * 100
    print(f"❌ Pipeline کلی بدتر از ConvE تنها است!")
    print(f"   ConvE alone: {conve_hits_at_10*100:.2f}%")
    print(f"   ConvE + BGE:  {effective_hits_10*100:.2f}%")
    print(f"   افت: {diff:.2f}%")
    print(f"\n💡 دلیل: Coverage پایین است ({coverage_top10*100:.1f}%)")
    print(f"   باید k رو بزرگتر کنی (مثلاً k=100 یا k=500)")
else:
    improvement = (effective_hits_10 - conve_hits_at_10) * 100
    print(f"✅ Pipeline کلی بهتر از ConvE تنها است!")
    print(f"   بهبود: +{improvement:.2f}%")

print("\n" + "=" * 70)
print("توصیه:")
print("=" * 70)
print("برای k=10، coverage خیلی پایینه (29%).")
print("پیشنهاد:")
print("  • k=50  → coverage ≈ 38%")
print("  • k=100 → coverage ≈ 42%")
print("  • k=500 → coverage ≈ 51%")
print("=" * 70)
