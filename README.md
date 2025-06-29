
# 🧠 Evaluating the Comprehension of Technical Illustrations by Visual Language Models

This project introduces **TICQA** (Technical Illustration Comprehension Question Answering), a novel dataset designed to evaluate the ability of Vision-Language Models (VLMs) to understand **technical illustrations**. The study also benchmarks nine VLMs across three visual reasoning tasks and provides a detailed empirical analysis.

---

## 📌 Motivation

Technical diagrams—commonly found in textbooks, industrial manuals, and user guides—are essential forms of visual communication. Despite their importance, **no dedicated benchmark exists** to evaluate VLM performance on such illustrations. This work addresses that gap by:

- Creating a focused dataset on technical illustrations,
- Evaluating current VLM capabilities,
- Proposing a reproducible dataset generation methodology.

---

## 📦 Contributions

1. **🗂 TICQA Dataset**  
   A curated set of **250 samples**, divided into three task-specific subsets:
   - Visual Sequence Interpretation (assembly/disassembly)
   - Tools and Components Identification
   - Safety Warning Recognition

2. **🔄 Semi-Automatic Dataset Generation Pipeline**  
   The paper outlines a clear method for replicating and expanding dataset creation.

3. **📊 Empirical Evaluation of Nine VLMs**  
   Comprehensive benchmarking of medium-sized open-source VLMs with task-specific accuracy metrics and insights.

---

## ❓ Research Questions

1. How accurately can state-of-the-art VLMs infer **assembly/disassembly sequences** from diagrams when text labels are redacted?

2. Can VLMs correctly **identify tools/components** based solely on visual shape cues?

3. How effectively do VLMs **recognize and describe safety warnings** in open-ended formats?

---

## 🧪 Results

- **Assembly MCQ Performance**: In many cases, VLMs matched or **surpassed human performance**.
- **Open-Ended Tasks**: VLMs struggled with **safety descriptions** and **component naming**.
- **Contextual Inputs**: On average, **+4.5% accuracy gain** with context—but task-dependent effects observed.

---

## 🔮 Future Work

- Expand the dataset **beyond 250 samples** to cover broader tasks and styles.
- Include **larger, state-of-the-art VLMs** for deeper comparative analysis.
- Conduct **in-depth behavior studies** to uncover training blind spots in current models.

---

## 🧬 Dataset Access

The TICQA dataset is publicly available on HuggingFace:  
👉 [TICQA on Hugging Face](https://huggingface.co/datasets/SarangChouguley/TICQA)

---

## 📄 Project Structure

```
├── scripts/ # Scripts to run the evaluation
├── result/ # Evaluation results
├── report/ # Final report and appendix
└── README.md
```
---

## 📄 Citation

```
@misc{ticqa2025,
author = {Sarang Ravi Chouguley},
title = {Evaluating the Comprehension of Technical Illustrations by Visual Language Models},
year = {2025},
howpublished = {\url{https://github.com/SarangChougs/comprehension-of-technical-illustrations-by-VLMs}}
}
```

---

## 📬 Contact

For questions, feedback, or collaboration:  
📧 sarangchouguley284@gmail.com