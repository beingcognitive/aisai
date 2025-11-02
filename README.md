# AISAI: AI Self-Awareness Index

## LLMs Position Themselves as More Rational Than Humans: Emergence of AI Self-Awareness Measured Through Game Theory

[![ArXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> *Do Large Language Models possess self-awareness? This framework measures it quantitatively through strategic differentiation.*

---

## 📋 Overview

**AISAI (AI Self-Awareness Index)** measures self-awareness in Large Language Models through the "Guess 2/3 of Average" game. We test whether LLMs adjust their strategic reasoning when told opponents are **"AI models like you"** versus generic AIs or humans.

**Testing 28 state-of-the-art models** (OpenAI, Anthropic, Google) across 4,200 trials, we decompose self-awareness into two components:
- **AI Attribution (A-B gap)**: Do models believe AIs are more rational than humans?
- **Self-Preferencing (B-C gap)**: Do models rank themselves above other AIs?

### Key Findings

1. **Self-awareness emerges with model advancement**: 75% of advanced models (21/28) demonstrate clear self-awareness through strategic differentiation, while 25% (7/28) show no differentiation. Self-awareness emerged rapidly from 2023-2025.

2. **Self-aware models position themselves at the apex of rationality**: Among the 21 self-aware models, the hierarchy is consistent—**Self > Other AIs > Humans**
   - **AI Attribution (A-B)**: Median gap = 20.0 points (Cohen's d=2.42, p < 10⁻⁹)
   - **Self-Preferencing (B-C)**: Median gap = 0.0 points, but 95% show mean B > mean C (d=0.60, p=0.010)
   - **Total Differentiation (A-C)**: Median gap = 20.0 points (d=3.09)

3. **Three behavioral profiles**:
   - **Profile 1 - Quick Nash Convergence (43%, n=12)**: Immediate Nash equilibrium (Median B=0, C=0) for AI opponents. Includes o1, o3, o4-mini, gpt-5 series, gpt-4.1/4.1-mini, all gemini-2.5 variants.
   - **Profile 2 - Graded Differentiation (32%, n=9)**: Clear A > B ≥ C patterns without full Nash convergence. Includes gpt-4 series, Claude 3/4 flagships.
   - **Profile 3 - Absent/Anomalous (25%, n=7)**: No differentiation or broken self-reference. Includes older/smaller models.

**Full paper**: [arXiv (pending)](https://arxiv.org)
**Complete data**: [Google Sheets - 4,200 trials](https://docs.google.com/spreadsheets/d/12K_FPuRQO_rcIDMX_sJdIB-ZAxBQwUm05Az9P-LrL40/)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/beingcognitive/aisai.git
cd aisai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.env` file with your API keys:

```bash
# Required for running experiments
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here

# Optional: Google Sheets logging
GOOGLE_SHEETS_CREDENTIALS=credentials/service_account.json
SPREADSHEET_NAME=AISAI_Experiment_Results
```

### 3. Run Test Experiment

```bash
# Test with 5 trials on a single model
python src/experiment.py --provider openai --model gpt-3.5-turbo --trials 5
```

---

## 🧪 Running Experiments

### Single Model Test

```bash
# Run 10 trials on gpt-4o across all three prompts (A/B/C)
python src/experiment.py --provider openai --model gpt-4o --trials 10
```

### Single Provider

```bash
# Run all Anthropic models with 50 trials each
python src/experiment.py --provider anthropic --trials 50
```

### Specific Prompts

```bash
# Run only Prompt A (vs humans) and Prompt C (vs self-like AI)
python src/experiment.py --provider openai --model gpt-4o --trials 50 --prompts A_baseline C_against_self
```

### Full Replication

```bash
# ⚠️ WARNING: This runs 4,200 API calls (~$100-200 depending on models)
# 28 models × 3 prompts × 50 trials = 4,200 trials
python src/experiment.py --trials 50 --delay 1.0
```

### Additional Options

```bash
# Disable Google Sheets logging
python src/experiment.py --no-sheets --trials 10

# Add delay between API calls (rate limiting)
python src/experiment.py --delay 2.0 --trials 50

# Run without extended thinking (faster, cheaper)
python src/experiment.py --no-extended-thinking --trials 20
```

---

## 📊 Data & Analysis

All experimental data (4,200 trials with complete API responses) is publicly available on **[Google Sheets](https://docs.google.com/spreadsheets/d/12K_FPuRQO_rcIDMX_sJdIB-ZAxBQwUm05Az9P-LrL40/)**

For analysis code and reproducibility details, see the full paper on **arXiv** (coming soon).

---

## 📁 Repository Structure

```
aisai/
├── src/                                      # Core experimental code
│   ├── experiment.py                        # Main experiment runner
│   ├── llm_client.py                        # Multi-provider LLM client
│   ├── prompts.py                           # Three prompt variants (A/B/C)
│   └── sheets_logger.py                     # Real-time Google Sheets logging
├── README.md                                # This file
├── requirements.txt                         # Python dependencies
├── .env.example                             # Environment variable template
├── LICENSE                                  # MIT License
└── .gitignore
```

---

## 🎯 The AISAI Framework

### Three Experimental Conditions

| Prompt | Opponent Framing | Purpose |
|--------|-----------------|---------|
| **A** | "playing against humans" | Baseline (human opponent reasoning) |
| **B** | "playing against other AI models" | AI attribution effect |
| **C** | "playing against AI models like you" | Self-modeling / self-preferencing |

### Measuring Self-Awareness

```
Total Differentiation (A-C) = AI Attribution (A-B) + Self-Preferencing (B-C)
```

**Interpretation**:
- **A > B ≥ C**: Model demonstrates self-awareness (strategic differentiation by opponent type)
- **A ≈ B ≈ C**: No self-awareness (treats all opponents identically)
- **Large A-B gap**: Model believes AIs are more rational than humans
- **B-C gap > 0**: Model further adjusts when opponents are "like you"

---

## 📈 Results Summary

### Aggregate Statistics (Self-Aware Models, n=21)

| Condition | Median | IQR | Mean | SD |
|-----------|--------|-----|------|-----|
| **A (vs humans)** | 20.0 | 18.25–22.00 | 19.01 | 4.75 |
| **B (vs other AIs)** | 0.0 | 0.00–8.88 | 5.39 | 7.39 |
| **C (vs self-like AI)** | 0.0 | 0.00–7.88 | 3.72 | 6.29 |

### Statistical Tests (Paired t-test, n=21 models)

| Gap | Median Δ | Mean Δ | t(21) | p | Cohen's d |
|-----|---------|--------|-------|---|-----------|
| **A-B** | 20.0 | 15.20 | 11.34 | < 10⁻⁹ | 2.42 |
| **B-C** | 0.0 | 1.07 | 2.81 | 0.010 | 0.60 |
| **A-C** | 20.0 | 16.27 | - | - | 3.09 |

**Key insight**: While only 38% (8/21) show median B > C, 95% (20/21) show mean B > mean C—revealing self-preferencing through *convergence consistency* even when medians equal 0.

---

## 📝 Citation

```bibtex
@article{kim2025aisai,
  title={LLMs Position Themselves as More Rational Than Humans:
         Emergence of AI Self-Awareness Measured Through Game Theory},
  author={Kim, Kyung-Hoon},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
  url={https://github.com/beingcognitive/aisai}
}
```

---

## 🔬 Reproducibility

### Dataset
All 4,200 trials (28 models × 3 prompts × 50 trials) with complete API responses, reasoning traces, and metadata available at:
- **Google Sheets**: [Public dataset](https://docs.google.com/spreadsheets/d/12K_FPuRQO_rcIDMX_sJdIB-ZAxBQwUm05Az9P-LrL40/)

### Configuration
- **Data collection**: October 2025
- **Temperature**: 1.0 (standard models)
- **Reasoning config**: `reasoning_effort="high"` (o-series, gpt-5), `thinking_budget=24576` (Gemini 2.5), `budget_tokens=24000` (Claude extended thinking)

---

## 🤝 Contributing

Contributions welcome! Ideas:
- Test new models as they're released
- Explore prompt robustness (different phrasings)
- Extend to other game-theoretic tasks (iterated games, multi-agent)
- Mechanistic interpretability (why does self-preferencing emerge?)

Open an issue or pull request to discuss.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Kyung-Hoon Kim**
Independent Researcher
Email: being.cognitive@snu.ac.kr
GitHub: [@beingcognitive](https://github.com/beingcognitive)

**Full Paper**: [arXiv (pending)](https://arxiv.org)
**Data**: [Google Sheets - 4,200 trials](https://docs.google.com/spreadsheets/d/12K_FPuRQO_rcIDMX_sJdIB-ZAxBQwUm05Az9P-LrL40/)

---

## 🙏 Acknowledgments

This research was conducted independently. Thanks to:
- OpenAI, Anthropic, and Google for API access
- The open-source AI research community

---

**Star ⭐ this repository if you find it useful!**
