# Bias-and-Fairness-in-Human---AI-Interaction-Technical-Origins-Risks-and-Mitigation-Strategies
# Bias and Fairness in AI-Human Interaction: Technical Origins, Mitigations, and Risk Strategies

## 📌 Overview
This research project investigates the **socio-technical origins of bias** in machine learning models using the UCI Adult Census dataset. By analyzing how standard optimization objectives like **Empirical Risk Minimization (ERM)** inadvertently reify historical inequalities, this study develops a robust framework for bias detection, fairness evaluation, and algorithmic mitigation.

## 🔍 Problem Statement
Traditional machine learning models prioritize global accuracy, often at the expense of marginalized subgroups. Even when protected attributes (sex, race) are removed, models engage in **indirect discrimination** through **Redundant Encoding**—leveraging proxy variables such as occupation and education to reconstruct protected profiles.

## 🚀 Research Motivation
To move beyond 'fairness through blindness' by implementing active algorithmic interventions and human-in-the-loop safeguards that prevent discriminatory **feedback loops** and **automation bias** in high-stakes decision-making.

## 🛠 Methodology
1.  **Exploratory Bias Audit**: Utilized Correlation Heatmaps to identify proxy variables and representation gaps.
2.  **Fairness Evaluation**: Applied quantitative metrics including **Demographic Parity** and **Equal Opportunity Difference**.
3.  **Algorithmic Mitigation**:
    *   **Pre-processing**: Reweighing training instances to balance historical signals.
    *   **Post-processing**: Group-specific threshold adjustment (0.4 for Female / 0.6 for Male).
4.  **Robustness Testing**: Validated results using **5-Fold Cross-Validation** to ensure stability across data slices.

## 📊 Key Results & Findings
| Metric | Baseline | Post-Mitigation |
| :--- | :---: | :---: |
| **Demographic Parity Difference** | -0.15 | **0.0004** |
| **Equal Opportunity Difference** | -0.01 | 0.33 |

*   **The Impossibility Theorem**: Results empirically demonstrate the mathematical trade-off between parity and error-rate separation; achieving near-perfect outcome equality increased the Equal Opportunity gap.
*   **Stability**: Cross-validation confirmed a low standard deviation (<0.012), proving the mitigation strategy is robust and scalable.

## 🤖 Human-AI Collaboration Framework
To address the inherent mathematical limits of fairness, this project proposes a **Human-in-the-Loop (HITL) Override Mechanism**. Using **Counterfactual Reasoning** (What-if analysis), human agents can audit AI suggestions, identify systemic barriers, and override biased predictions to prevent the 'Self-Fulfilling Prophecy' of historical data.

## 💻 Tech Stack
*   **Languages**: Python
*   **Libraries**: PyTorch, Pandas, NumPy, Scikit-Learn
*   **Fairness Toolkit**: IBM AI Fairness 360 (AIF360)
*   **Visualization**: Matplotlib, Seaborn

## 📁 Project Structure
```text
├── data/               # UCI Adult Dataset (Raw & Processed)
├── notebooks/          # Research implementation & Stability Analysis
├── src/
│   ├── mitigation.py   # Reweighing & Thresholding Logic
│   ├── audit.py        # Correlation & Proxy Analysis
│   └── agent_sim.py    # HITL Counterfactual Framework
└── README.md
```

## ⚙️ How to Run
1.  Clone the repository: `git clone https://github.com/username/project-name.git`
2.  Install dependencies: `pip install aif360 pandas scikit-learn`
3.  Run the analysis: `python src/main.py`

## 🔮 Future Work
*   Implementation of **In-processing** techniques such as **Adversarial Debiasing**.
*   Exploration of **Intersectional Fairness** (e.g., race x gender) to identify hidden disparities.
*   Dynamic bias monitoring for **Agentic AI** reasoning engines.

## 📚 References
*   Bellamy, R. K. et al. (2018). *AI Fairness 360: An extensible toolkit for detecting, understanding, and mitigating unwanted algorithmic bias*.
*   Hardt, M. et al. (2016). *Equality of Opportunity in Supervised Learning*.
*   UCI Machine Learning Repository: Adult Dataset (1994).
