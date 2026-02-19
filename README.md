The Cost of Convenience: Modeling Consumer Complicity in E-Commerce
1. Project Overview
In the modern digital landscape, the friction between "wanting" and "buying" has vanished. This project explores the data behind our relationship with material goods. Beyond standard sales metrics, this study investigates the psychology of the digital shopper and how e-commerce systems predict our "Breaking Point"—the moment a user reaches total fatigue and exits the ecosystem.
This project serves as the data-driven foundation for an awareness campaign on digital consumption habits and systemic complicity.

2. Research Objectives
Behavioral Archetypes: Identify distinct shopper personas based on habits of browsing, regret, and hesitation.
Quantifying Regret: Determine if return rates and financial remorse act as leading indicators for platform disengagement.
Total Predictability: Measure how accurately advanced algorithms (Random Forest, XGBoost, Neural Networks) can forecast a user's departure.

3. Behavioral Feature Engineering
To move beyond raw transaction logs, two custom psychological metrics were engineered:
Window Shopping Ratio: (Browsing Frequency / Total Orders). Measures the "dopamine hit" of digital window shopping without the intent to buy.
Regret Intensity: (Return Rate × Average Order Value). Quantifies the financial and emotional weight of buyer’s remorse.

4. Consumer Archetypes (Unsupervised Learning)
Using K-Means Clustering, the project segmented users into four distinct behavioral profiles:
Archetype 0: The Disengaged (The Breaking Point)
Churn Rate: 55.2% (Highest)
Profile: Characterized by high fatigue and moderate browsing. They represent users who have lost interest in the platform's value proposition.
Archetype 1: The Rational Minimalist
Churn Rate: 0.6% (Lowest)
Profile: Low browsing and high satisfaction. These users use the platform as a utility rather than entertainment.
Archetype 2: The Regretful High-Spender
Churn Rate: 16.9%
Profile: Highest return rates and highest financial regret. Trapped in a "Buy-Return" cycle.
Archetype 3: The Dopamine Scroller
Churn Rate: 1.5%
Profile: Highest window-shopping ratio. They use the platform primarily for browsing and entertainment, remaining loyal despite high cart abandonment.

5. Predictive Performance (Supervised Learning)
The project compared three different architectures to predict the "Breaking Point" (Churn). All models demonstrated that consumer departure is a highly predictable event.
## Model Performance Comparison

| Model                         | Accuracy | Precision (Churn) | Recall (Churn) |
|--------------------------------|----------|------------------|----------------|
| Random Forest (Optimized)     | 97.0%    | 0.92             | 0.87           |
| XGBoost (Optimized)           | 97.0%    | 0.92             | 0.87           |
| Sequential Neural Network     | 96.0%    | 0.89             | 0.87           |

Key Finding: Feature importance analysis revealed that engagement_score and days_since_last_purchase are the primary drivers of disengagement. The system identifies a "fade-out" in activity long before the user actually decides to leave.

6. Technical Stack
Data Manipulation: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-Learn (K-Means, Random Forest, GridSearchCV)
Advanced Boosting: XGBoost
Deep Learning: TensorFlow / Keras (Sequential Neural Network)

7. Ethical Reflection: The Architecture of Complicity
If a machine can predict our "breaking point" with 97% accuracy, it raises fundamental questions about free will in the digital marketplace. This project proves that e-commerce platforms are not passive marketplaces; they are active monitoring systems. They track our hesitation, our addiction, and our remorse to calculate our exact value and our predicted exit.
Our "Digital Twin" is transparent to the system. The question remains: is the system designed to help us buy what we need, or to keep us in a behavioral loop until we are exhausted?

8. How to Use
Clone this repository.
Install requirements: pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow


Open ecommerce_notebook.ipynb in Jupyter Notebook or Google Colab.
Author: Soner Yigitdol
Focus: Data Science / Behavioral Analytics / Consumer Psychology
