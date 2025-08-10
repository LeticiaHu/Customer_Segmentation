## Unsupervised Learning AI Agent: Customer Segmentation on Online Retail
A Streamlit + Python project that uses unsupervised learning (K-Means and hierarchical clustering) to segment customers from an online retail dataset and surface persona insights, cluster comparisons, and market basket patterns for cross-sell ideas.

## Why it matters
🎯 Personalize marketing to distinct segments

📦 Improve product recommendations and bundles

💰 Boost retention & sales efficiency with targeted outreach

## Business Recommendations based on model findings

- Cluster 0 — High-Value (Lapsed)
Goal: Re-engage high spenders who haven’t purchased recently.
Actions: Win-back offers, personalized recommendations, limited-time discounts, short survey to identify barriers.
Potential Revenue Impact: $4,949,011.

- Cluster 1 — At-Risk, Low Engagement
Goal: Nurture and activate newer/low-engagement customers.
Actions: Welcome/onboarding sequences, first-purchase discounts or bundles, highlight best-sellers & reviews, tips/guides to build trust.
Potential Revenue Impact: $2,241,851.

## What’s inside
- AI Agent (Segmentation): Automatically clusters customers (RFM + behavior features), labels personas, and highlights largest / at-risk groups.

- Cluster Explorer: Switch metrics to compare behavioral differences across clusters (R, F, M, AOV, diversity).

- Hierarchical View: Dendrogram to inspect sub-structure and choose meaningful cut levels.

- Market Basket: Apriori frequent itemsets + association rules (support, confidence, lift) and strongest pairs (Jaccard).

- Snapshot Panel: Auto-generated narrative with compact chips (top items, strongest pairs).

- ROI Mini-Calculator: Enter your numbers to estimate campaign ROI.

## Demo
Live app: https://customersegmentation-m6xdccdfr7cfhqwaw3tyjx.streamlit.app/

## Setup
1) Prerequisites
- Python 3.10 or 3.11

- Git (optional)

- A Kaggle account to download the dataset is optional. CSV file can be found on repository.

2) Clone and create a virtual environment

git clone https://github.com/<you>/<repo>.git
cd <repo>

# Create & activate venv
python -m venv .venv
- Windows
- 
  .venv\Scripts\activate
- macOS/Linux
  
    source .venv/bin/activate

## Install dependencies 

pip install -r requirements.txt

- Minimal requirements.txt (Can be found on repository)


## Dataset Overview
Source: https://www.kaggle.com/datasets/yusufdelikkaya/online-sales-dataset

Business: Online Sales Dataset

Time Period: January 2020 – May 2025

Size: 49,782 transactions

You can point the app to your local CSV or use the copy included in /data.

## Expected columns
- At minimum, the CSV should contain:
CustomerID, InvoiceDate, Quantity, UnitPrice, StockCode, Description

- The app derives/uses features such as:
Recency, Frequency, Monetary, UniqueProducts, TotalTransactions, ProductDiversity, AvgItemPrice, ReturnsRate, DiscountUsage, CategoryDiversity

## Download precomputed files
Run your data on the Juptier notebook and save all the files to use on Streamlit app 

## Run the app
streamlit run app.py

## Disclaimer

Support and confidence values may appear low because the dataset contains a large proportion of one-time customers. This creates a sparse basket matrix, so even meaningful co-purchases occur in a very small fraction of transactions.

## Acknowledgements
- This project draws on course materials by Prof. Mr. Avinash Jairam (CIS 9660 – Data Mining for Business Analytics) for parts of the preprocessing and modeling setup.
- I additionally leveraged ChatGPT for troubleshooting Streamlit code, improving UI/UX, and refining implementation details. All final modeling choices, analyses, and interpretations are solely my responsibility.
