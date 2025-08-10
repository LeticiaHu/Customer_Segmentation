## Unsupervised Learning AI Agent: Customer Segmentation on Online Retail
A Streamlit + Python project that uses unsupervised learning (K-Means and hierarchical clustering) to segment customers from an online retail dataset and surface persona insights, cluster comparisons, and market basket patterns for cross-sell ideas.

## Why it matters
🎯 Personalize marketing to distinct segments

📦 Improve product recommendations and bundles

💰 Boost retention & sales efficiency with targeted outreach

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
  
streamlit==1.37.1

pandas==2.2.2

numpy==1.26.4

scikit-learn==1.5.1

mlxtend==0.23.1

plotly==5.22.0

joblib==1.4.2


## Dataset Overview
Source: https://www.kaggle.com/datasets/yusufdelikkaya/online-sales-dataset

Business: Online Sales Dataset

Time Period: January 2020 – May 2025

Size: 49,782 transactions

You can replace the path/loader in the app to point to your local copy of the dataset.

## Expected columns
- Your CSV should include (at minimum):

CustomerID, InvoiceDate, Quantity, UnitPrice, StockCode, Description

- The app will create/expect derived features like:

Recency, Frequency, Monetary, UniqueProducts, TotalTransactions,
ProductDiversity, AvgItemPrice, ReturnsRate, DiscountUsage, CategoryDiversity

## Download precomputed files
Run your data on the Juptier notebook and save all the files to use on Streamlit app 

## Run the app
streamlit run app.py

## Acknowledgements
- This project draws on course materials by Prof. Mr. Avinash Jairam (CIS 9660 – Data Mining for Business Analytics) for parts of the preprocessing and modeling setup.
- I additionally leveraged ChatGPT for troubleshooting Streamlit code, improving UI/UX, and refining implementation details. All final modeling choices, analyses, and interpretations are solely my responsibility.
