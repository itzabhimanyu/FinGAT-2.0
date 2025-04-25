This project uses various Python libraries for building a Graph Attention Network (GAT) model for stock movement prediction using financial data.
- os
- hashlib
- itertools
- datetime
- numpy
- pandas
- matplotlib.pyplot
- yfinance
- scikit-learn
  - StandardScaler
  - train_test_split
  - accuracy_score
  - precision_score
- torch
  - torch.nn
  - torch.nn.functional
  - torch.optim.AdamW
- torch_geometric
  - GATConv
  - GraphNorm
  - from_networkx
- networkx
- tqdm
How to install  libraries
pip install numpy pandas matplotlib yfinance tqdm scikit-learn torch torch-geometric network

CSV file named ind_nifty500list_filtered_final.csv passed in the code

store.py downloads the data from yahoo finance into a folder called data which has a file named all_data.pkl with all the data
i.py gives mrr,irr,top k stocks,accuracy,precision at k=5,10,15,20
w.py stores daily predictions from ='2025-01-11' to '2025-03-22 in a folder called daily_preds which has daily csv file data in the format predictions_YYYYMMDD.csv.
