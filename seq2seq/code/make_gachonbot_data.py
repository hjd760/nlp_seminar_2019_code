import pandas as pd

golden = pd.read_csv('../datasets/golden.tsv', delimiter='\t', header=None)
intention = pd.read_csv('../datasets/intention.tsv', delimiter='\t', header=None)

golden = golden[[2, 4]]
intention = intention[[1, 3]]
golden.columns = ['라벨번호', '질문']
intention.columns = ['라벨번호', '답변']
datasets = pd.merge(golden, intention, on='라벨번호')
datasets = datasets[['질문', '답변']]
datasets.to_csv('../datasets/gachon_chatbot.tsv', header=False, index=False, sep='\t')
