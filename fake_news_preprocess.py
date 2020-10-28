import pandas as pd


# 簡單的數據清理，去除空白標題的 examples
df_train = pd.read_csv("dataset/FakeNews/train.csv")
empty_title = ((df_train['title2_zh'].isnull()) \
               | (df_train['title1_zh'].isnull()) \
               | (df_train['title2_zh'] == '') \
               | (df_train['title2_zh'] == '0'))
df_train = df_train[~empty_title]

# 剔除過長的樣本以避免 BERT 無法將整個輸入序列放入記憶體不多的 GPU
MAX_LENGTH = 30
df_train = df_train[~(df_train.title1_zh.apply(lambda x : len(x)) > MAX_LENGTH)]
df_train = df_train[~(df_train.title2_zh.apply(lambda x : len(x)) > MAX_LENGTH)]
# print(df_train.title2_zh[0])

# 只用 1% 訓練數據看看 BERT 對少量標註數據有多少幫助
SAMPLE_FRAC = 0.01
df_train = df_train.sample(frac=SAMPLE_FRAC, random_state=9527)

# 去除不必要的欄位並重新命名兩標題的欄位名
df_train = df_train.reset_index()
df_train = df_train.loc[:, ['title1_zh', 'title2_zh', 'label']]
df_train.columns = ['text_a', 'text_b', 'label']

# idempotence, 將處理結果另存成 tsv 供 PyTorch 使用
df_train.to_csv("dataset/FakeNews/train.tsv", sep="\t", index=False)

# 觀察一下 training set
print("訓練樣本數：", len(df_train))
print(df_train.head())
print(df_train.label.value_counts() / len(df_train))

# 處裡 test set
df_test = pd.read_csv("dataset/FakeNews/test.csv")
df_test = df_test.loc[:, ["title1_zh", "title2_zh", "id"]]
df_test.columns = ["text_a", "text_b", "Id"]
df_test.to_csv("dataset/FakeNews/test.tsv", sep="\t", index=False)

# 觀察一下 test set
print("預測樣本數：", len(df_test))
print(df_test.head())
print(df_train.label.value_counts() / len(df_train))
