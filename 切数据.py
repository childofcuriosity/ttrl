import pandas as pd
names=['train','test']
for name in names:
    # 读取 parquet 文件
    df = pd.read_parquet(f"data/MATH-TTT/{name}.parquet")
    # 取前 32 条
    df_small = df.head(32)

    # 保存为新的 parquet（可选）
    df_small.to_parquet(f"data/MATH-TTT/{name}_debug.parquet")


    print(df_small.shape)
    print(df_small.head())
