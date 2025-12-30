import matplotlib
# 重要：在导入 pyplot 之前设置后端为 Agg，防止在无 GUI 服务器上报错
matplotlib.use('Agg') 

import pandas as pd
import ast
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
# 引用服务器上的 verl 路径
from verl.utils.reward_score.ttrl_math import extract_answer, grade


class MathEquivalenceUF:
    """完全对齐原始代码的并查集实现"""
    def __init__(self, fast_mode=False): # 这里强制设为 False
        self.parent = {}
        self.fast_mode = fast_mode
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        if x is None or y is None:
            return False
        if self._are_equivalent(x, y):
            rx, ry = self.find(x), self.find(y)
            if rx != ry:
                self.parent[ry] = rx
            return True
        return False
    
    def _are_equivalent(self, a, b):
        try:
            # 这里的 fast=self.fast_mode 将在调用处传入 False
            return grade(a, b, fast=self.fast_mode) and grade(b, a, fast=self.fast_mode)
        except Exception:
            return False

def calculate_pseudo_for_subset(extracted_answers, gt, n):
    """
    完全对齐原始 _covalidate 的投票判定逻辑：
    1. 取前 n 个样本
    2. 构建等价类
    3. 过滤空字符串后的 Majority Vote
    4. grade(representative, gt, fast=False)
    """
    subset = extracted_answers[:n]
    if not subset:
        return False
    
    # 1. 还原并查集逻辑 (fast_mode 必须为 False)
    uf = MathEquivalenceUF(fast_mode=False) 
    
    for idx_a in range(len(subset)):
        a = subset[idx_a]
        # 获取当前的代表元
        represents = [x for x, p in uf.parent.items() if x == p]
        matched = False
        for rep in represents:
            if uf.union(a, rep):
                matched = True
                break
        if not matched and a not in uf.parent:
            uf.parent[a] = a
            
    # 2. 还原计数逻辑
    vote_counter = Counter()
    for ans in subset:
        root = uf.find(ans)
        vote_counter[root] += 1
        
    # 3. 还原“非空优先”的代表元选择逻辑 (关键细节对齐)
    non_empty_votes = [(ans, cnt) for ans, cnt in vote_counter.most_common() if ans.strip() != ""]
    
    if len(non_empty_votes) > 0:
        representative, vote_count = non_empty_votes[0]
    else:
        # 全部都是空字符串 —— fallback
        if not vote_counter: return False
        representative, vote_count = vote_counter.most_common(1)[0]
        
    # 4. 判定正确性 (fast 必须为 False)
    return grade(representative, gt, fast=False)
def analyze_3d_performance(expname):
    # 匹配文件逻辑
    pattern = os.path.join(expname, "covalidate_preload_t=*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"Error: No files found in {expname}")
        return

    n_votes_list = [16, 32, 48, 64]
    plot_data = []

    for file_path in files:
        # 正则提取温度
        match = re.search(r"t=(\d+\.?\d*)", os.path.basename(file_path))
        if not match: continue
        temp = float(match.group(1))
        
        print(f"Processing Temperature t={temp} ...")
        df = pd.read_csv(file_path)
        
        # 将字符串列表解析回 Python 列表
        df['votes_extract'] = df['votes_extract'].apply(ast.literal_eval)
        
        for n in n_votes_list:
            correct_count = 0
            for idx, row in df.iterrows():
                if calculate_pseudo_for_subset(row['votes_extract'], row['ground_truth'], n):
                    correct_count += 1
            
            acc = (correct_count / len(df)) * 100
            plot_data.append({'temp': temp, 'n_votes': n, 'acc': acc})
            print(f"  N={n} | Accuracy = {acc:.2f}%")

    # --- 绘图逻辑 ---
    df_plot = pd.DataFrame(plot_data).sort_values(['temp', 'n_votes'])
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    temps = sorted(df_plot['temp'].unique())
    votes = n_votes_list
    _x, _y = np.meshgrid(np.arange(len(temps)), np.arange(len(votes)))
    x, y = _x.ravel(), _y.ravel()
    
    z = np.zeros_like(x)
    dz = []
    for i in range(len(x)):
        t = temps[x[i]]
        v = votes[y[i]]
        val = df_plot[(df_plot['temp'] == t) & (df_plot['n_votes'] == v)]['acc'].values[0]
        dz.append(val)

    dx = dy = 0.4
    colors = plt.cm.plasma(np.array(dz) / 100.0)
    ax.bar3d(x, y, z, dx, dy, dz, color=colors, alpha=0.8)

    ax.set_xticks(np.arange(len(temps)) + dx/2)
    ax.set_xticklabels(temps)
    ax.set_yticks(np.arange(len(votes)) + dy/2)
    ax.set_yticklabels(votes)
    
    ax.set_xlabel('Temperature (t)')
    ax.set_ylabel('Number of Votes (N)')
    ax.set_zlabel('Accuracy (%)')
    ax.set_zlim(0, 100)
    ax.set_title(f'Strict 3D Accuracy Analysis: {expname}')

    # 保存图片
    output_img = os.path.join(expname, "3d_vote_analysis_server.png")
    plt.savefig(output_img, dpi=150)
    print(f"\n[DONE] Plot saved to: {output_img}")

if __name__ == "__main__":
    try:
        with open("expname.txt", "r", encoding="utf-8") as f:
            expname = f.readline().strip()
    except:
        expname = "."
    analyze_3d_performance(expname)