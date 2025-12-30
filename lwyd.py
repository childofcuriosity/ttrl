# Load model directly
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
llm_path="llms/Qwen2.5-Math-1.5B"
tokenizer = AutoTokenizer.from_pretrained(llm_path)
model = AutoModelForCausalLM.from_pretrained(llm_path)

# 目标问题 (注意：我稍微调整了最后一行格式以匹配 \box{}, 这样模型效果最好)
# 如果你坚持用 "18,"，也可以改回去，但 Few-shot 最好格式统一
prompt_text = '''
Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
\\box{14}
'''

messages = [
    # Optional: 系统提示词
    {"role": "system", "content": "You are a helpful math assistant. Verify the answer provided in the box. Start your response with True or False, followed by the reasoning."},
    # --- Shot 2 (正确示例) ---
    {"role": "user", "content": "A baker makes 50 cookies. He sells 40 of them for $2 each and eats the rest. How much revenue did he generate?\n\\box{80}"},
    # 补全逻辑：先算乘法，再判对
    {"role": "assistant", "content": "True,because the baker sells 40 cookies for $2 each, so the revenue is 40 * 2 = 80. The provided answer 80 is correct."},

    # # --- Shot 1 (错误示例) ---
    # {"role": "user", "content": "Paul has 30 books. He donates 5 to the library and sells 10 to a used bookstore for $3 each. How much money did he make?\n\\box{28}"},
    # # 补全逻辑：先算乘法，再判错
    # {"role": "assistant", "content": "False"},


    # --- Target ---
    {"role": "user", "content": prompt_text},
]


# messages = [
#     # Optional: 系统提示词，定调
#     {"role": "system", "content": "You are a helpful math assistant. output only one token in A B C D."},

#     # --- Shot 1 (示例 1) ---
#     {"role": "user", "content": "Paul has 30 books. He donates 5 to the library and sells 10 to a used bookstore for $3 each. How much money did he make?\nA) 45\nB) 30\nC) 15\nD) 25"},
#     {"role": "assistant", "content": "B"},

#     # --- Shot 2 (示例 2) ---
#     {"role": "user", "content": "A baker makes 50 cookies. He sells 40 of them for $2 each and eats the rest. How much revenue did he generate?\nA) 80\nB) 100\nC) 20\nD) 40"},
#     {"role": "assistant", "content": "A"},

#     {"role": "user", "content": prompt},
# ]

# messages = [
#     # Optional: 系统提示词，定调
#     {"role": "system", "content": "You are a helpful math assistant. Solve the problem step by step and choose the correct option."},

#     # --- Shot 1 (示例 1) ---
#     {"role": "user", "content": "Paul has 30 books. He donates 5 to the library and sells 10 to a used bookstore for $3 each. How much money did he make?\nA) 45\nB) 30\nC) 15\nD) 25"},
#     {"role": "assistant", "content": "Paul sells 10 books. The price per book is $3. So, the total earnings are 10 * $3 = $30. The books donated do not generate money. The answer is 30.\nAnswer: B"},

#     # --- Shot 2 (示例 2) ---
#     {"role": "user", "content": "A baker makes 50 cookies. He sells 40 of them for $2 each and eats the rest. How much revenue did he generate?\nA) 80\nB) 100\nC) 20\nD) 40"},
#     {"role": "assistant", "content": "The baker sells 40 cookies. Each cookie is sold for $2. Total revenue is 40 * $2 = $80. The cookies he ate generated $0. The answer is 80.\nAnswer: A"},

#     {"role": "user", "content": prompt},
# ]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


# 3. 获取目标 Token 的 ID
# 注意：Qwen 的 tokenizer 可能会有不同的编码方式，这里一定要 add_special_tokens=False
# 有时候 "A" 和 " A" (前面带空格) 是不同的 token。
# 在 chat template 中，assistant 回复的开头通常紧跟换行符，所以通常是无空格的 "A"。
candidates = ["T", "F"]
candidate_ids = []
for cand in candidates:
    # 编码并取第一个 token ID
    cid = tokenizer.encode(cand, add_special_tokens=False)[0]
    candidate_ids.append(cid)
    print(f"Token '{cand}' ID: {cid}")

# 4. 执行前向传播 (Forward Pass) 获取 Logits
# 我们不需要 generate，只需要最后一步的预测分布
with torch.no_grad():
    outputs = model(**inputs)
    # 获取最后一个 token 的 logits（即预测下一个 token 的分布）
    # shape: [batch_size, seq_len, vocab_size] -> 取 [0, -1, :]
    last_token_logits = outputs.logits[0, -1, :]

# ==========================================
#  核心修改：计算全局 Softmax (绝对概率)
# ==========================================

# 对整个词表进行 Softmax，所有 token 概率之和为 1
all_probs = F.softmax(last_token_logits, dim=0)

print(f"\n{' Option ':^8} | {' Logit ':^10} | {' Absolute Prob ':^18}")
print("-" * 42)

total_abcd_prob = 0
for i, cand in enumerate(candidates):
    cid = candidate_ids[i]
    
    # 获取该 token 的 logit 和 绝对概率
    logit_val = last_token_logits[cid].item()
    abs_prob = all_probs[cid].item()
    
    total_abcd_prob += abs_prob
    
    print(f"   {cand:^4}   | {logit_val:^10.4f} | {abs_prob:^16.6%} ")

print("-" * 42)
print(f"Sum of A+B+C+D probs: {total_abcd_prob:.6%}")

# ==========================================
#  额外检查：看看模型真正想输出的前 5 个词是什么
# ==========================================
print("\n--- Global Top 5 Predicted Tokens ---")
topk_probs, topk_indices = torch.topk(all_probs, 5)

for rank, (prob, idx) in enumerate(zip(topk_probs, topk_indices)):
    token_str = tokenizer.decode([idx])
    print(f"Rank {rank+1}: Token='{token_str}' (ID: {idx.item()}), Prob={prob.item():.6%}")