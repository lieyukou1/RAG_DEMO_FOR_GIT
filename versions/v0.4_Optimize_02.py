"""
超简单 RAG Demo - 通义千问版本
功能：基于文档的问答系统

使用前需要安装：
pip install dashscope chromadb
"""

import dashscope
import chromadb
from chromadb.utils import embedding_functions

# 1. 准备一些示例文档（实际项目中这里会读取真实文件）
documents = [
    "Python是一种高级编程语言，由Guido van Rossum在1991年创建。",
    "Python的设计哲学强调代码的可读性和简洁的语法，特别是使用空格缩进来表示代码块。",
    "Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。",
    "Python有丰富的标准库，被称为'自带电池'的语言。",
    "机器学习是人工智能的一个分支，让计算机能够从数据中学习并做出决策。",
    "深度学习是机器学习的子集，使用多层神经网络来学习数据的复杂模式。"
]

# 2. 初始化 Chroma 向量数据库（本地内存模式）
client = chromadb.Client()
collection = client.create_collection(
    name="my_docs",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

# 3. 将文档存入向量数据库
for i, doc in enumerate(documents):
    collection.add(
        documents=[doc],
        ids=[f"doc_{i}"]
    )

print("✅ 文档已加载到向量数据库")


# 4. 问答函数
def ask_question(question: str) -> str:
    # 4.1 检索相关文档（找最相关的3个，增加覆盖面）
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    context = "\n".join(results['documents'][0])

    # Debug: 打印检索到的文档
    print(f"\n🔍 检索到的文档:")
    for i, doc in enumerate(results['documents'][0], 1):
        print(f"  {i}. {doc}")
    print()

    # 4.2 调用通义千问生成答案
    dashscope.api_key = "sk-b6c13c57c648404b95bbffb80baa0133"  # 替换成你的通义千问API密钥

    response = dashscope.Generation.call(
        model='qwen-plus',  # 使用 qwen-plus 模型（免费额度充足）
        prompt=f"""你是一个专业的问答助手。请仔细阅读下面的文档内容，从中提取信息回答用户的问题。

文档内容：
{context}

用户问题：{question}

回答要求：
1. 如果文档中明确提到了答案，请直接回答
2. 如果文档中有相关信息但不够完整，请基于已有信息回答
3. 只有在文档完全没有相关信息时，才说"文档中没有找到相关信息"

请用中文简洁回答：""",
        result_format='message'
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        return f"调用失败: {response.message}"


# 5. 测试
if __name__ == "__main__":
    print("\n🤖 简单 RAG 问答系统已启动！\n")

    # 测试几个问题
    test_questions = [
        "Python是什么时候创建的？",
        "Python有什么特点？",
        "什么是深度学习？"
    ]

    for q in test_questions:
        print(f"❓ 问题: {q}")
        answer = ask_question(q)
        print(f"💡 答案: {answer}\n")
        print("-" * 50 + "\n")