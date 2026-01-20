"""
超简单 RAG Demo - 通义千问版本（优化版）
功能：基于文档的问答系统

使用前需要安装：
pip install dashscope chromadb
"""

import dashscope
import chromadb
from dashscope import TextEmbedding

# 设置 API Key（在初始化时就设置）
dashscope.api_key = "sk-b6c13c57c648404b95bbffb80baa0133"  # 替换成你的通义千问API密钥


# 自定义 Embedding 函数，使用通义千问的 text-embedding-v3
class QwenEmbeddingFunction:
    # 获取嵌入
    def _get_embeddings(self, texts):
        """调用通义千问 API 获取 embeddings"""
        # 统一为list格式
        if isinstance(texts, str):
            texts = [texts]

        try:
            # 根据输入和模型生成回答
            response = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v3,
                input=texts
            )

            # 调试：打印响应
            print(f"API 响应状态: {response.status_code}")
            if response.status_code != 200:
                print(f"API 错误: {response.code} - {response.message}")
                raise Exception(f"Embedding API 调用失败: {response.message}")

            embeddings = [item['embedding'] for item in response.output['embeddings']]
            return embeddings
        except Exception as e:
            print(f"❌ Embedding 调用出错: {e}")
            raise

    # 内部调用的魔法方法
    def __call__(self, input):
        """Chroma 会调用这个方法"""
        return self._get_embeddings(input)

    # 存储文档，chroma期望的函数
    def embed_documents(self, texts):
        """存储文档时调用"""
        return self._get_embeddings(texts)

    # 同上，但需要做格式转换
    def embed_query(self, input):
        """查询时调用 - 注意 Chroma 可能传入列表"""
        # 如果 input 是列表，取第一个元素
        if isinstance(input, list):
            if len(input) > 0:
                query_text = input[0]
            else:
                raise ValueError("input 列表为空")
        else:
            query_text = input

        # 确保是字符串
        if not isinstance(query_text, str):
            query_text = str(query_text)

        # 返回列表形式（Chroma 期望的格式）
        embeddings = self._get_embeddings([query_text])
        return embeddings  # 返回整个列表，不要取 [0]


# # 1. 准备一些示例文档（实际项目中这里会读取真实文件）
# documents = [
#     "Python是一种高级编程语言，由Guido van Rossum在1991年创建。",
#     "Python的设计哲学强调代码的可读性和简洁的语法，特别是使用空格缩进来表示代码块。",
#     "Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。",
#     "Python有丰富的标准库，被称为'自带电池'的语言。",
#     "机器学习是人工智能的一个分支，让计算机能够从数据中学习并做出决策。",
#     "深度学习是机器学习的子集，使用多层神经网络来学习数据的复杂模式。"
# ]

# 1->1.1 从文件中读取文档并按行拆分成知识点
def load_documents_from_file(file_path=r"data\your_notes.txt"):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 按段落分割（每个段落是一个知识点）
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    return paragraphs

# 2. 初始化 Chroma 向量数据库（使用通义千问的 embedding）
client = chromadb.Client()
collection = client.create_collection(
    name="my_docs",
    embedding_function=QwenEmbeddingFunction()  # 使用通义千问的 embedding
)

# 3. 将文档存入向量数据库
documents = load_documents_from_file(file_path="../data/your_notes.txt")
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

    # 4.2 调用通义千问生成答案（不需要再设置 api_key，已在开头设置）
    response = dashscope.Generation.call(
        model='qwen-plus',  # 使用 qwen-plus 模型（免费额度充足）

#         prompt=f"""你是一个专业的问答助手。请仔细阅读下面的文档内容，从中提取信息回答用户的问题。
#
# 文档内容：
# {context}
#
# 用户问题：{question}
#
# 回答要求：
# 1. 如果文档中明确提到了答案，请直接回答
# 2. 如果文档中有相关信息但不够完整，请基于已有信息回答
# 3. 只有在文档完全没有相关信息时，才说"文档中没有找到相关信息"
#
# 请用中文简洁回答：""",
        # 1.3 修改提示词工程
        prompt=f"""你是一个优秀的学习助手，请基于下面的知识库内容，用自然、易懂的方式回答问题。

    相关背景知识：
    {context}

    用户问题：{question}

    请按照以下要求回答：
    1. 首先理解文档中的核心概念
    2. 用你自己的话解释，而不是直接复制原文
    3. 如果文档中有例子，可以用自己的话重述例子
    4. 保持回答简洁明了，适合学习者理解
    5. 如果文档信息不足，可以基于常识补充，但要注明"基于一般知识"

    请开始回答：""",
        result_format='message'
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        return f"调用失败: {response.message}"


# 1.2.a 添加对话历史（简单版）
conversation_history = []


def ask_question_with_history(question):
    # 将历史对话加入上下文
    history_text = "\n".join(conversation_history[-3:])  # 最近3轮
    # enhanced_question = f"对话历史：{history_text}\n当前问题：{question}" # 这个是增强过后的问题，如果
    # # ask_question接收的是增强对话，那么函数所记录的历史记录是检索到的知识点而不是实际对话的Q&A，故应当注释掉

    # 原有检索逻辑...
    answer = ask_question(question)

    # 保存到历史
    conversation_history.append(f"Q: {question}")
    conversation_history.append(f"A: {answer}")

    return answer


# 1.2.b. 添加简单的答案评估(暂时不做该功能)
def evaluate_answer(question, answer):
    """简单评估答案质量"""
    criteria = {
        "是否相关": "答案是否直接回答问题",
        "是否完整": "是否覆盖了问题的各个方面",
        "是否准确": "基于文档内容是否正确"
    }
    # 这里可以添加更复杂的评估逻辑
    return criteria

# 5. 测试
if __name__ == "__main__":
    print("\n🤖 简单 RAG 问答系统已启动！\n")

    # # 1. 测试几个问题
    # test_questions = [
    #     "Python是什么时候创建的？",
    #     "Python有什么特点？",
    #     "什么是深度学习？"
    # ]
    # 1.1 问题
    test_questions = [
        "什么是过拟合？如何解决？",  # 综合问题
        "机器学习的要素有哪些？",  # 直接检索
        "CNN和RNN分别用于什么？",  # 对比问题
        "什么是Embedding？",  # 概念解释
        "RAG有什么优势？",  # 应用理解
        "激活函数的作用是什么？"  # 基础概念
    ]


    for q in test_questions:
        print(f"❓ 问题: {q}")
        answer = ask_question(q)
        print(f"💡 答案: {answer}\n")
        ask_question_with_history(q)
        print("-" * 50 + "\n")

    print(f'当前窗口对话历史：{conversation_history}')