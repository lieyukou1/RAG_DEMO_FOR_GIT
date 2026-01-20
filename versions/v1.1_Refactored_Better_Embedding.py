"""
超简单 RAG Demo - 通义千问版本（优化版）
功能：基于文档的问答系统
版本：v1.2 - 2026.01.19 - 包含文件读取、优化提示词、对话历史

使用前需要安装：
pip install dashscope chromadb
"""

import dashscope
import chromadb
from dashscope import TextEmbedding

# ==================== 配置常量 ====================
# 提示词模板集中管理
BASIC_PROMPT_TEMPLATE = """你是一个专业的问答助手。请仔细阅读下面的文档内容，从中提取信息回答用户的问题。

文档内容：
{context}

用户问题：{question}

回答要求：
1. 如果文档中明确提到了答案，请直接回答
2. 如果文档中有相关信息但不够完整，请基于已有信息回答
3. 只有在文档完全没有相关信息时，才说"文档中没有找到相关信息"

请用中文简洁回答："""

TEACHER_PROMPT_TEMPLATE = """你是一个优秀的学习助手，请基于下面的知识库内容，用自然、易懂的方式回答问题。

相关背景知识：
{context}

用户问题：{question}

请按照以下要求回答：
1. 首先理解文档中的核心概念
2. 用你自己的话解释，而不是直接复制原文
3. 如果文档中有例子，可以用自己的话重述例子
4. 保持回答简洁明了，适合学习者理解
5. 如果文档信息不足，可以基于常识补充，但要注明"基于一般知识"

请开始回答："""

# 文件路径常量
KNOWLEDGE_FILE = "data/your_notes.txt"
VECTOR_DB_NAME = "my_docs"

# 检索参数
TOP_K_RESULTS = 3
HISTORY_WINDOW_SIZE = 3  # 对话历史窗口大小

# ==================== API配置 ====================
# 设置 API Key
dashscope.api_key = "sk-b6c13c57c648404b95bbffb80baa0133"


# ==================== Embedding类 ====================
class QwenEmbeddingFunction:
    """通义千问 Embedding 函数封装"""

    def _get_embeddings(self, texts):
        """调用通义千问 API 获取 embeddings"""
        if isinstance(texts, str):
            texts = [texts]

        try:
            response = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v3,
                input=texts
            )

            print(f"API 响应状态: {response.status_code}")
            if response.status_code != 200:
                print(f"API 错误: {response.code} - {response.message}")
                raise Exception(f"Embedding API 调用失败: {response.message}")

            return [item['embedding'] for item in response.output['embeddings']]
        except Exception as e:
            print(f"❌ Embedding 调用出错: {e}")
            raise

    def __call__(self, input):
        """Chroma 会调用这个方法"""
        return self._get_embeddings(input)

    def embed_documents(self, texts):
        """存储文档时调用"""
        return self._get_embeddings(texts)

    def embed_query(self, input):
        """查询时调用 - 注意 Chroma 可能传入列表"""
        if isinstance(input, list):
            if len(input) > 0:
                query_text = input[0]
            else:
                raise ValueError("input 列表为空")
        else:
            query_text = input

        if not isinstance(query_text, str):
            query_text = str(query_text)

        embeddings = self._get_embeddings([query_text])
        return embeddings


# ==================== 文档处理函数 ====================
def load_documents_from_file(file_path=KNOWLEDGE_FILE):
    """从文件中读取文档并按段落分割"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 按段落分割（每个段落是一个知识点）
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    return paragraphs


# ==================== 初始化向量数据库 ====================
def initialize_vector_database():
    """初始化Chroma向量数据库并加载文档"""
    client = chromadb.Client()
    collection = client.create_collection(
        name=VECTOR_DB_NAME,
        embedding_function=QwenEmbeddingFunction()
    )

    documents = load_documents_from_file()
    for i, doc in enumerate(documents):
        collection.add(
            documents=[doc],
            ids=[f"doc_{i}"]
        )

    print(f"✅ 已加载 {len(documents)} 个文档到向量数据库")
    return collection


# ==================== 核心RAG函数 ====================
def retrieve_context(collection, question, top_k=TOP_K_RESULTS):
    """检索相关文档"""
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    print(f"\n🔍 检索到的文档:")
    for i, doc in enumerate(results['documents'][0], 1):
        print(f"  {i}. {doc[:100]}...")  # 只打印前100字符
    print()

    context = "\n".join(results['documents'][0])
    return context


def ask_question(question: str, prompt_template=TEACHER_PROMPT_TEMPLATE) -> str:
    """核心问答函数"""
    # 1. 检索相关文档
    context = retrieve_context(collection, question)

    # 2. 构造提示词
    prompt = prompt_template.format(context=context, question=question)

    # 3. 调用通义千问生成答案
    response = dashscope.Generation.call(
        model='qwen-plus',
        prompt=prompt,
        result_format='message'
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        return f"调用失败: {response.message}"


# ==================== 对话历史管理 ====================
conversation_history = []


def ask_question_with_history(question):
    """带历史上下文的问答"""
    answer = ask_question(question)

    # 保存到历史
    conversation_history.append(f"Q: {question}")
    conversation_history.append(f"A: {answer}")

    # 保持历史窗口大小
    if len(conversation_history) > HISTORY_WINDOW_SIZE * 2:
        conversation_history.pop(0)
        conversation_history.pop(0)

    return answer


def get_recent_history(window_size=HISTORY_WINDOW_SIZE):
    """获取最近的对话历史"""
    return conversation_history[-(window_size * 2):] if conversation_history else []


# ==================== 测试函数 ====================
def run_test_questions(questions, use_history=False):
    """运行测试问题集"""
    print("\n" + "=" * 60)
    print("🤖 RAG 问答系统测试开始")
    print("=" * 60 + "\n")

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] 问题: {q}")

        if use_history:
            answer = ask_question_with_history(q)
        else:
            answer = ask_question(q)

        print(f"💡 答案: {answer}")
        print("-" * 50)

    if use_history:
        print(f"\n📝 对话历史: {get_recent_history()}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 初始化向量数据库
    collection = initialize_vector_database()

    # 测试问题集
    test_questions = [
        "什么是过拟合？如何解决？",
        "机器学习的要素有哪些？",
        "CNN和RNN分别用于什么？",
        "什么是Embedding？",
        "RAG有什么优势？",
        "激活函数的作用是什么？"
    ]

    # 运行测试（不带历史）
    run_test_questions(test_questions, use_history=False)

    # 清空历史，重新测试带历史的版本
    conversation_history.clear()
    print("\n\n" + "=" * 60)
    print("🔄 开始带历史上下文的测试")
    print("=" * 60 + "\n")

    run_test_questions(test_questions[:3], use_history=True)  # 只测试前3个

    print("\n🎉 测试完成！")