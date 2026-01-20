"""
    核心RAG逻辑
"""
import dashscope
import chromadb
from src.embeddings import QwenEmbeddingFunction
from src.config import *


# ==================== 文档处理函数 ====================
def load_documents_from_file(file_path=KNOWLEDGE_FILE):
    """从文件中读取文档并按段落分割"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 按段落分割（每个段落是一个知识点）
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    return paragraphs


# ==================== 核心RAG函数 ====================
collection = None


def get_collection():
    global collection
    if collection is None:
        print("正在初始化向量数据库...")
        collection = initialize_vector_database()
    return collection


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
    # 获取collection
    current_collection = get_collection()

    # 1. 检索相关文档
    context = retrieve_context(current_collection, question)

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


# ==================== 初始化向量数据库 ====================
def initialize_vector_database():
    global collection


    """初始化Chroma向量数据库并加载文档"""
    client = chromadb.Client()
    try:
        collection = client.create_collection(
            name=VECTOR_DB_NAME,
        )
        print(f"✅ 找到现有集合: {VECTOR_DB_NAME} (已有 {collection.count()} 个文档)")
    except:
        # 集合不存在，创建新集合
        print(f"🆕 创建新集合: {VECTOR_DB_NAME}")
        collection = client.create_collection(
            name=VECTOR_DB_NAME,
            embedding_function=QwenEmbeddingFunction()
        )

    if collection.count() == 0:
        print("正在加载文档到向量数据库...")
        documents = load_documents_from_file()
        for i, doc in enumerate(documents):
            collection.add(
                documents=[doc],
                ids=[f"doc_{i}"]
            )

        print(f"✅ 已加载 {len(documents)} 个文档到向量数据库")
    else:
        print(f"✅ 集合已有 {collection.count()} 个文档，无需重复添加")
    return collection


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
