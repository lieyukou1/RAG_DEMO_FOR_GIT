"""
git版本1.2.0
超简单 RAG Demo - 通义千问版本（优化版）
功能：基于文档的问答系统
版本：v1.2 - 2026.01.19 - 包含文件读取、优化提示词、对话历史

使用前需要安装：
pip install dashscope chromadb
"""
import chromadb

from src.rag_core import (
    initialize_vector_database,
    ask_question,
    ask_question_with_history,
    conversation_history,
    get_recent_history,
    VECTOR_DB_NAME
)


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


# ==================== 清理函数 ====================
def clear_vector_database():
    """清理向量数据库"""
    global collection
    client = chromadb.Client()

    try:
        client.delete_collection(name=VECTOR_DB_NAME)
        collection = None
        print(f"🗑️  已清理集合: {VECTOR_DB_NAME}")
    except Exception as e:
        print(f"清理失败: {e}")


# ==================== 主程序 ====================
if __name__ == "__main__":

    # 重新开始时清理：
    clear_vector_database()

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
