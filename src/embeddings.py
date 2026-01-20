"""
    管理embedding类
"""
import dashscope
from dashscope import TextEmbedding
from src.config import DASHSCOPE_API_KEY  # 导入配置

# 设置API Key
dashscope.api_key = DASHSCOPE_API_KEY

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
