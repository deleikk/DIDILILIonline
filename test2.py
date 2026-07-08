import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv(encoding='utf-8')

model = init_chat_model(
    model="deepseek-v4-flash" ,
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE"),
    temperature=0.3
)


print(model.invoke("写一句关于春天的词，14 字以内"))
# <class 'langchain_openai.chat_models.base.ChatOpenAI'>
print(type(model))
# <class 'str'>
print(type(model.invoke("写一句关于春天的词，14 字以内").content))
# <class 'langchain_core.messages.ai.AIMessage'>
print(type(model.invoke("写一句关于春天的词，14 字以内")))

# ========== 3. 多次调用观察参数效果（如 temperature 对多样性的影响） ==========
# 你可以把 temperature 改成 0、0.7、1.2 等再运行，对比回答是否更稳定、更多样。
for i in range(3):
    print(f"--- 第 {i + 1} 次 ---")
    print(model.invoke("写一句关于春天的词，14 字以内").content)
