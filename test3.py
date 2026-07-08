import asyncio
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(encoding='utf-8')

model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE")
)

def demo_message_objects():
    """ 显式"""
    message = [
        SystemMessage(content="你是一个专业的数学助手，回答要简短。"),
        HumanMessage(content="你是谁？") 
    ]
    response = model.invoke(message)
    print(type(response), response.content[:80]if response.content else "No content")


def demo_tuple_list():
    """ 元祖列表"""
    message = [
        ("system", "你是一个专业的数学助手，回答要简短。"),
        ("human", "你是谁？") 
    ]
    response = model.invoke(message)
    print(type(response), response.content[:80]if response.content else "No content")


def demo_dict_list():
    """ 字典列表"""
    message = [
        {"role": "system", "content": "你是一个专业的数学助手，回答要简短。"},
        {"role": "human", "content": "你是谁？"} 
    ]
    response = model.invoke(message)
    print(type(response), response.content[:80]if response.content else "No content")


async def demo_async():
    """" 异步调用"""
    response = await model.ainvoke([("user", "用一句话说明什么是素数")])
    print(type(response), response.content[:80]if response.content else "No content")



if __name__ == "__main__":
    print("---------- 1. 显式消息对象调用 ---------")
    demo_message_objects()
    print("---------- 2. 元祖列表调用 ---------")
    demo_tuple_list()
    print("---------- 3. 字典列表调用 ---------")
    demo_dict_list()
    print("---------- 4. 异步调用+元祖 ---------")
    asyncio.run(demo_async())

""""""