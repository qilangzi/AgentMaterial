from openai import OpenAI


def call_deepseek_with_openai(prompt, model="deepseek-r1:1.5b", host="localhost", port=11434):
    """
    使用openai库调用Ollama服务中的deepseek-r1:1.5b模型

    参数:
        prompt: 输入的提示文本
        model: 模型名称
        host: Ollama服务主机地址
        port: Ollama服务端口号

    返回:
        模型的响应文本
    """
    # 配置客户端，指向本地Ollama服务
    client = OpenAI(
        base_url=f"http://{host}:{port}/v1",  # Ollama的OpenAI兼容API端点
        api_key="ollama",  # Ollama不需要实际API密钥，这里任意填写即可
    )

    try:
        # 调用聊天完成接口
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        # 提取并返回响应内容
        return response.choices[0].message.content

    except Exception as e:
        return f"调用出错：{str(e)}"


if __name__ == "__main__":
    # 示例使用
    user_prompt = "请简要介绍一下机器学习的基本概念"

    print(f"使用 deepseek-r1:1.5b 模型，提示词：{user_prompt}")
    print("正在获取响应...\n")

    # 调用模型
    response = call_deepseek_with_openai(user_prompt)

    # 输出结果
    print("模型响应：")
    print("-" * 50)
    print(response)
    print("-" * 50)
