import replicate

outputs = replicate.run(
    "deepseek-ai/deepseek-67b-base:0f2469607b150ffd428298a6bb57874f3657ab04fc980f7b5aa8fdad7bd6b46b",
    input={
        "prompt": """import replicate

outputs = replicate.run(
    "deepseek-ai/deepseek-67b-base:0f2469607b150ffd428298a6bb57874f3657ab04fc980f7b5aa8fdad7bd6b46b",
    input={
        "prompt": "The thing you hang clothes on so they don't wrinkle is called a "
    }
)

# DeepSeek on Replicate usually returns a generator
for text in outputs:
    print(text, end="")

Now here is the code with a nice visual UI accessable via web browser: """
    }
)

# DeepSeek on Replicate usually returns a generator
for text in outputs:
    print(text, end="")