import replicate

outputs = replicate.run(
    "deepseek-ai/deepseek-67b-base:0f2469607b150ffd428298a6bb57874f3657ab04fc980f7b5aa8fdad7bd6b46b",
    input={"input": "WThe the you hang clothes so they don't wrinkle is called a "}
)

# If outputs is a generator of strings
for text in outputs:
    print(text, end="")