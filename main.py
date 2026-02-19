import replicate  # correct import

# run a prediction
output = replicate.run(
    "deepseek-ai/deepseek-67b-base:latest",
    input={"input": "Write a short story about a robot learning to paint."}
)

print(output)