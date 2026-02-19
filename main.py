import replicate

try:
    result = replicate.run(
        "deepseek-ai/deepseek-67b-base:0f2469607b150ffd428298a6bb57874f3657ab04fc980f7b5aa8fdad7bd6b46b",
        input={"input": "Write a short story about a robot learning to paint."}
    )
    print(result)

except replicate.RateLimitError as e:
    print("Rate limit hit – wait and retry:", e)

except replicate.APIStatusError as e:
    print("API returned non-success status:", e.status_code, e.response)

except replicate.APIConnectionError as e:
    print("Connection issue:", str(e))

except Exception as e:
    print("Other error:", str(e))