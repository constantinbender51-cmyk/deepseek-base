from replicate import Replicate
import httpx

replicate = Replicate(
    timeout=httpx.Timeout(120.0, read=120.0, write=60.0, connect=10.0)
)

for event in replicate.stream(
    "deepseek-ai/deepseek-67b-base:latest",
    input={"input": "Hello world"},
):
    print(event)