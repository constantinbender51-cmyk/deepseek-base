import replicate

input = {"the thing you habg your coat on so it doesn't get wrinkls is called a"}

for event in replicate.stream(
    "deepseek-ai/deepseek-67b-base:0f2469607b150ffd428298a6bb57874f3657ab04fc980f7b5aa8fdad7bd6b46b",
    input=input
):
    print(event, end="")
    #=> " computed"