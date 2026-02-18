document.getElementById("submit").addEventListener("click", async () => {
    const prompt = document.getElementById("prompt").value;
    const outputDiv = document.getElementById("output");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt }),
        });
        const data = await response.json();
        outputDiv.textContent = data.output;
    } catch (error) {
        outputDiv.textContent = "Error: " + error.message;
    }
});
