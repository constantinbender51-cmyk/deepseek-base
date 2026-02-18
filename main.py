import streamlit as st
import replicate
import os

# --- Page Configuration ---
st.set_page_config(page_title="DeepSeek Base AI", page_icon="🤖")

st.title("🤖 DeepSeek Base Generator")
st.markdown("Powered by Replicate & Railway. This version uses a blocking API call.")

# --- Authentication ---
# Retrieve the API key from the 'RKEY' environment variable.
# The replicate library automatically looks for REPLICATE_API_TOKEN,
# so we will set it here for the library to use.
try:
    api_token = os.environ["RKEY"]
    os.environ["REPLICATE_API_TOKEN"] = api_token
except KeyError:
    st.error("🚨 RKEY environment variable not found! Please set it in your Railway project variables.")
    st.stop()

# --- User Interface ---
with st.form("generation_form"):
    prompt = st.text_area(
        "Enter your prompt:", 
        height=150, 
        placeholder="Write a python script to list all files in a directory."
    )
    
    with st.expander("Advanced Settings"):
        temperature = st.slider("Temperature (Creativity)", 0.1, 2.0, 0.75, 0.05)
        max_tokens = st.number_input("Max New Tokens", min_value=64, max_value=4096, value=512)

    submit_button = st.form_submit_button("Generate Response")

# --- API Call Logic ---
if submit_button:
    if not prompt:
        st.warning("Please enter a prompt first.")
    else:
        # Show a spinner while the API call is in progress (blocking)
        with st.spinner("🤖 Generating response... Please wait."):
            try:
                # This is the blocking call, similar to your example's behavior.
                # The app will wait here until the entire output is generated.
                output = replicate.run(
                    "constantinbender51-cmyk/deepseek-base",
                    input={
                        "prompt": prompt,
                        "max_new_tokens": max_tokens,
                        "temperature": temperature
                    }
                )

                # 'output' is an iterator, so we join its parts to form the full response string.
                full_response = "".join(list(output))

                st.markdown("---")
                st.subheader("Generated Response:")
                st.markdown(full_response)

            except Exception as e:
                st.error(f"An error occurred with the Replicate API: {e}")