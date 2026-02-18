import streamlit as st
import replicate
import os

# --- Page Configuration ---
st.set_page_config(page_title="My DeepSeek Deployment", page_icon="🚀")

st.title("🚀 My Custom AI Deployment")
st.markdown("Using a personal Replicate deployment on Railway.")

# --- Authentication ---
# We need to explicitly pass the token when creating the client for deployments.
try:
    api_token = os.environ["RKEY"]
except KeyError:
    st.error("🚨 RKEY environment variable not found! Please set it in your Railway project variables.")
    st.stop()

# Create a Replicate client with your API token
client = replicate.Client(api_token=api_token)

# --- User Interface ---
with st.form("generation_form"):
    prompt = st.text_area(
        "Enter your prompt:", 
        height=150, 
        placeholder="Write a python script to list all files in a directory."
    )
    
    with st.expander("Advanced Settings"):
        temperature = st.slider("Temperature (Creativity)", 0.01, 2.0, 0.75, 0.05)
        max_tokens = st.number_input("Max New Tokens", min_value=64, max_value=4096, value=512)

    submit_button = st.form_submit_button("Generate Response")

# --- API Call Logic (for a Private Deployment) ---
if submit_button:
    if not prompt:
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("🚀 Sending request to your deployment... Please wait."):
            try:
                # IMPORTANT: This name must EXACTLY match your deployment name on Replicate.
                deployment_name = "constantinbender51-cmyk/deepseek-base"
                
                # 1. Get a client object for your specific deployment
                deployment = client.deployments.get(deployment_name)

                # 2. Create a prediction on that deployment
                prediction = deployment.predictions.create(
                    input={
                        "prompt": prompt,
                        "max_new_tokens": max_tokens,
                        "temperature": temperature
                    }
                )

                # 3. Wait for the prediction to complete (blocking)
                prediction.wait()

                # 4. Check the status and get the output
                if prediction.status == "succeeded":
                    # The output is often a list of strings, so we join them.
                    full_response = "".join(prediction.output)
                    
                    st.markdown("---")
                    st.subheader("Generated Response:")
                    st.markdown(full_response)
                else:
                    # If the prediction failed, show the error message
                    st.error(f"The prediction failed with status: {prediction.status}")
                    st.error(f"Error details: {prediction.error}")

            except Exception as e:
                # This will catch errors like the 404 if the deployment name is wrong
                st.error(f"An error occurred while communicating with the deployment: {e}")