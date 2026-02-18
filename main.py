import streamlit as st
import replicate
import os

# Page configuration
st.set_page_config(page_title="DeepSeek Base AI", page_icon="🤖")

# Custom CSS to make it look cleaner
st.markdown("""
<style>
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 DeepSeek Base Generator")
st.markdown("Powered by Replicate & Railway")

# 1. Setup Authentication
# We retrieve the key from the environment variable 'RKEY' as requested
api_token = os.getenv("RKEY")

if not api_token:
    st.error("🚨 Error: RKEY environment variable not found. Please set it in Railway settings.")
    st.stop()

# Configure the replicate client with the custom token
client = replicate.Client(api_token=api_token)

# 2. User Interface Inputs
with st.form("generation_form"):
    prompt = st.text_area("Enter your prompt:", height=150, placeholder="Write a python script to...")
    
    # Optional parameters sidebar
    with st.expander("Advanced Settings"):
        temperature = st.slider("Temperature (Creativity)", 0.1, 2.0, 0.7)
        max_tokens = st.number_input("Max Tokens", 64, 4096, 512)

    submit_button = st.form_submit_button("Generate Response")

# 3. Logic to call the API
if submit_button and prompt:
    # Create a placeholder for the streaming output
    response_placeholder = st.empty()
    full_response = ""

    try:
        # Define the model input schema
        input_data = {
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": temperature
        }

        # Run the model
        # Note: We use the owner/model format. If a specific version hash is needed, 
        # Replicate usually handles the latest version automatically with this format.
        output = client.stream(
            "constantinbender51-cmyk/deepseek-base",
            input=input_data
        )

        # Stream the results to the UI
        for event in output:
            full_response += str(event)
            response_placeholder.markdown(full_response + "▌")
        
        # Final update without the cursor
        response_placeholder.markdown(full_response)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

elif submit_button and not prompt:
    st.warning("Please enter a prompt first.")
