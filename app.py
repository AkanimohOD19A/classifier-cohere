import streamlit as st
import cohere
# import pandas as pd

API_KEY = st.secrets["COHERE_API_KEY"]
MODEL_ID = st.secrets["MODEL_ID"]


# co = cohere.Client(API_KEY)

@staticmethod
def reset_chat():
    st.session_state.messages = []
    st.rerun()


# Initialize messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "full_response" not in st.session_state:
    st.session_state.response = []

# Sidebar
st_side = st.sidebar
with st_side:
    api_key = st.text_input("Enter your Cohere API Key", type="password")
    st.link_button("Get one @ Cohere 🔗", "https://dashboard.cohere.com/api-keys")
    if not api_key:
        st.warning("Please enter a valid COHERE API-KEY")

    st.markdown("")
    st.markdown("")

# Main Page
### Uploader and
#


## Chat UI
st.markdown("### Conversation History")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# Converter
if api_key:
    co = cohere.Client(api_key)
    # Handle user input
    if prompt := st.chat_input("What do you think about this?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        # Simulate assistant response (replace with actual response logic)
        # response = f"Assistant response to: {prompt}"
        with st.chat_message("assistant"):
            response = co.classify(
                inputs=[prompt],
                model=MODEL_ID + "-ft"
            )

            parsed_response = f"""
            Classification: {response.classifications[0].input}
            Text: {response.classifications[0].input}
            Prediction: {response.classifications[0].prediction}
            Confidence: {response.classifications[0].confidence}
            """
            st.text(parsed_response)
            st.session_state.messages.append({"role": "assistant", "content": parsed_response})
            # st.write_stream(response)

            batch_results = [
                {
                    'text': item.input,
                    'prediction': item.prediction,
                    'confidence': item.confidence
                }
                for item in response.classifications
            ]

            # output_df = pd.DataFrame(batch_results)
            # print(output_df)
            # st.sidebar.dataframe(batch_results)


            # if st.sidebar.toggle("Full Response"):
            #     st.markdown(response)


# Clear chat button
if st.session_state.messages:
    st.button("Clear ↺", on_click=reset_chat)
