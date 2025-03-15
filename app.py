import streamlit as st
import cohere
import pandas as pd
import io
import re
import datetime

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "full_response" not in st.session_state:
    st.session_state.full_response = None
# New: store all results
if "results_history" not in st.session_state:
    st.session_state.results_history = pd.DataFrame(columns=['text', 'prediction', 'confidence', 'timestamp'])

@st.cache_data
def reset_chat():
    st.session_state.messages = []
    st.session_state.full_response = None
    st.session_state.results_history = pd.DataFrame(columns=[
        'text', 'prediction', 'confidence', 'timestamp'])
    st.rerun()

def process_file_content(file_input):
    """Extract text content from uploaded file."""
    try:
        # For chat_input files, we need to check differently
        if hasattr(file_input, 'name') and hasattr(file_input, 'getvalue'):
            file_name = file_input.name
            file_content = file_input.getvalue()

            # Try to determine file type from name
            if file_name.endswith('.txt'):
                # For text files, get raw text
                raw_text = file_content.decode('utf-8')
                # Split by semicolons and clean up
                sentences = [s.strip() for s in raw_text.split(';') if s.strip()]
                return sentences
            elif file_name.endswith('.csv'):
                try:
                    # For CSV files, try to read as DataFrame
                    df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
                    # Take first column as input text
                    sentences = df.iloc[:, 0].astype(str).tolist()
                    # Clean up
                    sentences = [s.strip() for s in sentences if s.strip() and s.lower() != 'nan']
                    return sentences
                except Exception as e:
                    st.error(f"Error processing CSV file: {str(e)}")
                    # Fall back to raw text split by commas
                    raw_text = file_content.decode('utf-8')
                    sentences = [s.strip() for s in raw_text.split(',') if s.strip()]
                    return sentences
            elif file_name.endswith(('.xls', '.xlsx')):
                try:
                    df = pd.read_excel(io.BytesIO(file_content))
                    sentences = df.iloc[:, 0].astype(str).tolist()
                    sentences = [s.strip() for s in sentences if s.strip() and s.lower() != 'nan']
                    return sentences
                except Exception as e:
                    st.error(f"Error processing Excel file: {str(e)}")
                    return None
            elif file_name.endswith('.pdf'):
                st.warning("PDF processing would require additional libraries like PyPDF2")
                return ["PDF content placeholder"]
            else:
                # Default to trying to decode as text
                try:
                    raw_text = file_content.decode('utf-8')
                    # First try semicolons
                    sentences = [s.strip() for s in raw_text.split(';') if s.strip()]
                    if len(sentences) <= 1:
                        # If no semicolons, try newlines
                        sentences = [s.strip() for s in raw_text.split('\n') if s.strip()]
                    return sentences
                except UnicodeDecodeError:
                    st.error(f"Unsupported file type or binary content: {file_name}")
                    return None
        else:
            st.error("Invalid file input format")
            return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    with st.expander("**Application Guide**"):
        st.info(
            """
            ### How to Use
            1. Supply your **Cohere API Key** - you may create one afresh
            2. Optionally specify the **Model ID** if you want to use a different model than the default
            3. Use the chat input method:
            - 3.1 Single text: Type a statement or question in the chat input
            - 3.2 Multiple texts: Type multiple statements separated by semicolons
            - 3.3 File upload: Upload a text file (.txt), CSV, or Excel file containing text to classify
            _Up to 70 entries can be processed at once_
            4. Results summary are below with the classification results
            - 4.1 Latest Results: View detailed results of your most recent classification in the sidebar
            - 4.2 All Results History: Browse all classification results since your last reset in the sidebar
            - 4.3 All results include timestamps showing when each classification was performed
            _Results are available for downloads
            5. Reset by clicking **"Clear ↺"** to reset the chat history and all accumulated results
            """
        )
    st.markdown("#")
    api_key = st.text_input("Enter your Cohere API Key", type="password")
    st.link_button("Get one @ Cohere 🔗", "https://dashboard.cohere.com/api-keys")

    model_id = st.text_input("Model ID", value=st.secrets.get("MODEL_ID", ""), type="password")

    if not api_key:
        st.warning("Please enter a valid COHERE API-KEY")

    st.markdown("### All Results History")
    if not st.session_state.results_history.empty:
        # Format the timestamp for better display
        display_df = st.session_state.results_history.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(display_df, use_container_width=True)

    # with st.toggle("### Latest Results"):
    if st.toggle("### Latest Results"):
        if st.session_state.full_response is not None:
            st.dataframe(st.session_state.full_response, use_container_width=True)

    st.markdown("#")

    # Debug section
    if st.toggle("### Review"):
        with st.expander("Debug Info", expanded=False):
            st.text("Input and processing information will appear here")

# Main Page
st.title("Cohere Text Classifier Application")
st.write(
    """
    This is a Streamlit application that uses Cohere's AI classification model to analyze and categorize text. 
    The app can process individual text inputs or batch process multiple texts from files.
    """
)

with st.expander("### ⚠️ Disclaimer"):
    st.write('''
            This fine-tuned model is evaluated against a set of metrics that make up its overall performance. 
            After training, these metrics are compared to the metrics that would otherwise achieve from using Cohere's Default model that is not based off the training data.
            
            Do use with caution and rely on additional context where presented with ambiguous results.
        ''')

st.markdown("#")

st.markdown("### Conversation History")
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input (text or file)
if api_key:
    co = cohere.Client(api_key)

    user_input = st.chat_input("Enter text or upload a file for classification", accept_file=True)

    if user_input is not None:
        # Debug info
        with st.sidebar.expander("Debug Info", expanded=False):
            st.text(f"Input type: {type(user_input)}")
            st.text(f"Input value: {user_input}")

        # Extract text and files from ChatInputValue
        if hasattr(user_input, 'text'):
            # This is a ChatInputValue object
            text_input = user_input.text
            files = getattr(user_input, 'files', [])

            if files and len(files) > 0:
                # Process the first file
                sentences = process_file_content(files[0])
                if sentences and len(sentences) > 0:
                    input_text = f"File: {files[0].name}\n\n{sentences[0][:100]}..." if len(sentences[0]) > 100 else sentences[0]
                    if len(sentences) > 1:
                        input_text += f"\n... and {len(sentences)-1} more entries"
                    input_type = "file"
                    classification_items = sentences
                else:
                    st.error("Failed to process file or no valid content found")
                    input_text = None
                    input_type = None
                    classification_items = None
            elif text_input.strip():
                # Process text input - split by semicolons if present
                if ';' in text_input:
                    sentences = [s.strip() for s in text_input.split(';') if s.strip()]
                else:
                    sentences = [text_input.strip()]
                input_text = text_input
                input_type = "text"
                classification_items = sentences
            else:
                st.error("No text or file content provided")
                input_text = None
                input_type = None
                classification_items = None
        else:
            # Direct text input (string)
            input_text = str(user_input)
            input_type = "text"
            # Split by semicolons if present
            if ';' in input_text:
                classification_items = [s.strip() for s in input_text.split(';') if s.strip()]
            else:
                classification_items = [input_text.strip()]

        if input_text and classification_items:
            # Add user message to chat
            display_text = f"Input Type: {input_type}\n{input_text[:500]}..." if len(input_text) > 500 else input_text
            st.session_state.messages.append({"role": "user", "content": display_text})

            with st.chat_message("user"):
                st.markdown(display_text)

            # Process with Cohere
            with st.chat_message("assistant"):
                with st.spinner("Classifying..."):
                    try:
                        # Limit to prevent API overload
                        if len(classification_items) > 50:
                            st.warning(f"Processing only the first 50 of {len(classification_items)} entries")
                            classification_items = classification_items[:50]

                        # Debug info
                        with st.sidebar.expander("Debug Info", expanded=False):
                            st.text(f"Processing {len(classification_items)} items")
                            st.text(f"First item: {classification_items[0]}")

                        # Make sure inputs are properly formatted as strings
                        string_inputs = [str(item) for item in classification_items]

                        response = co.classify(
                            inputs=string_inputs,
                            model=model_id
                        )

                        # Get current timestamp
                        current_time = datetime.datetime.now()

                        # Create results dataframe with full original text and timestamp
                        batch_results = [
                            {
                                'text': item.input,
                                'prediction': item.prediction,
                                'confidence': round(item.confidence, 4),
                                'timestamp': current_time
                            }
                            for item in response.classifications
                        ]

                        st.session_state.full_response = pd.DataFrame(batch_results)

                        # Add to historical results
                        new_results_df = pd.DataFrame(batch_results)
                        st.session_state.results_history = pd.concat([st.session_state.results_history, new_results_df],
                                                                     ignore_index=True)

                        # Display summary response
                        if len(batch_results) == 1:
                            parsed_response = f"""
                            **Classification Results**:
                            - **Text**: {batch_results[0]['text'][:100]}{"..." if len(batch_results[0]['text']) > 100 else ""}
                            - **Prediction**: {batch_results[0]['prediction']}
                            - **Confidence**: {batch_results[0]['confidence']}
                            """
                        else:
                            predictions = {}
                            for result in batch_results:
                                pred = result['prediction']
                                predictions[pred] = predictions.get(pred, 0) + 1

                            summary = ", ".join([f"{k}: {v}" for k, v in predictions.items()])
                            parsed_response = f"""
                            **Batch Classification Results**:
                            - **Total Items**: {len(batch_results)}
                            - **Summary**: {summary}
                            - **Timestamp**: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
                            
                            *See sidebar for full results table and history*
                            """

                        st.markdown(parsed_response)
                        st.session_state.messages.append({"role": "assistant", "content": parsed_response})

                    except Exception as e:
                        error_msg = f"Error during classification: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})


# Clear chat button
if st.session_state.messages:
    st.button("Clear ↺", on_click=reset_chat)
