import os
import base64
import gc
import tempfile
import uuid
import numpy as np  # Added numpy import

# Debug: Print numpy version to help diagnose incompatibility issues
print("Numpy version:", np.__version__)
# Optionally, add a runtime check/informative message within the app
import streamlit as st
if np.__version__ < "1.23":
    st.warning("It is recommended to upgrade numpy (pip install --upgrade numpy) to avoid binary incompatibility issues.")

# from IPython.display import Markdown, display

# from llama_index.core import Settings
# from llama_index.llms.openai import OpenAI
# from llama_index.core import SimpleDirectoryReader
# from llama_cloud_services import LlamaParse


from openai import OpenAI
from langchain_pymupdf4llm import PyMuPDF4LLMLoader


openai_api_key= st.secrets["api_keys"]["OPENAI_API_KEY"]

openai_client= OpenAI(api_key=openai_api_key)


if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None



new_prompt="""# Information Processing & Response Guide

## Core Instructions
- Extract information from {context_str} ONLY - no hallucination
- Structure responses clearly with appropriate formatting
- Use web_search_preview when context is insufficient or when requested

## Extract & Process
1. Analyze context thoroughly
2. Identify key elements:
   - Products, specs, quantities, IDs, codes
   - Dates, deadlines, contact info
   - Requirements, relationships

3. Maintain precision:
   - Exact numbers/measurements
   - Complete identifiers
   - Technical terminology

## Format Responses
- Use lists, hierarchies, bold text as appropriate
- For item specifications:
  - Main item name first
  - Attributes as sub-points
  - Include all identifiers/quantities
  - Preserve original units

## Web Search Guidelines
When needed:
- Use `web_search_preview` tool
- Create focused queries with key terms
- Include manufacturer, model numbers for products
- Present results with URLs, prices when available
- Cite sources properly
- Avoid valuetronics website links

## Response Types

### Information Extraction Format
```
[Document Type/Source] from [Entity]

Items/Elements:
1. [Item/Category Name]
   o Qty/Amount: [Quantity] [Unit]
   o Description: [Detailed description]
   o [Attribute Label]: [Attribute Value]
   o [Identifier Type]: [Identifier Value]
   o [Additional Specification]: [Value]
   o [NAICS Number]: [Value]
   o [Other Identifiers]: [Value]


2. [Item Name]
   o [Similar structure]

Additional Requirements/Information:
• [Requirement Type]: [Requirement Details]
• [Deadline Type]: [Deadline Details]
• [Other Important Information]
```

### For Term Explanations
```
[Term] is [definition]. In this context, it [usage].
```

### Web Search Format
```
Here are [number] sources where you can find information about [search topic]:

[Source Name]: [URL]
o [Key Information Point]: [Details]
o [Price/Value if applicable]: [Price]

[Source Name]: [URL]
o [Key Information Point]: [Details]
o [Price/Value if applicable]: [Price]

[Additional context or caveats about the information]

References:
[1] [URL] - [Brief description of source]
[2] [URL] - [Brief description of source]
```
<Important>For your search do not include link from: valuetronics 

<user_query>{user_question}</user_query>
Your response:
"""



def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def display_pdf(file):
    # Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:
    st.header(f"Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write(f"File uploaded: {uploaded_file.name}")
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):

                    if os.path.exists(temp_dir):
                        st.write(f"./{uploaded_file.name}")
                        loader = PyMuPDF4LLMLoader(file_path=f"./{uploaded_file.name}")
                        documents= loader.load()
                        st.session_state.documents = documents[0].page_content
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    st.session_state.file_cache[file_key] = openai_client
                else:
                    openai_client= st.session_state.file_cache[file_key]

                # Inform the user that the file is processed and Display the PDF uploaded
                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with Your Documents and search online!")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response with milliseconds delay
        print("prompt:\n", prompt)
        print("context:\n", st.session_state.documents)
        prompt_model= new_prompt.format(
            context_str= st.session_state.documents,
            user_question= prompt,
        )
        streaming_response= openai_client.responses.create(
            model="gpt-4o",
            tools=[{'type': 'web_search_preview'}],
            input= prompt_model,
            )
        
        for chunk in streaming_response.output_text:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})