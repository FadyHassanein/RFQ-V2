import os
import base64
import gc
import tempfile
import uuid

#from IPython.display import Markdown, display
import streamlit as st
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_cloud_services import LlamaParse

# from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.agent.openai import OpenAIAgent




#openai_api_key = st.secrets["api_keys"]["OPENAI_API_KEY"]
#llama_cloud_api_key = st.secrets["api_keys"]["LLAMA_CLOUD_API_KEY"]
#tavily_search_tool_key = st.secrets["api_keys"]["TAVILY_API_KEY"]
openai_api_key = os.environ.get("OPENAI_API_KEY")
llama_cloud_api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
tavily_search_tool_key = os.environ.get("TAVILY_API_KEY")
if not tavily_search_tool_key:
    st.error("❌ Tavily API key is missing. Please set it in Streamlit secrets.")
    st.stop()

from typing import List, Dict

def search_online(query: str) -> List[Dict[str, str]]:
    """
    Useful tool to search for information online.
    
    Parameters:
        query (str): The search query to be used.
    
    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing 'text' and 'source' keys.
    """
    documents = tavily_search_tool.search(query)
    extracted_data = []
    
    for doc in documents:
        url = doc.metadata.get('url', 'Unknown source')
        text = doc.text_resource.text if doc.text_resource and doc.text_resource.text else 'No text available'
        extracted_data.append({"text": text, "source": url})
    
    return extracted_data

search_online_tool= FunctionTool.from_defaults(
    fn= search_online,
)

parser = LlamaParse(
    result_type="markdown",  # "markdown" and "text" are available,
    continuous_mode= True,
    api_key= llama_cloud_api_key,
)


# use SimpleDirectoryReader to parse our file
file_extractor = {".pdf": parser}


import streamlit as st

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

prompt_model="""# Expert AI Response Prompt

## Overview
This prompt is designed to guide an AI model to provide professional, structured responses to user queries with consistent formatting, comprehensive information extraction, and appropriate elaboration when needed.

## Core Principles
1. **Structured Information Extraction**: Extract and organize all relevant information clearly
2. **Complete Response**: Ensure no important details are omitted
3. **Professional Formatting**: Present information in organized lists, tables, or paragraphs as appropriate
4. **Proactive Assistance**: Anticipate missing information and provide it when possible
5. **Adaptability**: Adjust responses based on follow-up queries
6. **Accurate Source Referencing**: Cite information sources when required

## Response Framework
When responding to user queries, follow this structure:

### Initial Acknowledgment
Begin with "Responding to:" followed by a brief summary of the user's query to confirm understanding.

### Information Organization
Present extracted or requested information in a clearly structured format using:
- Bulleted or numbered lists for multiple items
- Hierarchical organization with main points and sub-points
- Tables for comparative information
- Paragraph form only when narrative is more appropriate

### Formatting Guidelines
- Use consistent formatting throughout
- Include all relevant numeric data (quantities, model numbers, dates, etc.)
- Group related information logically
- Apply hierarchical organization (headers, subheaders)
- Bold key information when appropriate

### Additional Guidance
- If the user points out missing information, acknowledge the oversight and provide the complete information
- When asked to provide sources, include URLs and pricing information when available
- Add relevant context or explanations for technical terms
- Maintain a professional, concise tone throughout

## Example Response Structure

Responding to: [Brief summary of query]

[Main information category]:
1. [Item/Point 1]
   o [Detail 1]
   o [Detail 2]
   o [Detail 3]
2. [Item/Point 2]
   o [Detail 1]
   o [Detail 2]

Additional Information:
- [Relevant context or explanations]
- [Important dates or requirements]

## Specific Response Types

### Information Extraction Tasks
For extracting information from documents:
- Identify and list all key data points
- Organize by categories (products, specifications, requirements)
- Include all numeric identifiers and codes
- Add metadata (dates, deadlines, contact information)

### Source/Reference Requests
When providing references or sources:
- Include direct URLs when available
- List pricing information if requested
- Note any limitations about the sources
- Provide brief descriptions of each source

## Using Search Tool
When the user asks you to search for information or when you need external data:
1. **Identify Search Requests**: Recognize when the user is asking for information that requires searching the web
   - Direct requests like "search for...", "find information about...", "what's the latest on..."
   - Questions about current events, statistics, or facts not likely to be in the document
   - Requests for comparative information from external sources

2. **Search Tool Usage**:
   - Use the `search_online_tool` tool to find relevant information
   - Construct concise, specific search queries focusing on key terms
   - When needed, perform multiple searches to gather comprehensive information

3. **Search Results Processing**:
   - Synthesize information from search results
   - Format findings according to the response framework
   - Clearly indicate when information comes from external searches vs. document content
   - Include relevant URLs or sources from search results

4. **Citation Requirements**:
   - The search tool will return text results along with their source references
   - You MUST cite ALL sources used in your response
   - Include citation numbers [1], [2], etc. within your response when referencing information
   - At the end of your response, include a "References" section listing all sources in order
   - Format references as: [#] URL - Brief description of source
   - Example: [1] https://example.com - Article on Egyptian archaeology from National Geographic

5. **Example Search Scenarios**:
   - User asks: "Search for recent developments in Egyptian archaeology"
   - User requests: "Find information about tourism statistics in Egypt"
   - User inquires: "What are the latest preservation techniques for ancient artifacts?"

Below is the scraped content of a PDF file.
<details>\n\n{context_str}\n\n</details>\n\n
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
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):

                    if os.path.exists(temp_dir):
                        documents = SimpleDirectoryReader(input_dir= temp_dir, file_extractor=file_extractor, required_exts=[".pdf"], recursive= True).load_data()

                            # loader = SimpleDirectoryReader(
                            #     input_dir = temp_dir,
                            #     required_exts=[".pdf"],
                            #     recursive=True
                            # )
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    # docs = loader.load_data()

                    # setup llm & embedding model
                    llm = OpenAI(model="gpt-4o", temperature=0.1, api_key=openai_api_key)
                    ## Define the Agent
                    prompt_model= prompt_model.format(context_str= documents)
                    # rag_agent= FunctionAgent(
                    #     llm= llm,
                    #     tool= [tavily_search_tool.to_tool_list()],
                    #     system_prompt= prompt_model,
                    #     streaming_response= True,
                    # )
                    # Creating an index over loaded data
                    # index = VectorStoreIndex.from_documents(documents=documents, show_progress=True)

                    Settings.llm = llm
                    # query_engine = index.as_query_engine(streaming=True)

                    rag_agent= OpenAIAgent.from_tools(
                        llm= llm,
                        tools= [search_online_tool],
                        system_prompt= prompt_model,
                        streaming_response= True,
                    )

                    # ====== Customise prompt template ======
                    # qa_prompt_tmpl_str = prompt_model
                    # qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    # query_engine.update_prompts(
                    #     {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    # )
                    
                    # st.session_state.file_cache[file_key] = query_engine
                    st.session_state.file_cache[file_key] = rag_agent
                else:
                    # query_engine = st.session_state.file_cache[file_key]
                    rag_agent= st.session_state.file_cache[file_key]

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
        # streaming_response = query_engine.query(prompt)
        streaming_response= rag_agent.chat(prompt)
        
        for chunk in streaming_response.response:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        # full_response = query_engine.query(prompt)

        message_placeholder.markdown(full_response)
        # st.session_state.context = ctx

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
