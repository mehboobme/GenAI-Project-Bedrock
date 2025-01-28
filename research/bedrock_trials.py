from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import boto3
import streamlit as st

# Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-south-1"
)

model_id = "meta.llama3-8b-instruct-v1:0"

# Initialize Bedrock LLM
llm = Bedrock(
    model_id=model_id,
    client=bedrock_client,
    model_kwargs={"temperature": 0.9}
)

# Chatbot function
def my_chatbot(language, user_text):
    try:
        prompt = PromptTemplate(
            input_variables=['user_text'],  # Removed 'language' variable
            template="{user_text}"  # Focus only on the user's question
        )
        bedrock_chain = LLMChain(llm=llm, prompt=prompt)
        response = bedrock_chain({'user_text': user_text})
        return response
    except Exception as e:
        return {"error": str(e)}

# Streamlit UI
st.title("Bedrock Test")
language = st.sidebar.selectbox("Language", ["English", "Spanish", "Hindi"])

if language:
    user_text = st.sidebar.text_area(label="What is your question?", max_chars=100)
    if user_text:
        response = my_chatbot(language, user_text)
        if "error" in response:
            st.error(f"Error: {response['error']}")
        else:
            st.write(response.get('text', "No response received."))
