import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFLoader

# Ollama 언어 모델 서버의 기본 URL
CUSTOM_URL = "http://localhost:11434"


# 요약을 위한 Ollama 언어 모델 초기화
llm = Ollama(
    model="llama3", 
    base_url=CUSTOM_URL, 
    temperature=0,    
    num_predict=200
)

# PDF 파일을 읽고 처리하기 위한 함수
def read_file(file_name):
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(file_name.getbuffer())
        file_path = tf.name
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    return text_splitter.split_documents(documents)

# 문서 청크 리스트가 있으면 요약을 해주는 함수
def summarize_documents(txt_input):

    map_prompt_template = """
    - you are a professional translator
    - translate the provided content into English
    - only respond with the translation
    {text}
    """
    summary_result = ""
    message_placeholder = st.empty()
    
    for doc in txt_input:
        prompt_text = map_prompt_template.format(text=doc)
        stream_generator = llm.stream(prompt_text)
        
        for chunk in stream_generator:
            summary_result += chunk
            message_placeholder.markdown(summary_result)

# Streamlit 앱의 제목 구성
st.title(" 🦜 PDF을 번역해드려요")

def main():
    """
    Streamlit 앱을 실행하는 메인 함수.
    """
    if 'summary_result' not in st.session_state:
        st.session_state.summary_result = ""

    st.markdown("#### PDF 업로드 ▼ ")
    uploaded_file = st.file_uploader('pdfuploader', label_visibility="hidden", accept_multiple_files=False, type="pdf")
    
    if uploaded_file is not None:
        txt_input = read_file(uploaded_file)
        with st.spinner("문서를 번역하는 중..."):
            summarize_documents(txt_input)

if __name__ == "__main__":
    main()
