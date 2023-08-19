import os
import requests
from bs4 import BeautifulSoup
import csv
from googlesearch import search
import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def get_news_articles(keyword):
    base_url = "https://news.google.com"
    search_url = f"{base_url}/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"

    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, "html.parser")

    article_links = []
    for article in soup.find_all("article"):
        link_element = article.find("a", {"class": "VDXfz"})
        if link_element:
            article_links.append(base_url + link_element["href"])

    return article_links

def get_user_inputs():
    search_query = input("Enter the search query: ")
    #num_results = int(input("Enter the number of results to retrieve: "))
    return search_query

def get_search_results(query, num_results=10):
    search_results = []

    for i, result in enumerate(search(query, num_results=num_results), start=1):
        search_results.append(result)
        if i >= num_results:
            break

    return search_results

def scrape_website_details(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        title = soup.title.string.strip() if soup.title else "No title"
        text_content = soup.get_text().replace("\n", " ")
        
        return title, text_content
    except Exception as e:
        print(f"An error occurred while scraping {url}: {e}")
        return None, None

def save_to_csv(results, csv_filename):
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Source/URL", "Title", "Content"])

        for url, title, content in results:
            csv_writer.writerow([url, title, content])

DB_FAISS_PATH = 'vectorstore/db_faiss'
DATA_PATH = 'csv_files/'  # Change this to the actual data directory

# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_1.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# The main Streamlit app
def main():
    st.title("Chat with Creazy.ai using Llama2 🦙🦜")
    st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)

    # Process all CSV files in the specified data directory
    csv_files = [filename for filename in os.listdir(DATA_PATH) if filename.endswith('.csv')]
    data = []
    for csv_file in csv_files:
        loader = CSVLoader(file_path=os.path.join(DATA_PATH, csv_file), encoding="utf-8", csv_args={
            'delimiter': ','
        })
        data.extend(loader.load())

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about current NEWS 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # Container for the chat history
    response_container = st.container()
    # Container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your CSV data here (:")
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            # Save the user's input as the search key for code 1
            search_key = user_input
            num_results = 3

            # Run code 1 with the provided search key
            create_folder_if_not_exists("csv_files")
            news_articles = get_news_articles(search_key)[:num_results]
            news_scraped_results = []

            for article_url in news_articles:
                response = requests.get(article_url)
                article_soup = BeautifulSoup(response.content, "html.parser")

                title = article_soup.find("h1").text.strip() if article_soup.find("h1") else "N/A"
                content = "\n".join([p.get_text() for p in article_soup.find_all("p")])

                news_scraped_results.append((title, article_url, content))

            news_csv_filename = os.path.join("csv_files", "news_web_data.csv")
            save_to_csv(news_scraped_results, news_csv_filename)
              # Code 1: Scraping website details
            web_search_results = get_search_results(search_key, num_results)
            web_scraped_results = []

            for url in web_search_results:
                print(f"Scraping website: {url}")
                title, content = scrape_website_details(url) 
                web_scraped_results.append((url, title, content))
            web_csv_filename = os.path.join("csv_files", "website_details.csv")
            save_to_csv(web_scraped_results, web_csv_filename)
            print(f"Website scraping completed. Details saved to '{web_csv_filename}'.")

            # Run code 2 with the generated CSV file
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Displaying the chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

if __name__ == "__main__":
    main()
