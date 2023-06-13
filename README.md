# LangChain Doc Heler (LangChain Practice)

## Description
weba application that uses Pinecone as a vectorsotre and answers questions about the given documents using LangChain and OpenAi

## Requirements
create OpenAi api key, Pinecone access key 

## Run locally

download langchain documentation

```bash
  mkdir langchain-docs
  wget -r -A.html -P langchain-docs https://python.langchain.com/en/latest/index.html
```

intsall dependencies 

```bash
  pipenv install
```

Start the server

```bash
  streamlit run main.py
```

