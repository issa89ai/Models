#llm_functions
import os
import requests
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

HF_API_KEY = os.environ["HF_API_KEY"]
hf_headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Hugging Face public model endpoints
HF_SUMMARIZATION_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_QA_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

API_NINJAS_KEY = os.environ["API_NINJAS_KEY"]
API_NINJAS_URL = "https://api.api-ninjas.com/v1/textsimilarity"
api_ninjas_headers = {"X-Api-Key": API_NINJAS_KEY}


def split_into_chunks(text, max_length=3000):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def summarize_text(text, model_name):
    chunks = split_into_chunks(text)
    summaries = []

    for i, chunk in enumerate(chunks):
        if "huggingface" in model_name:
            summary = hf_summarize(chunk)
        else:
            summary = groq_summarize(chunk, model_name)

        summaries.append(f"🟪 Chunk {i+1} Summary:\n{summary}\n")

    return "\n".join(summaries)


def groq_summarize(text_chunk, model_name):
    prompt = f"Summarize the following text:\n\n{text_chunk}"
    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Groq Error]: {str(e)}"


def hf_summarize(text_chunk):
    payload = {"inputs": text_chunk}
    response = requests.post(HF_SUMMARIZATION_URL, headers=hf_headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0]['summary_text']
        else:
            return str(result)
    else:
        return f"[HF Summarization Error]: {response.status_code} - {response.text}"


def answer_question(text, question, model_name):
    text = text[:3000]
    if "huggingface" in model_name:
        return hf_answer_question(text, question)
    else:
        return groq_answer_question(text, question, model_name)


def groq_answer_question(text, question, model_name):
    prompt = f"Answer the question based on this text:\n\nText: {text}\n\nQuestion: {question}"
    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq Error: {str(e)}"


def hf_answer_question(text, question):
    payload = {
        "inputs": {
            "question": question,
            "context": text
        }
    }
    response = requests.post(HF_QA_URL, headers=hf_headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        if isinstance(result, dict) and 'answer' in result:
            return result['answer']
        else:
            return str(result)
    else:
        return f"[HF QA Error]: {response.status_code} - {response.text}"


def compute_full_similarity_matrix_with_original(original_text, summaries):
    combined_texts = {"Original PDF": original_text}
    combined_texts.update(summaries)

    keys = list(combined_texts.keys())
    texts = list(combined_texts.values())

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return pd.DataFrame(similarity_matrix, index=keys, columns=keys)


def compute_similarity_matrix_api_ninjas_with_original(original_text, summaries):
    combined_texts = {"Original PDF": original_text}
    combined_texts.update(summaries)

    models = list(combined_texts.keys())
    matrix = pd.DataFrame(index=models, columns=models)

    for i in models:
        for j in models:
            if i == j:
                matrix.loc[i, j] = 1.0
            else:
                matrix.loc[i, j] = get_similarity_api_ninjas(combined_texts[i], combined_texts[j])
    return matrix


def get_similarity_api_ninjas(text1, text2):
    payload = {'text_1': text1, 'text_2': text2}
    response = requests.post(API_NINJAS_URL, headers=api_ninjas_headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result.get('similarity', 0)
    else:
        print(f"API-Ninjas error: {response.status_code}")
        return 0
