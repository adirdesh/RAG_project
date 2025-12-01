import os
import shutil
import zipfile
import base64
import numpy as np
import faiss
import torch
import boto3
from pathlib import Path
from typing import List, Callable
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from better_profanity import profanity
from dotenv import load_dotenv
from groq import Groq
from transformers import BertModel, BertTokenizer
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.text import partition_text
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# AWS and Groq setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
s3 = boto3.client("s3")
groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allow all. For prod, specify your frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
profanity.load_censor_words()

# Embedding model
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Global variables
docs = []
doc_keys = []
doc_ids = []
faiss_index = None

def encode_documents(texts: list[str]) -> np.ndarray:
    inputs = bert_tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

def set_up_faiss_index(documents: list[str], encode_fn: Callable) -> faiss.Index:
    embeddings = encode_fn(documents).astype(np.float32)
    embeddings = np.atleast_2d(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    if embeddings.shape[0] > 0 and embeddings.shape[1] == dim:
        if isinstance(index, faiss.IndexIVFFlat) and not index.is_trained:
            index.train(embeddings)
        index.add(embeddings)
    return index

def load_documents_from_s3(prefix: str) -> list[tuple[str, str, str]]:
    documents = []
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith(".txt"):
                file_obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
                content = file_obj['Body'].read().decode("utf-8")
                doc_id = Path(key).parts[-3]  # folder name (original doc name)
                documents.append((doc_id, key, content))
    except Exception as e:
        print(f"S3 Load Error: {e}")
    return documents

def reload_index_s3(prefix: str):
    global docs, doc_keys, doc_ids, faiss_index
    doc_triples = load_documents_from_s3(prefix)
    docs = [content for _, _, content in doc_triples]
    doc_keys = [key for _, key, _ in doc_triples]
    doc_ids = [doc_id for doc_id, _, _ in doc_triples]
    faiss_index = set_up_faiss_index(docs, encode_documents)

def retrieve_similar_documents(
    query: str,
    index: faiss.Index,
    documents: list[str],
    encode_fn: Callable,
    k: int,
    threshold: float = 90.0
) -> list[dict]:
    if index is None or not documents:
        return []
    emb = encode_fn([query]).astype(np.float32)
    emb = np.atleast_2d(emb)
    k = min(max(1, k), len(documents))
    if emb.size == 0 or k < 1:
        return []
    distances, indices = index.search(emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(documents) and dist <= threshold:
            results.append({
                "content": documents[idx],
                "score": float(dist),
                "s3_key": doc_keys[idx],
                "doc_id": doc_ids[idx]
            })
    return results

def format_prompt(query: str, context: str) -> str:
    return f"""Use the following pieces of context to answer the question at the end.\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""

def generate_answer(query: str, docs_with_scores: list[tuple[str, float]]) -> str:
    context = "\n".join(doc for doc, _ in docs_with_scores)
    prompt = format_prompt(query, context)
    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300,
            top_p=1
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            return "No answer generated."
    except Exception as e:
        return f"Error generating response: {str(e)}"

class Query(BaseModel):
    query: str

class UploadRequest(BaseModel):
    file_path: str
    base_output_dir: str
    org: str
    dept: str
    debug_mode: bool = False

def upload_to_s3(key: str, data: bytes, content_type: str):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)
    print(f"[Upload] {key} ({content_type}) - {len(data)} bytes")

@app.post("/upload_docs")
def process_file(data: UploadRequest) -> List[str]:
    def extract_images_from_zip(file_path: str, zip_prefix: str, output_dir: str) -> List[str]:
        images = []
        try:
            with zipfile.ZipFile(file_path, 'r') as z:
                media_files = [f for f in z.namelist() if f.startswith(zip_prefix)]
                os.makedirs(output_dir, exist_ok=True)
                for i, media_file in enumerate(media_files):
                    data = z.read(media_file)
                    ext = Path(media_file).suffix or ".png"
                    path = os.path.join(output_dir, f"image_{i}{ext}")
                    with open(path, "wb") as f:
                        f.write(data)
                    images.append(path)
        except Exception as e:
            print(f"[Zip Extraction Error] {e}")
        return images

    file_path = data.file_path
    ext = Path(file_path).suffix.lower()
    file_stem = Path(file_path).stem
    s3_prefix = f"{data.base_output_dir}/{data.org}/{data.dept}/{file_stem}/"
    text_chunks = []

    image_paths = extract_images_from_zip(file_path, "word/media/" if ext == ".docx" else "ppt/media/", "/tmp") if ext in [".docx", ".pptx"] else []

    # Upload images extracted from zip (for DOCX or PPTX)
    for img_path in image_paths:
        try:
            with open(img_path, "rb") as f:
                upload_to_s3(s3_prefix + "images/" + Path(img_path).name, f.read(), "image/png")
        except Exception as e:
            if data.debug_mode:
                print(f"[Image Upload Error] {img_path}: {e}")

    try:
        if ext == ".docx":
            elements = partition_docx(filename=file_path, infer_table_structure=True)
        elif ext == ".pptx":
            elements = partition_pptx(filename=file_path)
        elif ext == ".txt":
            elements = partition_text(filename=file_path)
        elif ext == ".pdf":
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                extract_images=True
            )

            # Upload PDF-extracted images
            for i, el in enumerate(elements):
                image_path = getattr(el.metadata, "image_path", None)
                if image_path:
                    if os.path.exists(image_path):
                        try:
                            with open(image_path, "rb") as f:
                                image_data = f.read()
                                s3_key = s3_prefix + f"images/pdf_image_{i}.png"
                                upload_to_s3(s3_key, image_data, "image/png")
                        except Exception as img_err:
                            if data.debug_mode:
                                print(f"Image upload error for {image_path}: {img_err}")
                    else:
                        if data.debug_mode:
                            print(f"[Warning] Image path not found on disk: {image_path}")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        if data.debug_mode:
            print(f"[Partitioning Error] {e}")
        return []

    # Upload a .keep file if no images were uploaded
    if not image_paths and ext in [".pdf", ".docx", ".pptx"]:
        try:
            dummy_key = s3_prefix + "images/.keep"
            s3.put_object(Bucket=S3_BUCKET, Key=dummy_key, Body=b"", ContentType="text/plain")
            if data.debug_mode:
                print(f"[Info] Created placeholder image folder: {dummy_key}")
        except Exception as e:
            if data.debug_mode:
                print(f"[Warning] Could not create placeholder folder: {e}")

    # Upload text chunks
    for i, el in enumerate(elements):
        if el.text and el.text.strip():
            try:
                text = el.text.encode("utf-8")
                key = s3_prefix + f"text/{file_stem}{i}.txt"
                upload_to_s3(key, text, "text/plain")
                text_chunks.append(key)
            except Exception as text_err:
                if data.debug_mode:
                    print(f"[Text Upload Error] {text_err}")

    return text_chunks

@app.post("/Answer")
def answer_question(data: Query):
    if faiss_index is None:
        return {"result": "No Relevant Document Found (index not built)"}
    retrieved = retrieve_similar_documents(data.query, faiss_index, docs, encode_documents, k=5)
    if not retrieved:
        return {"result": "No Relevant Document Found"}

    answer = generate_answer(data.query, [(r["content"], r["score"]) for r in retrieved])
    full_docs = list({r["doc_id"] for r in retrieved})  # Unique original doc names

    return {
        "result": answer,
        "source_documents": full_docs
    }

@app.get("/get_source_document/{doc_id}")
def get_source_document(doc_id: str, org: str, dept: str, base_output_dir: str = "test-output"):
    prefix = f"{base_output_dir}/{org}/{dept}/{doc_id}/"
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        if "Contents" not in response:
            raise HTTPException(status_code=404, detail="Document not found in S3")

        text_files = []
        image_files = []

        for obj in response["Contents"]:
            key = obj["Key"]

            if key.endswith(".txt"):
                file_obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
                content = file_obj['Body'].read().decode("utf-8")
                text_files.append({
                    "key": key,
                    "content": content
                })

            elif key.lower().endswith((".png", ".jpg", ".jpeg")):
                file_obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
                image_data = file_obj['Body'].read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")
                image_files.append({
                    "key": key,
                    "base64": image_base64
                })

        return {
            "doc_id": doc_id,
            "s3_prefix": prefix,
            "text_chunks": text_files,
            "images": image_files
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 access failed: {str(e)}")

#test block for local running
if __name__ == "__main__":
    test_request = UploadRequest(
        file_path="C:\\Users\\KIIT\\Desktop\\Indian_Foods.docx",
        base_output_dir="test-output",
        org="demo-org",
        dept="demo-dept",
        debug_mode=True
    )

    #uploaded_keys = process_file(test_request)
    #print("Uploaded keys:", uploaded_keys)

    # Rebuild the FAISS index
    s3_prefix = f"{test_request.base_output_dir}/{test_request.org}/{test_request.dept}"
    reload_index_s3(s3_prefix)

    # Perform a query
    query = "What does Staple ingredients in Indian cooking include ?"
    if profanity.contains_profanity(query):
        print("Profanity in query.")
    else:
        if faiss_index is None:
            print("No Relevant Document Found (index not built).")
        else:
            retrieved = retrieve_similar_documents(query, faiss_index, docs, encode_documents, k=5)
            if not retrieved:
                print("No Relevant Document Found.")
            else:
                answer = generate_answer(query, [(r["content"], r["score"]) for r in retrieved])
                full_docs = list({r["doc_id"] for r in retrieved})
                print("Answer:", answer)
                print("Source Document IDs:", full_docs)

                # === Simulate /get_source_document/{doc_id} ===
                for doc_id in full_docs:
                    print(f"\n--- Retrieving Full Document: {doc_id} ---")
                    prefix = f"{test_request.base_output_dir}/{test_request.org}/{test_request.dept}/{doc_id}/"
                    try:
                        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
                        if "Contents" not in response:
                            print("No files found for document:", doc_id)
                            continue

                        text_files = []
                        image_files = []

                        for obj in response["Contents"]:
                            key = obj["Key"]

                            if key.endswith(".txt"):
                                file_obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
                                content = file_obj['Body'].read().decode("utf-8")
                                text_files.append((key, content))

                            elif key.lower().endswith((".png", ".jpg", ".jpeg")):
                                file_obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
                                image_data = file_obj['Body'].read()
                                print(f"Image file: {key} (Size: {len(image_data)} bytes)")
                                image_files.append(key)

                        # Print text contents
                        for key, content in text_files:
                            print(f"\nText File: {key}\nContent:\n{content}\n")

                        if not image_files:
                            print("No images found.")
                    except Exception as e:
                        print(f"Error retrieving {doc_id}: {e}")