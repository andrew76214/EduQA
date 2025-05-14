import glob
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import csv

def check_rag_pipeline(pipeline):
    issues = []
    if not hasattr(pipeline, 'retriever'):
        issues.append("Pipeline is missing a retriever component.")
    if not hasattr(pipeline, 'generator'):
        issues.append("Pipeline is missing a generator component.")
    if hasattr(pipeline, 'retriever'):
        try:
            tq = "Test query"
            ret = pipeline.retriever.invoke(tq)
            results = ret.get("documents") if isinstance(ret, dict) else ret
            if not results:
                issues.append("Retriever returned no results for a test query.")
        except Exception as e:
            issues.append(f"Retriever raised an exception: {e}")
    if hasattr(pipeline, 'generator'):
        try:
            out = pipeline.generator.generate("Test input")
            if not out:
                issues.append("Generator returned no output for a test input.")
        except Exception as e:
            issues.append(f"Generator raised an exception: {e}")
    return {"status": "Issues found" if issues else "Pipeline is functional", "details": issues}

def embed_and_store_csvs_in_chroma(src_directory, chroma_db_directory):
    # Initialize Chroma vector store
    vectorstore = Chroma(
        persist_directory=chroma_db_directory,
        embedding_function=HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': False}
        ),
    )

    # Find all CSV files in the source directory
    csv_files = glob.glob(f"{src_directory}/**/*.csv", recursive=True)

    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        with open(csv_file, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Add all rows to ChromaDB without format restrictions
                combined_text = str(row)
                existing_results = vectorstore.similarity_search(combined_text, k=1)
                if existing_results and existing_results[0].page_content == combined_text:
                    continue
                vectorstore.add_texts(
                    texts=[combined_text],
                    metadatas=[row],
                    ids=[row.get("id", str(hash(str(row))))]  # Use 'id' if available, otherwise hash the row
                )

    print("All CSV files have been processed and stored in ChromaDB.")