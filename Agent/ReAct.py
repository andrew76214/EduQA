import os, csv, re, glob, json
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict
from langchain_core.output_parsers import PydanticOutputParser

from SimplifiedChineseChecker import SimplifiedChineseChecker
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever

from LLM_prompt import prompt1, prompt2, prompt3, prompt4

class QAOutput(TypedDict):
    Question: str
    Answer: str

class CombinedProcessor:
    def __init__(self, input_csv, db_name, output_csv="output.csv"):
        # Retrieve API key from environment
        self.api_key = "Your OpenAI API Key"

        # Initialize LLM and embeddings
        self.llm = ChatOpenAI(
            model="o3",
            api_key=self.api_key
        )

        """self.llm = ChatOllama(
            model="deepseek-r1:70b",
        )"""

        """self.embeddings = OpenAIEmbeddings(
            # model="zylonai/multilingual-e5-large:latest",
            model="text-embedding-3-large",
            openai_api_key=self.api_key
        )"""

        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': False}
        )
        self.prompt1 = prompt1
        self.prompt2 = prompt2
        self.prompt3 = prompt3
        self.prompt4 = prompt4

        # Initialize Chroma vector store
        self.vectorstore = Chroma(
            persist_directory=db_name,
            embedding_function=self.embeddings
        )

        # Metadata for SelfQueryRetriever
        self.document_content_description = "Educational terms, definitions, and related contextual information, including scientific terms, idioms, world heritage sites, and their detailed descriptions."
        self.metadata_field_info = [
            {"name": "term", "description": "The term, idiom, or site being defined or described"},
            {"name": "definition", "description": "The definition, explanation, or detailed description of the term, idiom, or site"},
            {"name": "source", "description": "The source or category of the information, such as 'Life Sciences', 'Chemistry', 'Idioms', or 'World Heritage Sites'"},
            {"name": "context", "description": "Additional contextual information, such as historical background, geographical location, or cultural significance"}
        ]

        # Create SelfQueryRetriever
        self.retriever = SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vectorstore,
            document_contents=self.document_content_description,
            document_content_description=self.document_content_description,
            metadata_field_info=self.metadata_field_info,
            enable_limit=True,
            verbose=True
        )

        # Define pipeline steps
        def step1(inp):
            print('Step1 processing...')
            # Retrieve related sentences using self.retriever
            related_sentences = self.retriever.invoke(inp["question"])
            prompt_text = self.prompt1.format(
                original_question=inp["question"],
                original_answer=inp["original_answer"],
                related_sentences="\n".join([doc.page_content for doc in related_sentences])
            )
            result = self.llm.invoke(prompt_text)
            return {"original_question": inp["question"], "result": result}

        def step2(prev):
            print('Step2 processing...')
            # Retrieve related sentences using self.retriever
            related_sentences = self.retriever.invoke(prev["original_question"])
            prompt_text = self.prompt2.format(
                simplified_explanation=prev["result"],
                related_sentences="\n".join([doc.page_content for doc in related_sentences])
            )
            result = self.llm.invoke(prompt_text)
            return {"original_question": prev["original_question"], "result": result}

        def step3(prev):
            print('Step3 processing...')
            # Retrieve related sentences using self.retriever
            related_sentences = self.retriever.invoke(prev["original_question"])
            prompt_text = self.prompt3.format(
                context_sentence=prev["result"],
                related_sentences="\n".join([doc.page_content for doc in related_sentences])
            )
            result = self.llm.invoke(prompt_text)
            return {"original_question": prev["original_question"], "result": result}

        def step4(prev):
            print('Step4 processing...')
            # Retrieve related sentences using self.retriever
            related_sentences = self.retriever.invoke(prev["original_question"])
            if isinstance(related_sentences, list):
                sentences = [getattr(doc, "page_content", str(doc)) for doc in related_sentences]
            else:
                sentences = [str(related_sentences)]
            prompt_text = self.prompt4.format(
                question=prev["result"],
                related_sentences="\n".join(sentences)
            )
            result = self.llm.invoke(prompt_text)
            return {"original_question": prev["original_question"], "result": result}

        def step5_with_structured_output_and_map(prev):
            print('Step5 processing with structured output and mapping...')
            parser = PydanticOutputParser(pydantic_object=QAOutput)
            # Retrieve related sentences using self.retriever
            related_sentences = self.retriever.invoke(prev["original_question"])
            if isinstance(related_sentences, list):
                sentences = [getattr(doc, "page_content", str(doc)) for doc in related_sentences]
            else:
                sentences = [str(related_sentences)]
            prompt_text = self.prompt4.format(
                question=prev["result"],
                related_sentences="\n".join(sentences)
            )
            result = self.llm.invoke(prompt_text)
            content = result.content
            json_match = re.search(r"```json\s*([\s\S]+?)```", content)
            if json_match:
                try:
                    qa_json = json.loads(json_match.group(1))
                    q = qa_json.get("Q", prev["original_question"])
                    a = qa_json.get("A", "")
                except Exception:
                    q, a = prev["original_question"], json_match.group(1).strip()
            else:
                # fallback: Q/A 正則，只取最後一組
                matches = list(re.finditer(r"Q[：:](.*?)A[：:](.*?)(?=Q[：:]|$)", content, re.DOTALL))
                if matches:
                    last_match = matches[-1]
                    q = last_match.group(1).strip()
                    a = last_match.group(2).strip()
                else:
                    q = prev["original_question"]
                    a = content.strip()
        
            q = q.replace('\n', '').replace('\r', '')
            a = a.replace('\n', '').replace('\r', '')    
            print("Final question:", q)
            print("Final answer:", a)
            return {"messages": [{"role": "assistant", "content": {"Question": q, "Answer": a}}]}

        # Assemble pipeline
        self.chain = (
            RunnableLambda(step1)
            | RunnableLambda(step2)
            | RunnableLambda(step3)
            | RunnableLambda(step4)
            | RunnableLambda(step5_with_structured_output_and_map)
        )

        # Add generator component
        class _Generator:
            def __init__(self, llm):
                self.llm = llm
            def generate(self, prompt: str):
                return self.llm.invoke(prompt)
        self.generator = _Generator(self.llm)

        # CSV setup
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.csv_fields = ['Question', 'Answer', 'URL']
        if not os.path.exists(self.output_csv):
            with open(self.output_csv, mode='w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fields)
                writer.writeheader()

    def run_chain_pipeline(self, initial_input):
        try:
            return self.chain.invoke(initial_input)
        except Exception as e:
            print("執行 chain pipeline 時錯誤:", e)
            return None

    def process_csv_file(self):
        with open(self.input_csv, mode='r', encoding='utf-8-sig') as f_in, \
             open(self.output_csv, mode='a', encoding='utf-8-sig', newline='') as f_out:
            reader = csv.DictReader(f_in)
            writer = csv.DictWriter(f_out, fieldnames=self.csv_fields)
            for idx, row in enumerate(reader, 1):
                q = row.get("Question", "").strip()
                oa = row.get("Answer", "").strip()
                url = row.get("URL", "").strip()
                initial = {"question": q, "original_answer": oa}
                out = self.run_chain_pipeline(initial)
                content = out.get("messages")[0]["content"] if out and "messages" in out else {}
                writer.writerow({
                    "Question": content.get("Question", ""),
                    "Answer": content.get("Answer", ""),
                    "URL": url
                })
                if idx == 1:
                    break
        print("處理完成，結果存於", self.output_csv)


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

if __name__ == '__main__':
    src_directory = "./user_manual_files"
    chroma_db_directory = "./chroma_db_multilingual-e5-large_cleaned"
    # embed_and_store_csvs_in_chroma(src_directory, chroma_db_directory)

    output_file_name = "output.csv"
    proc = CombinedProcessor(input_csv="./QA_v1.0.csv", db_name=chroma_db_directory, output_csv=output_file_name)
    rag_status = check_rag_pipeline(proc)
    print("RAG Pipeline Check:", rag_status)
    proc.process_csv_file()
    SimplifiedChineseChecker(output_file_name).check_csv_for_simplified_chinese()