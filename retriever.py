import datasets
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever

guest_dataset = datasets.load_dataset("parquet", data_files="data.parquet", split = "train")

docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        meta_data={"name": guest["name"]}
    )
    for guest in guest_dataset 
]

# fictional_guest = Document(
#     page_content="\n".join([
#         "Name: Captain Nova Starfire",
#         "Relation: Intergalactic Ambassador",
#         "Description: A charismatic ambassador representing the United Federation of Planets. Known for diplomatic brilliance and cosmic fashion sense.",
#         "Email: nova.starfire@ufp.galaxy"
#     ]),
#     meta_data={"name": "Captain Nova Starfire"}
# )

# docs.append(fictional_guest)

bm25_retriever = BM25Retriever.from_documents(docs)

def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:1]])
    else:
        return "No matching guest information found."

