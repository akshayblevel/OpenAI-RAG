# pip install azure-search-documents
import os
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY","")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://akkioai.openai.azure.com/")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "akkigpt-4")
API_VERSION = "2025-01-01-preview"

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://akkiragsearch.search.windows.net")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "ragindex")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def query_search_service(query_text: str, top_n: int = 5):
    """Query Azure Cognitive Search and return top documents."""
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX,
        credential=AzureKeyCredential(SEARCH_KEY),
    )
    results = search_client.search(
        search_text=query_text, top=top_n, include_total_count=True
    )
    return list(results)


def format_documents_for_prompt(documents: list) -> str:
    """Format search results into readable text for prompting."""
    formatted = []
    for i, doc in enumerate(documents, start=1):
        lines = [f"Document {i}:"]
        lines.extend(f"{k}: {v}" for k, v in doc.items() if k != "@search.score")
        formatted.append("\n".join(lines))
    return "\n\n".join(formatted)


def ask_question_with_rag(question: str) -> str:
    """Use RAG (search + OpenAI) to answer a question."""
    docs = query_search_service(question)
    if not docs:
        return "No relevant information found to answer your question."

    context = format_documents_for_prompt(docs)

    system_message = (
        "You are an assistant that answers questions based only on the provided documents.\n"
        "If the answer cannot be found in the documents, respond with:\n"
        "'I don't have information on that topic.'\n"
        "Be accurate, helpful, and concise."
    )

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error while generating answer: {e}"


def main():
    print("=== Azure OpenAI RAG Demo ===")
    question = "What are the projects Akshay Kumar worked on?"
    print(f"\nQuestion: {question}")
    print("\nRetrieving information and generating answer...")
    answer = ask_question_with_rag(question)
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()

'''
=== Azure OpenAI RAG Demo ===

Question: What are the projects Akshay Kumar worked on?

Retrieving information and generating answer...

Answer:
Akshay Kumar Patel worked on the following projects:

1. **SNE (Subaru of New England) – Enterprise Solution for Car Company**
   - Involved Inventory, Accessorization, Allocation, Authentication & Authorization, and Integration with SOA (Subaru of America) and Business Central using technologies such as Microservice Architecture, Dotnet 8, SQL 2016, Azure APIM, Entra, App Insights, Service Bus, Redis, Blob Storage with CDN, Key Vault, App Service, Function, Logic App, Elastic Cloud, DAPR Sidecar, gRPC, GraphQL.

2. **Tax Management**
   - Focused on capturing and maintaining tax-related information using office fabric controls, implemented with SharePoint SPFx using React.

3. **Document Management**
   - Managed and maintained document metadata providing services like document search using metadata from Apache SOLR, document comments, and document audits. Implemented with Asp.net Core 3.1, Apache SOLR.

4. **Hotel & Flight Booking Product**
   - A product for hotel and flight booking for the UK and IE Market, involving search, valuation, booking with payment gateways, and built with Asp.net Core, My SQL, Apache Kafka, Redis, ELK Stack.

5. **SAAS Based E-Commerce**
   - An E-commerce product providing services like platform APIs with cart, order, payment gateways, warehouses, logistics, using Asp.net 4.0 with C#, MySql, WCF, Log4Net, Quartz Scheduler, MSMQ.

6. **Scaled Utility Bill Payment System**
   - Implemented a real-time accounting system through double entry accounting.

7. **M-Pos (Merchant Point-of-Sale)**
   - Provided multiple services with a single platform including Topup Through Cyberplat & Euronet APIs, Voucher Recharge Through E-pin, International Topup Through Ezetop APIs, Bill.
'''