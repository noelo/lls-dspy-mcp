# oc port-forward -n skills svc/llama-stack-service 8321

import uuid
from llama_stack_client import RAGDocument, LlamaStackClient, types
from termcolor import cprint
import sys
import requests
from io import BytesIO
import dspy
from dspy.utils.mcp import convert_input_schema_to_tool_args
import asyncio
from llama_stack_client import AsyncLlamaStackClient


def dump_api(client):
    models = client.models.list()
    for x in models:
        print(f"{x}\n")
    print("*" * 60)

    toolg = client.toolgroups.list()
    for t in toolg:
        print(f"Tool Groups {t}\n")

    print(":" * 60)

    toolr = client.tool_runtime.list_tools()
    for t in toolr:
        print(f"Tool runtimes {t}\n")
    print("-" * 60)

    print("*" * 60)
    providers = client.providers.list()
    for x in providers:
        print(f"{x}\n")


def create_vdb(client):
    # This tells Llama Stack how to connect to and use your vector database
    vs = client.vector_stores.create(
        name="my_citations_db",
        extra_body={
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "provider_id": "milvus",
            "vector_db_id": "test",
        },
    )
    print(f"📊 Created vector database with ID: {vs.id}")
    return vs


def add_docs_vdb(client, vs):
    # List of URLs (in this case just 1) to process
    urls = [
        "https://raw.githubusercontent.com/rhoai-genaiops/deploy-lab/main/university-data/canopy-in-botany.pdf",
    ]

    # Display what we're about to ingest
    print("📖 Ingesting documents into RAG system...")
    for i, url in enumerate(urls):
        print(f"  • Document {i+1}: {url}")

    try:
        uploaded_file_ids = []

        for i, url in enumerate(urls):
            print(f"\n📥 Downloading document from: {url}")
            response = requests.get(url)
            response.raise_for_status()

            # Create a file-like object from the downloaded content
            file_content = BytesIO(response.content)
            file_content.name = f"canopy-in-botany-{i}.pdf"

            # Upload file using the new files API
            uploaded_file = client.files.create(
                file=file_content, purpose="assistants"  # Required purpose parameter
            )

            uploaded_file_ids.append(uploaded_file.id)
            print(f"✅ Uploaded file with ID: {uploaded_file.id}")

        # Add files to the vector store with chunking configuration
        for file_id in uploaded_file_ids:
            client.vector_stores.files.create(
                vector_store_id=vs.id,
                file_id=file_id,
                attributes={
                    "filename": "canopy-in-botany.pdf",
                    "metadata": "this is is a test",
                },
                chunking_strategy={
                    "type": "static",
                    "static": {
                        "max_chunk_size_tokens": 512,
                        "chunk_overlap_tokens": 50,
                    },
                },
            )
            print(f"✅ Added file {file_id} to vector store with chunking")

        print("\n✅ Document ingestion complete!")
        print("🎯 Your documents are now searchable via semantic similarity!")

    except Exception as e:
        print(f"\n❌ Document ingestion failed: {e}")
        print(
            "💡 This might be due to PDF processing issues or network connectivity. Try with different documents or check the PDF accessibility."
        )

    v_stores = client.vector_stores.list()
    for x in v_stores:
        print(f"{x}\n")

    files = client.files.list()
    for x in files:
        print(f"{x}\n")

    vs_files = client.vector_stores.files.list(vector_store_id=vs.id)
    for x in vs_files:
        print(f"{x}\n")


def query_rag(client, vs, prompt):
    search_results = client.vector_stores.search(
        vector_store_id=vs.id,
        query=prompt,
        max_num_results=5,
        search_mode="keyword",  # Use vector similarity search
    )

    for x in search_results.data:
        print(f"{x}\n")


def delete_vdb(client, vs):
    client.vector_stores.delete(vs.id)


class QA(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()
    # tools: list[dspy.Tool] = dspy.InputField()
    # outputs: dspy.ToolCalls = dspy.OutputField()


def weather(city: str) -> str:
    return "it is currently raining"


def timeofday(city: str) -> str:
    return "the current time is 21:00"


tools = {"weather": dspy.Tool(weather), "timeofday": dspy.Tool(timeofday)}

tool_list = [dspy.Tool(weather), dspy.Tool(timeofday)]


async def execute_dspy(predict: dspy.ReAct, user_request: str):
    history = dspy.History(messages=[])
    result = await predict.acall(question=user_request, history=history)
    print(result)
    dspy.inspect_history(n=50)


def lls_mcp_to_dspy(t: types.ToolDef, client: AsyncLlamaStackClient):
    args, arg_types, arg_desc = convert_input_schema_to_tool_args(t.input_schema)

    # Convert the MCP tool and Session to a single async method
    async def func(*args, **kwargs):
        print(f"Executing {t.name}")
        result = await client.tool_runtime.invoke_tool(tool_name=t.name,kwargs=kwargs)
        
        print(f"HEREREERERERERERERERERE {result}")

        # return _convert_mcp_tool_result(result)
        return result

    return dspy.Tool(
        func=func,
        name=t.name,
        desc=t.description,
        args=args,
        arg_types=arg_types,
        arg_desc=arg_desc,
    )


async def main(lls_client) -> None:
    toolr = await lls_client.tool_runtime.list_tools()
    for t in toolr:
        print(f"Tool runtimes {t}\n")
    print("-" * 60)
    model_list = await lls_client.models.list()
    llm = dspy.LM(
        "openai/" + model_list[0].identifier,
        api_base=base_url + "/v1/openai/v1",
        model_type="chat",
        api_key="this is a fake key",
    )
    LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"
    dspy.configure(lm=llm)
    tools = await lls_client.tools.list()
    dspy_tools = []
    for t in tools:
        if "mcp::" in t.toolgroup_id:
            dspy_tools.append(lls_mcp_to_dspy(t, lls_client))

    predict = dspy.ReAct(QA, dspy_tools)
    await execute_dspy(predict=predict, user_request="what skills are available ?")
    
    dspy.inspect_history()


if __name__ == "__main__":
    base_url = "http://localhost:8321"

    lls_client = AsyncLlamaStackClient(base_url=base_url)


    # dump_api(lls_client)

    asyncio.run(main(lls_client))
    
    #     execute_dspy(predict=predict, user_request="what skills are available ?")
    # )
    # while True:
    #     question = input("Type your question, end conversation by typing 'finish': ")
    #     if question == "finish":
    #         break
    #     response = await predict.acall(question=question, history=history,tools=tools)
    #     print(f"\n{response.answer} {response.outputs}\n")
    #     # history.messages.append({"question": question, **response})
    #     # result = None
    #     # for call in response.outputs.tool_calls:
    #     #     # Execute the tool call
    #     #     result = call.execute()
    #     #     # For versions earlier than 3.0.4b2, use: result = tools[call.name](**call.args)
    #     #     print(f"Tool: {call.name}")
    #     #     print(f"Args: {call.args}")
    #     #     print(f"Result: {result}")
    #     history.messages.append({"question": question, **response})


