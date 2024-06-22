from typing import List

from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from langflow.base.vectorstores.model import LCVectorStoreComponent
from langflow.field_typing import Retriever
from langflow.io import DropdownInput, HandleInput, IntInput, MultilineInput, SecretStrInput, TextInput
from langflow.schema import Data


class CohereRerankComponent(LCVectorStoreComponent):
    display_name = "Cohere Rerank"
    description = "Rerank documents using the Cohere API."
    icon = "Cohere"

    inputs = [
        MultilineInput(
            name="search_query",
            display_name="Search Query",
        ),
        DropdownInput(
            name="model",
            display_name="Model",
            options=[
                "rerank-english-v3.0",
                "rerank-multilingual-v3.0",
                "rerank-english-v2.0",
                "rerank-multilingual-v2.0",
            ],
            value="rerank-english-v3.0",
        ),
        SecretStrInput(name="api_key", display_name="API Key"),
        IntInput(name="top_n", display_name="Top N", value=3),
        TextInput(name="user_agent", display_name="User Agent", value="langflow", advanced=True),
        HandleInput(name="retriever", display_name="Retriever", input_types=["Retriever"]),
    ]

    def build_base_retriever(self) -> Retriever:
        cohere_reranker = CohereRerank(
            api_key=self.api_key, model=self.model, top_n=self.top_n, user_agent=self.user_agent
        )
        retriever = ContextualCompressionRetriever(base_compressor=cohere_reranker, base_retriever=self.retriever)
        return retriever

    async def search_documents(self) -> List[Data]:
        retriever = self.build_base_retriever()
        documents = await retriever.ainvoke(self.search_query)
        data = self.to_data(documents)
        self.status = data
        return data