# Copyright (c) Timescale, Inc. (2023)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code

from llama_index.vector_stores import TimescaleVectorStore
from llama_index import StorageContext
from llama_index.indices.vector_store import VectorStoreIndex

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from timescale_vector import client

from typing import List, Tuple

from llama_index.schema import TextNode
from llama_index.embeddings import OpenAIEmbedding
import psycopg2

def get_repos():
    with psycopg2.connect(dsn=st.secrets["TIMESCALE_SERVICE_URL"]) as connection:
        # Create a cursor within the context manager
        with connection.cursor() as cursor:
            select_data_sql = "SELECT * FROM time_machine_catalog;"
            cursor.execute(select_data_sql)

            catalog_entries = cursor.fetchall()

            catalog_dict = {}
            for entry in catalog_entries:
                repo_url, table_name = entry
                catalog_dict[repo_url] = table_name

            return catalog_dict

def tm_demo():
    repos = get_repos()

    if len(repos) > 0:
        repo = st.selectbox("Choose a repo", repos.keys())
    else:
        st.error("No repositiories found, please load some data first")
        return

    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about the git history"}
        ]

    vector_store = TimescaleVectorStore.from_params(
        service_url=st.secrets["TIMESCALE_SERVICE_URL"],
        table_name=repos[repo],
        time_partition_interval=timedelta(days=7),
    );

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    #storage_context = StorageContext.from_defaults(vector_store=vector_store)
    #index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history

st.set_page_config(page_title="Time machine demo", page_icon="📈")
st.markdown("# Time Machine Demo")
st.sidebar.header("Time_machine Demo")
st.write(
    """Time Machine Demo!"""
)

tm_demo()

#show_code(tm_demo)
