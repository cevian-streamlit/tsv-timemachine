# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
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

# Helper function to split name and email given an author string consisting of Name Lastname <email>
def split_name(input_string: str) -> Tuple[str, str]:
    if input_string is None:
        return None, None
    start = input_string.find("<")
    end = input_string.find(">")
    name = input_string[:start].strip()
    email = input_string[start+1:end].strip()
    return name, email

def create_uuid(date_string: str):
    if date_string is None:
        return None
    time_format = "%a %b %d %H:%M:%S %Y %z"
    datetime_obj = datetime.strptime(date_string, time_format)
    uuid = client.uuid_from_time(datetime_obj)
    return str(uuid)

def create_date(input_string: str) -> datetime:
    if input_string is None:
        return None
    # Define a dictionary to map month abbreviations to their numerical equivalents
    month_dict = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }

    # Split the input string into its components
    components = input_string.split()
    # Extract relevant information
    day = components[2]
    month = month_dict[components[1]]
    year = components[4]
    time = components[3]
    timezone_offset_minutes = int(components[5])  # Convert the offset to minutes
    timezone_hours = timezone_offset_minutes // 60  # Calculate the hours
    timezone_minutes = timezone_offset_minutes % 60  # Calculate the remaining minutes
    # Create a formatted string for the timestamptz in PostgreSQL format
    timestamp_tz_str = (
        f"{year}-{month}-{day} {time}+{timezone_hours:02}{timezone_minutes:02}"
    )
    return timestamp_tz_str

# Create a Node object from a single row of data
def create_node(row):
    record = row.to_dict()
    record_name, _ = split_name(record["author"])
    record_content = (
        str(record["date"])
        + " "
        + record_name
        + " "
        + str(record["change summary"])
        + " "
        + str(record["change details"])
    )
    # Can change to TextNode as needed
    node = TextNode(
        id_=create_uuid(record["date"]),
        text=record_content,
        metadata={
            "commit": record["commit"],
            "author": record_name,
            "date": create_date(record["date"]),
        },
    )
    return node

def get_vector_store():
    return TimescaleVectorStore.from_params(
        service_url=st.secrets["TIMESCALE_SERVICE_URL"],
        table_name="li_commit_history",
        time_partition_interval=timedelta(days=7),
    );

def init_db():
    file_path = Path("data/commit_history.csv")
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Light data cleaning on CSV
    df.dropna(inplace=True)
    df = df.astype(str)
    df = df[:1000]

    nodes = [create_node(row) for _, row in df.iterrows()]

    embedding_model = OpenAIEmbedding()
    embedding_model.api_key = st.secrets["OPENAI_API_KEY"]

    start = time.time()
    for node in nodes:
        node_embedding = embedding_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    duration = time.time()-start
    st.markdown(f"Getting ai embeddings took {duration} seconds")

    start = time.time()
    ts_vector_store = get_vector_store()
    _ = ts_vector_store.add(nodes)

    duration = time.time()-start
    st.markdown(f"Writing to the db took {duration} seconds")


def tm_demo():
    if st.button("Re-import data into the database"):
        init_db()

    if st.button("Reset state"):
        st.session_state.clear()

    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about Timescale's git history"}
        ]

    vector_store = get_vector_store()

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
    #VectorStoreIndex.from_documents(docs, service_context=service_context)


st.set_page_config(page_title="Time machine demo", page_icon="📈")
st.markdown("# Time Machine Demo")
st.sidebar.header("Time_machine Demo")
st.write(
    """Time Machine Demo!"""
)

tm_demo()

show_code(tm_demo)
