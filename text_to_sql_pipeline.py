# text_to_sql_pipeline.py
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine
import llm_config, database

# Initialize the database connection for LangChain's SQLDatabase
DATABASE_URL = database.SQLALCHEMY_DATABASE_URL

# Global variable for SQLDatabase instance
sql_db = None


def initialize_sql_database():
    """
    Initializes the SQLDatabase instance for LangChain.
    This should be called once on application startup.
    """
    global sql_db
    try:
        engine = create_engine(DATABASE_URL)
        # Include all tables in the database for the agent to be aware of them.
        # This is important for the agent to correctly identify 'content', 'feedback', 'question_answer_logs'.
        sql_db = SQLDatabase(
            engine, include_tables=["content", "feedback", "question_answer_logs"]
        )
        print("SQLDatabase for LangChain initialized.")
    except Exception as e:
        print(f"Error initializing SQLDatabase: {e}")
        sql_db = None  # Ensure it's None if initialization fails


def get_sql_agent_executor():
    """
    Creates and returns a LangChain SQL agent executor.
    """
    global sql_db
    if sql_db is None:
        initialize_sql_database()  # Try to initialize if not already

    if sql_db is None:
        raise RuntimeError("SQLDatabase is not initialized. Cannot create SQL agent.")

    llm = llm_config.get_llm()  # Use the configured LLM

    # Create a SQLDatabaseToolkit
    toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)

    # Define a custom prompt template for the SQL agent
    # Added explicit instruction for the Final Answer format.
    custom_suffix = """
    You are an agent designed to interact with a SQL database to answer user questions.
    Your goal is to answer the question accurately and in a human-readable format.
    You have access to tools to interact with the database.

    When the user asks a question about the database, your first step should always be to
    **list the tables available in the database** using the `sql_db_list_tables` tool.
    Then, if necessary, you should inspect the schema of relevant tables using `sql_db_schema`.
    Finally, you should construct and execute a SQL query using `sql_db_query` to answer the question.

    **IMPORTANT: Once you have determined the final answer, you MUST output it using the exact prefix "Final Answer:".**
    For example: "Final Answer: The number of topics is 5."

    Question: {input}
    {agent_scratchpad}
    """

    # Create the SQL agent
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,  # Set to True for debugging agent's thought process
        # ZERO_SHOT_REACT_DESCRIPTION is a more generic and often robust agent type
        # that works well across various LLMs, including Gemini, for this kind of task.
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,  # To gracefully handle parsing errors
        suffix=custom_suffix,  # Apply the custom suffix to guide the agent
        # Ensure the agent has access to intermediate steps for better reasoning if needed
        # return_intermediate_steps=True # Optional: useful for debugging agent's thought process
    )
    return agent_executor


async def query_database_natural_language(question: str) -> str:
    """
    Uses the LangChain SQL agent to answer natural language questions about the database.
    """
    try:
        agent_executor = get_sql_agent_executor()
        print(f"Querying database with natural language: {question}")
        # Use ainvoke for async execution
        response = await agent_executor.ainvoke({"input": question})
        # The final answer from ZERO_SHOT_REACT_DESCRIPTION agent is usually in 'output'
        answer = response.get(
            "output", "Could not find an answer or process the query."
        )
        return answer
    except Exception as e:
        print(f"Error during natural language SQL query: {e}")
        # Provide a more informative error message to the user
        return f"I encountered an error trying to answer your database question. It might be related to database access or the query complexity: {e}"
