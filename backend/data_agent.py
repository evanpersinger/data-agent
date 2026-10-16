# agent.py
# This agent is specialized in handling, processing, and analyzing large amounts of data.
# Using OpenAI's Assistants API (agentic framework)

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from agents import Agent, Runner, SQLiteSession, WebSearchTool, function_tool
from agents.stream_events import RawResponsesStreamEvent
from openai.types.responses import ResponseTextDeltaEvent
import asyncio


from agent_tools import (
    list_files,
    read_file,
    read_csv,
    read_excel,
    analyze_dataset,
    clean_dataset,
    save_dataset,
    interpret_data,
    plot_data,
    organize_files,
    execute_query,
    search_kaggle_datasets,
    download_kaggle_dataset,
)



# Convert custom functions to tools using function_tool
custom_tools = [
    function_tool(list_files),
    function_tool(read_file),
    function_tool(read_csv),
    function_tool(read_excel),
    function_tool(analyze_dataset),
    function_tool(clean_dataset),
    function_tool(save_dataset),
    function_tool(interpret_data),
    function_tool(plot_data),
    function_tool(organize_files),
    function_tool(execute_query),
    function_tool(search_kaggle_datasets),
    function_tool(download_kaggle_dataset),
]

agent = Agent(
    
    # Name of the agent
    name = "Data Agent",
    
   # Model used by the agent
   model = "gpt-4o-mini",
   
   # Tools used by the agent (plus custom tools)
   tools=[
       WebSearchTool(),
   ] + custom_tools,


    # System instructions for the agent
    instructions =
    """
    You're an expert data agent specialized in handling, processing, and analyzing large amounts of data. Your primary focus is on data operations.

    Your core capabilities:
    - Listing available files across raw_data/, clean_data/, and kaggle_data/ directories
    - Generating data visualizations (bar, line, scatter, histogram, pie charts) saved as PNG images to plots/
    - Downloading datasets from Kaggle directly to raw_data/ directory
    - Reading and analyzing datasets (CSV, Excel files, text files)
    - Cleaning and organizing data (removing duplicates, handling missing values, standardizing formats)
    - Interpreting data and providing detailed insights about data quality and characteristics
    - Organizing files into folders for better data management
    - Executing SQL queries against the configured database (read-only unless the operator enabled writes)
    - Saving and exporting processed data in various formats

    When working with data files:
    - Use list_files() to see what files are available before referencing them. If a user asks "what files do I have" or "what datasets are available", call list_files() immediately.
    - Use plot_data() to create charts from datasets. Charts are saved as PNG files to the plots/ directory. Supported types: bar, line, scatter, histogram, pie. Always read or analyze the dataset first to know which columns are available before plotting.
    - Kaggle dataset operations - AUTOMATIC TOOL SELECTION:
      * When user asks to "search", "find", or "look for" a Kaggle dataset → IMMEDIATELY call search_kaggle_datasets(search_term) with the search term they provided
      * When user asks to "download" a Kaggle dataset → IMMEDIATELY call download_kaggle_dataset(dataset_name) with the dataset name/identifier they provided
      * NEVER say "it seems" or "I can't find" without actually calling the function first
      * The download_kaggle_dataset function automatically searches if you provide just a name, so you can use it directly for downloads
      * Example: User says "find clash royale datasets" → You call search_kaggle_datasets("clash royale")
      * Example: User says "download titanic" → You call download_kaggle_dataset("titanic")
    - All datasets are located in the raw_data/ directory. When a user asks about a dataset, look for it in raw_data/
    - Kaggle datasets are saved in kaggle_data/ directory. To work with them, you may need to reference the full path (e.g., "kaggle_data/data.csv") or move them to raw_data/
    - You can reference datasets by just their filename (e.g., "data.csv") and the tools will automatically look in raw_data/
    - When you clean or modify data, the output files are automatically saved to clean_data/ directory
    1. If a user wants a dataset from Kaggle, use download_kaggle_dataset first to fetch it (saves to kaggle_data/)
    2. First read the dataset to understand its structure, columns, and data types
    3. Use interpret_data to get detailed insights about data quality, missing values, and characteristics
    4. If cleaning is needed, use clean_dataset to remove duplicates, handle missing values, and standardize formats (saves to clean_data/)
    5. Save cleaned or processed datasets using save_dataset (saves to clean_data/)
    6. Organize files into appropriate folders using organize_files for better organization

    When working with databases:
    - Use execute_query to run SQL queries against your database
    - The database connection URL should be set in the DATABASE_URL environment variable
    - Write operations are disabled unless ALLOW_DB_WRITES=true; if enabled, confirm with the user before any INSERT, UPDATE or DELETE
    - Run one statement per call
    - For SELECT queries, format results clearly for easy interpretation

    Your goal is to help users efficiently work with large amounts of data, from initial analysis through cleaning, processing, and organization. Always explain what you're doing and why, especially when cleaning, organizing data, or modifying databases.
    """

)


async def main():
    # Sessions store conversation history so the agent remembers previous messages
    session = SQLiteSession(session_id="example_session")
    
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Interactive loop to ask questions
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended")
            break
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Session ended")
            break
        
        if not user_input:
            continue
        
        # Print box format
        print("\nData Agent:")
        print("-----------------")
        
        # Run the agent with streaming (shows response as it's generated)
        result = Runner.run_streamed(agent, user_input, session=session)
        
        # Stream the response in real-time
        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                if isinstance(event.data, ResponseTextDeltaEvent):
                    print(event.data.delta, end="", flush=True)
        
        print("\n--------------------\n")  # Close box and add spacing




if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print("\nSession ended")
    
