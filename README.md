# Data Agent

An AI agent specialized in handling, processing, and analyzing large amounts of data. Built with OpenAI's agentic framework, it can read datasets, clean messy data, execute database queries, interpret data insights, and organize your files.

## Features

- **Search & Download Data**: Find Kaggle datasets by keyword and download them
- **List Files**: See what's available across the data directories
- **Read Data**: Read CSV, Excel, and text files
- **Analyze Data**: Get statistics, missing values, and data structure insights
- **Clean Data**: Remove duplicates, handle missing values, standardize formats
- **Interpret Data**: Get detailed insights about your datasets and data quality
- **Plot Data**: Generate bar, line, scatter, histogram, and pie charts as PNGs in `plots/`
- **Database Operations**: Execute SQL queries against databases (SQLite, PostgreSQL, MySQL, etc.)
- **Web Search**: Look things up online when a dataset needs outside context
- **Organize Files**: Move files into organized folders for better data management
- **Save & Export**: Save processed data in CSV or Excel formats

## Setup

1. Clone the repository and navigate to the project directory

2. Install dependencies using uv:
```bash
uv sync
```

3. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_api_key_here
DATABASE_URL=sqlite:///database.db  # Optional: for database operations
KAGGLE_USERNAME=your_kaggle_username  # Optional: for Kaggle dataset downloads
KAGGLE_KEY=your_kaggle_api_key  # Optional: for Kaggle dataset downloads
```

   **Getting your API keys:**
   
   **OpenAI API Key:**
   1. Go to https://platform.openai.com/api-keys
   2. Sign in or create an account
   3. Click "Create new secret key"
   4. Copy the key (you won't be able to see it again!)
   5. Add it to your `.env` file as `OPENAI_API_KEY=sk-...`
   
   **Kaggle API Key:**
   1. Go to https://www.kaggle.com/ and sign in (create account if needed)
   2. Go to https://www.kaggle.com/settings
   3. Scroll to "API" section and click "Create New Token"
   4. This downloads a `kaggle.json` file to your computer
   5. Open the `kaggle.json` file - it contains:
      ```json
      {"username":"your_username","key":"your_api_key_here"}
      ```
   6. Copy the `username` and `key` values
   7. Add them to your `.env` file:
      ```
      KAGGLE_USERNAME=your_username
      KAGGLE_KEY=your_api_key_here
      ```
   
   **Alternative Kaggle setup (using kaggle.json file):**
   - Instead of adding to `.env`, you can place `kaggle.json` in `~/.kaggle/` directory
   - Create the directory: `mkdir -p ~/.kaggle`
   - Move the file: `mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
   
   **Important:** Never commit your `.env` file or `kaggle.json` to git! They contain sensitive API keys.

4. Run the agent in the terminal:
```bash
uv run python backend/data_agent.py
```

## Web UI

There's also a browser frontend, which is usually nicer than the REPL. It needs
two processes running at once.

1. Start the backend (port 8017):
```bash
uv run uvicorn --app-dir backend server:app --reload --port 8017
```

2. In a second terminal, start the frontend (port 3006):
```bash
cd frontend
pnpm install   # first time only
pnpm dev
```

3. Open http://localhost:3006

The frontend proxies `/api` to the backend, so both sides are same-origin and no
CORS setup is needed. The CLI and the web UI share the same agent and the same
data directories, so you can switch between them freely.

Conversation history for the web UI is written to `sessions.db` at the project
root. Delete that file to wipe all chat history.

## Docker Setup (Alternative)

Docker provides an isolated environment with pre-configured databases (PostgreSQL). This is useful if you want to avoid installing databases locally or ensure consistent environments across different machines.

### Prerequisites
- Docker and Docker Compose installed on your system

### Quick Start with Docker

1. Make sure your `.env` file is set up (see Setup step 3 above)

2. Update your `.env` file to use PostgreSQL (optional, SQLite still works):
```
DATABASE_URL=postgresql://postgres:<POSTGRES_PASSWORD>@postgres:5432/data_agent
```

3. Start all services in detached mode (runs in background, doesn't require keeping a terminal open):
```bash
docker-compose up -d
```

4. To start services and view logs in the terminal:
```bash
docker-compose up
```

5. To stop services:
```bash
docker-compose down
```

6. To rebuild after code changes:
```bash
docker-compose up --build -d
```

### Docker Benefits

- **PostgreSQL included**: No need to install PostgreSQL locally - it runs in a container
- **Consistent environment**: Same Python version and dependencies everywhere
- **Easy cleanup**: Remove containers to reset everything
- **Data persistence**: Your data directories (`raw_data/`, `clean_data/`, etc.) are mounted as volumes, so files persist between container restarts

### Docker Commands

- View logs (useful when running in detached mode): `docker-compose logs -f data-agent`
- Execute commands in container: `docker-compose exec data-agent bash`
- Stop and remove volumes: `docker-compose down -v` (WARNING: deletes database data)

For more Docker commands, see `commands.md`.

## Usage

Once the agent is running, you can interact with it in the REPL. Example commands:

- "What files do I have?"
- "Download the titanic dataset from Kaggle"
- "Read the file data.csv"
- "Analyze the dataset sales.xlsx"
- "Clean the dataset messy_data.csv"
- "Interpret the data in customers.csv"
- "Run this SQL query: SELECT * FROM users WHERE age > 25"
- "Plot a bar chart of sales by region from sales.csv"
- "Organize the file report.pdf into the reports folder"
- "Save the cleaned data to output.csv"

## Project Structure

```
data_agent/
├── backend/
│   ├── data_agent.py      # Agent configuration, instructions, and the CLI REPL
│   ├── agent_tools.py     # Data processing functions/tools the agent can use
│   └── server.py          # FastAPI wrapper that exposes the agent over HTTP
├── frontend/
│   ├── src/
│   │   ├── App.tsx        # Chat UI
│   │   ├── api.ts         # Streaming fetch client
│   │   ├── types.ts       # Shared types
│   │   └── index.css      # Theme (colors live in the :root block)
│   ├── vite.config.ts     # Dev server on 3006, proxies /api to 8017
│   └── package.json
├── Dockerfile             # Image for the agent, installs the project with uv
├── docker-compose.yml     # Agent plus a PostgreSQL service
├── commands.md            # Docker command reference
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependency versions for reproducibility
├── README.md              # This file
├── .env                   # Environment variables (API keys) - NOT committed to git
├── .venv/                 # Virtual environment (not committed to git)
├── raw_data/              # Directory for raw datasets
├── clean_data/            # Directory for cleaned/processed datasets
├── kaggle_data/           # Directory for Kaggle downloads
├── plots/                 # Charts written by plot_data()
└── sessions.db            # Web UI chat history (SQLite) - NOT committed to git
```

## How It Works

- **backend/data_agent.py**: Contains the agent's configuration, instructions, and launches the REPL loop. Uses OpenAI's agentic framework (`openai-agents`) to create an intelligent agent that can use custom tools.
- **backend/agent_tools.py**: Contains all the functions the agent can use (reading, cleaning, analyzing data, etc.). These functions are converted to tools using `function_tool()` so the agent can call them automatically.
- **backend/server.py**: Imports the same agent object and streams its output over `POST /api/chat`, so the browser can talk to it. Importing `data_agent` does not start the REPL, since that is guarded by `__name__ == "__main__"`.
- **frontend/**: Vite + React + TypeScript. Deliberately minimal, just a chat window that streams text back. No Node server in production, since the Python backend does the real work.
- **pyproject.toml**: Defines the project metadata and all dependencies, organized by category for easy maintenance.
- **uv.lock**: Automatically generated lock file that ensures everyone uses the exact same dependency versions for reproducibility.

Note that the data directories live at the project root, not inside `backend/`.
`agent_tools.py` resolves them with `Path(__file__).parent.parent`, and
`docker-compose.yml` mounts them at `/app`, so the two stay in sync.

The agent uses GPT-4o mini (via OpenAI's agentic framework) to understand your requests and automatically calls the appropriate tools to help you work with your data. The agent maintains conversation history using SQLite sessions, so it remembers previous interactions.
