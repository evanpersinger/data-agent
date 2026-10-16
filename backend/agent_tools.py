# agent_tools.py
# Functions/tools that the agent can use

import io
import pandas as pd
from pathlib import Path
import os
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables from .env file
load_dotenv()


# Base directory for file operations. This file lives in backend/, but the data
# directories sit at the project root (and are mounted there by docker-compose),
# so walk up one level.
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
CLEAN_DATA_DIR = BASE_DIR / "clean_data"
KAGGLE_DATA_DIR = BASE_DIR / "kaggle_data"
PLOTS_DIR = BASE_DIR / "plots"

# Ensure directories exist
CLEAN_DATA_DIR.mkdir(exist_ok=True)
KAGGLE_DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


DATA_DIRS = (RAW_DATA_DIR, CLEAN_DATA_DIR, KAGGLE_DATA_DIR)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_data_path(filename: str, default_dir: Path = RAW_DATA_DIR) -> Path:
    """Resolve a user/model supplied filename to a path inside a data directory.

    A bare name is looked up in default_dir. A name containing a separator is
    taken relative to the project root. Either way the resolved path (symlinks
    and ".." included) must sit inside raw_data/, clean_data/ or kaggle_data/,
    otherwise ValueError is raised. This keeps the agent away from .env,
    kaggle.json, sessions.db and the source tree.
    """
    if "/" not in filename and "\\" not in filename:
        candidate = default_dir / filename
    else:
        candidate = BASE_DIR / filename
    resolved = candidate.resolve()
    if not any(_is_within(resolved, d.resolve()) for d in DATA_DIRS):
        raise ValueError(
            f"Access denied: '{filename}' is outside raw_data/, clean_data/ and kaggle_data/."
        )
    return resolved


def _safe_output_path(directory: Path, name: str) -> Path:
    """Return directory/name, rejecting names that carry a path component."""
    if not name or name != Path(name).name or name in (".", ".."):
        raise ValueError(f"Invalid output filename '{name}': use a plain file name, no folders.")
    return directory / name


def _dataset_summary(df: pd.DataFrame) -> str:
    """Shape, column names/dtypes and the first rows, so the agent knows what to plot or clean."""
    summary = "### Dataset Overview\n"
    summary += f"- **Total Rows:** {df.shape[0]:,}\n"
    summary += f"- **Total Columns:** {df.shape[1]}\n\n"
    summary += "### Columns\n"
    summary += "\n".join(f"- {col} ({dtype})" for col, dtype in df.dtypes.items())
    summary += "\n\n### First 5 rows\n" + df.head().to_string(index=False)
    return summary


# list files in data directories
def list_files(directory: str = "all") -> str:
    """
    List files available in the data directories.

    Args:
        directory: Which directory to list. Options:
                   "raw_data"    - files in raw_data/
                   "clean_data"  - files in clean_data/
                   "kaggle_data" - files in kaggle_data/
                   "all"         - all three directories (default)
    """
    dirs = {
        "raw_data": RAW_DATA_DIR,
        "clean_data": CLEAN_DATA_DIR,
        "kaggle_data": KAGGLE_DATA_DIR,
    }

    if directory != "all" and directory not in dirs:
        return f"Unknown directory '{directory}'. Choose from: raw_data, clean_data, kaggle_data, all"

    targets = dirs if directory == "all" else {directory: dirs[directory]}
    output = ""

    for name, path in targets.items():
        output += f"{name}/\n"
        if not path.exists():
            output += "  (directory does not exist)\n"
            continue
        files = sorted(path.iterdir())
        if not files:
            output += "  (empty)\n"
        else:
            for f in files:
                if f.is_file():
                    size_kb = f.stat().st_size / 1024
                    output += f"  {f.name}  ({size_kb:.1f} KB)\n"
                elif f.is_dir():
                    output += f"  {f.name}/  (folder)\n"
        output += "\n"

    return output.strip()


# read content of a text file
def read_file(filename: str) -> str:
    """Read a text file from raw_data/, clean_data/ or kaggle_data/."""
    try:
        file_path = _resolve_data_path(filename)
    except ValueError as e:
        return str(e)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Could not read {file_path}: {e}"


# read CSV file
def read_csv(filename: str) -> str:
    """Read a CSV file and return a summary of its structure and first few rows.
    If filename doesn't contain a path separator, looks in raw_data/ directory."""
    try:
        file_path = _resolve_data_path(filename)
    except ValueError as e:
        return str(e)
    try:
        df = pd.read_csv(file_path)
        return _dataset_summary(df)
    except Exception as e:
        return f"Could not read CSV {file_path}: {e}"


# read Excel file
def read_excel(filename: str, sheet_name: str = None) -> str:
    """Read an Excel file and return a summary of its structure.
    If filename doesn't contain a path separator, looks in raw_data/ directory."""
    try:
        file_path = _resolve_data_path(filename)
    except ValueError as e:
        return str(e)
    try:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        return _dataset_summary(df)
    except Exception as e:
        return f"Could not read Excel {file_path}: {e}"


# analyze dataset
def analyze_dataset(filename: str) -> str:
    """Analyze a dataset (CSV/Excel) and return statistics and insights.
    If filename doesn't contain a path separator, looks in raw_data/ directory."""
    try:
        file_path = _resolve_data_path(filename)
    except ValueError as e:
        return str(e)
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return f"Unsupported file type. Use CSV or Excel files."
        
        summary = f"Analysis of {filename}:\n\n"
        summary += f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        summary += "Column info:\n"
        buf = io.StringIO()
        df.info(buf=buf)
        summary += buf.getvalue()
        summary += "\n\nBasic statistics:\n"
        summary += df.describe().to_string()
        summary += "\n\nMissing values:\n"
        summary += df.isnull().sum().to_string()
        return summary
    except Exception as e:
        return f"Could not analyze {file_path}: {e}"


# clean dataset
def clean_dataset(filename: str, remove_duplicates: bool = True, fill_missing: str = "drop", output_filename: str = None) -> str:
    """
    Clean a dataset by removing duplicates, handling missing values, and standardizing formats.
    
    Args:
        filename: Input file to clean (if just a filename, looks in raw_data/)
        remove_duplicates: Whether to remove duplicate rows (default: True)
        fill_missing: How to handle missing values - "drop" (drop rows), "fill" (fill with mean/median), or "zero" (fill with 0)
        output_filename: Optional output filename. If not provided, saves as "cleaned_{original_filename}"
    """
    try:
        file_path = _resolve_data_path(filename)
    except ValueError as e:
        return str(e)
    try:
        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return f"Unsupported file type. Use CSV or Excel files."
        
        if fill_missing not in ("drop", "fill", "zero"):
            return f"Invalid fill_missing '{fill_missing}'. Use 'drop', 'fill' or 'zero'."

        original_shape = df.shape
        changes = []
        
        # Remove duplicates
        if remove_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            after = len(df)
            if before != after:
                changes.append(f"Removed {before - after} duplicate rows")
        
        # Handle missing values
        if fill_missing == "drop":
            before = len(df)
            df = df.dropna()
            after = len(df)
            if before != after:
                changes.append(f"Dropped {before - after} rows with missing values")
        elif fill_missing == "fill":
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
            changes.append("Filled missing numeric values with median")
        elif fill_missing == "zero":
            df = df.fillna(0)
            changes.append("Filled missing values with 0")
        
        # Standardize column names (remove spaces, lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        changes.append("Standardized column names")
        
        # Determine output filename
        if output_filename is None:
            original_name = Path(filename).name
            if original_name.endswith('.csv'):
                output_filename = f"cleaned_{original_name}"
            else:
                output_filename = f"cleaned_{original_name.rsplit('.', 1)[0]}.csv"
        
        # Save to clean_data/ directory
        output_path = _safe_output_path(CLEAN_DATA_DIR, output_filename)
        df.to_csv(output_path, index=False)
        
        result = f"Cleaned dataset saved to clean_data/{output_filename}\n"
        result += f"Original shape: {original_shape[0]} rows × {original_shape[1]} columns\n"
        result += f"Cleaned shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        result += f"Changes made:\n" + "\n".join(f"  - {change}" for change in changes)
        return result
    except Exception as e:
        return f"Could not clean {file_path}: {e}"


# save dataset
def save_dataset(filename: str, output_filename: str, format: str = "csv") -> str:
    """
    Save a dataset to a new file in the specified format. Saves to clean_data/ directory.
    
    Args:
        filename: Input file to save (if just a filename, looks in raw_data/)
        output_filename: Name of the output file (will be saved in clean_data/)
        format: Output format - "csv" or "excel" (default: "csv")
    """
    try:
        file_path = _resolve_data_path(filename)
    except ValueError as e:
        return str(e)
    try:
        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return f"Unsupported input file type. Use CSV or Excel files."
        
        # Save to clean_data/ directory
        output_path = _safe_output_path(CLEAN_DATA_DIR, output_filename)
        
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "excel":
            df.to_excel(output_path, index=False)
        else:
            return f"Unsupported output format. Use 'csv' or 'excel'."
        
        return f"Dataset saved to clean_data/{output_filename} ({df.shape[0]} rows × {df.shape[1]} columns)"
    except Exception as e:
        return f"Could not save dataset: {e}"


# interpret data
def interpret_data(filename: str) -> str:
    """
    Provide detailed interpretation and insights about a dataset.
    If filename doesn't contain a path separator, looks in raw_data/ directory.
    """
    try:
        file_path = _resolve_data_path(filename)
    except ValueError as e:
        return str(e)
    try:
        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return f"Unsupported file type. Use CSV or Excel files."
        
        if df.empty:
            return f"{filename} has no rows or no columns, nothing to interpret."

        interpretation = f"Data Interpretation for {filename}:\n\n"
        
        # Basic info
        interpretation += f"Dataset Overview:\n"
        interpretation += f"  - Total records: {df.shape[0]:,}\n"
        interpretation += f"  - Total columns: {df.shape[1]}\n"
        interpretation += f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB\n\n"
        
        # Column types and insights
        interpretation += "Column Analysis:\n"
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            
            interpretation += f"  - {col} ({dtype}): "
            if null_count > 0:
                interpretation += f"{null_count} missing values ({null_pct:.1f}%)\n"
            else:
                interpretation += "No missing values\n"
            
            # Add specific insights based on data type
            if pd.api.types.is_numeric_dtype(df[col]):
                interpretation += f"    Range: {df[col].min()} to {df[col].max()}, Mean: {df[col].mean():.2f}\n"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                interpretation += f"    Date range: {df[col].min()} to {df[col].max()}\n"
            else:
                unique_count = df[col].nunique()
                interpretation += f"    {unique_count} unique values"
                if unique_count <= 10:
                    interpretation += f": {', '.join(map(str, df[col].unique()[:10]))}\n"
                else:
                    interpretation += f"\n"
        
        # Data quality summary
        interpretation += "\nData Quality Summary:\n"
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        interpretation += f"  - Completeness: {((total_cells - missing_cells) / total_cells * 100):.1f}%\n"
        interpretation += f"  - Duplicate rows: {df.duplicated().sum()}\n"
        
        return interpretation
    except Exception as e:
        return f"Could not interpret {file_path}: {e}"


# organize files
def organize_files(filename: str, target_folder: str = None) -> str:
    """
    Organize files into folders for better data management.
    
    Args:
        filename: Name of the file to organize (if just a filename, looks in current directory)
        target_folder: Target folder name. If not provided, organizes based on file type.
    """
    try:
        import shutil
        
        file_path = _resolve_data_path(filename)
        
        if not file_path.is_file():
            return f"File {filename} not found."
        
        # Determine target folder
        if target_folder is None:
            # Organize by file extension
            ext = file_path.suffix.lower()
            if ext in ['.csv', '.xlsx', '.xls']:
                target_folder = "data_files"
            elif ext in ['.txt', '.md', '.json']:
                target_folder = "text_files"
            elif ext in ['.pdf', '.doc', '.docx']:
                target_folder = "documents"
            else:
                target_folder = "other_files"
        
        # The target must be a single folder name. It is created inside the data
        # directory the file already lives in, so moved files stay reachable by the
        # other tools and nothing can be dropped into the source tree.
        if (target_folder != Path(target_folder).name or target_folder.startswith('.')
                or target_folder in ('.', '..', '')):
            return f"Invalid target folder '{target_folder}': use a plain folder name."
        root = next(d.resolve() for d in DATA_DIRS if _is_within(file_path, d.resolve()))
        target_dir = root / target_folder
        target_dir.mkdir(exist_ok=True)
        
        # Move file
        destination = target_dir / file_path.name
        if destination.exists():
            return f"Not moved: {target_folder}/{file_path.name} already exists."
        shutil.move(str(file_path), str(destination))
        
        return f"File {filename} moved to {target_folder}/{file_path.name}"
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Could not organize file: {e}"


# execute database query
def execute_query(query: str, database_url: str = None) -> str:
    """
    Execute a SQL query against a database and return the results.
    
    Args:
        query: SQL query to execute (SELECT, INSERT, UPDATE, DELETE, etc.)
        database_url: Database connection string. If not provided, uses DATABASE_URL from .env file.
                     Examples:
                     - SQLite: "sqlite:///database.db"
                     - PostgreSQL: "postgresql://user:password@localhost/dbname"
                     - MySQL: "mysql://user:password@localhost/dbname"
    """
    try:
        # Get database URL from parameter or environment variable
        if database_url is None:
            database_url = os.getenv("DATABASE_URL")
            if database_url is None:
                return "Error: No database URL provided. Set DATABASE_URL in .env file or pass it as a parameter."
        
        # Create database connection
        engine = create_engine(database_url)
        
        # Execute query
        with engine.connect() as connection:
            # For SELECT queries, return results as a formatted table
            if query.strip().upper().startswith('SELECT'):
                df = pd.read_sql(text(query), connection)
                if df.empty:
                    return "Query executed successfully. No rows returned."
                
                result_str = f"Query returned {len(df)} rows:\n\n"
                result_str += df.to_string(index=False)
                return result_str
            else:
                # For INSERT, UPDATE, DELETE, etc., execute and commit
                result = connection.execute(text(query))
                connection.commit()
                return f"Query executed successfully. Rows affected: {result.rowcount if hasattr(result, 'rowcount') else 'N/A'}"
                
    except Exception as e:
        return f"Database error: {str(e)}"


# Resolve Kaggle credentials from the environment, falling back to kaggle.json in
# the project root. Both Kaggle libraries read these off os.environ rather than
# taking arguments, so they are exported here. Returns the pair as found so
# callers can report which half is missing.
def _load_kaggle_credentials() -> tuple[str | None, str | None]:
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_KEY")

    if not username or not key:
        kaggle_json_path = BASE_DIR / "kaggle.json"
        if kaggle_json_path.exists():
            try:
                with open(kaggle_json_path, 'r') as f:
                    kaggle_creds = json.load(f)
                username = username or kaggle_creds.get("username", "")
                key = key or kaggle_creds.get("key", "")
            except Exception:
                pass  # If we can't read it, the caller reports the missing credentials

    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key

    return username, key


# search for Kaggle datasets
def search_kaggle_datasets(search_term: str, max_results: int = 5) -> str:
    """
    Search for Kaggle datasets by name and return matching dataset identifiers.
    
    Args:
        search_term: The name or keywords to search for (e.g., "titanic", "housing prices")
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        A formatted string with matching datasets and their identifiers
    """
    try:
        kaggle_username, kaggle_key = _load_kaggle_credentials()
        if not kaggle_username or not kaggle_key:
            return "Error: Kaggle authentication not configured. Please set up your Kaggle API credentials."
        
        # Use kaggle library for searching
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            # Search for datasets - try multiple strategies since Kaggle search is literal
            # Strategy 1: Try the full search term
            datasets = api.dataset_list(search=search_term, max_size=1000, sort_by='hottest')
            
            # Strategy 2: If no results, try intelligent keyword combinations
            if not datasets:
                # Extract meaningful keywords (remove stop words and short words)
                stop_words = {'data', 'dataset', 'datasets', 'the', 'a', 'an', 'for', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'is', 'are', 'was', 'were'}
                keywords = [w.strip('.,!?;:') for w in search_term.lower().split() 
                           if len(w.strip('.,!?;:')) > 2 and w.strip('.,!?;:') not in stop_words]
                
                if keywords:
                    # Collect all results from different search strategies
                    all_results = []
                    seen_refs = set()
                    
                    # Strategy 2a: Try all 2-word combinations of keywords
                    if len(keywords) >= 2:
                        from itertools import combinations
                        for combo in combinations(keywords, 2):
                            combo_search = ' '.join(combo)
                            results = api.dataset_list(search=combo_search, max_size=100, sort_by='hottest')
                            for result in results:
                                if result.ref not in seen_refs:
                                    all_results.append(result)
                                    seen_refs.add(result.ref)
                    
                    # Strategy 2b: Try all 3-word combinations if we have enough keywords
                    if len(keywords) >= 3 and len(all_results) < 10:
                        for combo in combinations(keywords, 3):
                            combo_search = ' '.join(combo)
                            results = api.dataset_list(search=combo_search, max_size=50, sort_by='hottest')
                            for result in results:
                                if result.ref not in seen_refs:
                                    all_results.append(result)
                                    seen_refs.add(result.ref)
                    
                    # Strategy 2c: Try individual important keywords (prioritize longer/more specific ones)
                    if len(all_results) < 5:
                        # Sort keywords by length (longer = more specific)
                        sorted_keywords = sorted(keywords, key=len, reverse=True)
                        for keyword in sorted_keywords[:3]:  # Try top 3 most specific keywords
                            results = api.dataset_list(search=keyword, max_size=50, sort_by='hottest')
                            for result in results:
                                if result.ref not in seen_refs:
                                    all_results.append(result)
                                    seen_refs.add(result.ref)
                    
                    # Use collected results if we found any
                    if all_results:
                        datasets = all_results
            
            if not datasets:
                return f"No datasets found matching '{search_term}'. Try a different search term."
            
            # Format results
            result = f"Found {len(datasets)} dataset(s) matching '{search_term}':\n\n"
            for i, dataset in enumerate(datasets[:max_results], 1):
                result += f"{i}. {dataset.ref}\n"
                result += f"   Title: {dataset.title}\n"
                # Use the correct attribute name (download_count)
                if hasattr(dataset, 'download_count') and dataset.download_count:
                    result += f"   Downloads: {dataset.download_count:,}\n"
                if hasattr(dataset, 'usability_rating') and dataset.usability_rating:
                    result += f"   Rating: {dataset.usability_rating:.2f}\n"
                if hasattr(dataset, 'description') and dataset.description:
                    desc = dataset.description[:100] + "..." if len(dataset.description) > 100 else dataset.description
                    result += f"   Description: {desc}\n"
                result += "\n"
            
            if len(datasets) > max_results:
                result += f"... and {len(datasets) - max_results} more results. Use the full identifier (username/dataset-name) to download."
            
            return result
            
        except ImportError:
            return "Error: kaggle library not installed. Install it with: pip install kaggle"
        except Exception as e:
            return f"Error searching datasets: {str(e)}"
            
    except Exception as e:
        return f"Error: {str(e)}"


# download dataset from Kaggle
def download_kaggle_dataset(dataset: str, unzip: bool = True) -> str:
    """
    MANDATORY: Use this function whenever the user asks to download a dataset from Kaggle.
    This function automatically searches Kaggle and downloads the best matching dataset.
    
    Download a dataset from Kaggle and save it to the kaggle_data/ directory.
    The function automatically searches Kaggle if you provide just a dataset name (e.g., "titanic" or "Clash Royal Cards data").
    It finds the most popular matching dataset and downloads it.
    
    Args:
        dataset: Dataset name or identifier. Can be:
                 - Just a name (e.g., "titanic", "Clash Royal Cards data") - function will search and find best match
                 - Full identifier in format "username/dataset-name" (e.g., "heptapod/titanic") - downloads directly
        unzip: Whether to automatically unzip downloaded files (default: True, always unzipped with kagglehub)
    
    Returns:
        A message indicating success and where files were saved, or an error message if dataset not found.
        The function handles all searching automatically - you don't need to search first.
    """
    try:
        import shutil
        
        kaggle_username, kaggle_key = _load_kaggle_credentials()
        if not kaggle_username or not kaggle_key:
            return f"Error: Kaggle authentication not configured. Please set up your Kaggle API credentials:\n\n" \
                   f"1. Go to https://www.kaggle.com/settings and create an API token\n" \
                   f"2. Add to your .env file:\n" \
                   f"   KAGGLE_USERNAME=your_username\n" \
                   f"   KAGGLE_KEY=your_api_key\n\n" \
                   f"Current status:\n" \
                   f"  KAGGLE_USERNAME: {'SET' if kaggle_username else 'NOT SET'}\n" \
                   f"  KAGGLE_KEY: {'SET' if kaggle_key else 'NOT SET'}\n\n" \
                   f"After adding to .env, restart the script."
        
        # Use kagglehub (newer, simpler API)
        try:
            import kagglehub
        except ImportError:
            return f"Error: kagglehub library not installed. Install it with: pip install kagglehub"
        
        # If dataset doesn't contain "/", search for it first to find the best match
        original_dataset = dataset
        result_prefix = ""  # Initialize empty, will be set if we search
        if '/' not in dataset:
            # Search for the dataset to find the best match
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                
                # Search for datasets - try multiple search strategies
                # Kaggle search is literal, so we need to try different keyword combinations
                search_results = api.dataset_list(search=dataset, max_size=1000, sort_by='hottest')
                
                # Extract meaningful keywords for flexible searching
                stop_words = {'data', 'dataset', 'datasets', 'cards', 'card', 'the', 'a', 'an', 'latest', 'for', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'is', 'are', 'was', 'were'}
                keywords = [w.strip('.,!?;:') for w in dataset.lower().split() 
                           if len(w.strip('.,!?;:')) > 2 and w.strip('.,!?;:') not in stop_words]
                
                # If no results, try more flexible searches
                is_fallback_search = False
                if not search_results and keywords:
                    # Collect all results from different search strategies
                    all_results = []
                    seen_refs = set()
                    
                    # Strategy 1: Try all 2-word combinations of keywords
                    if len(keywords) >= 2:
                        from itertools import combinations
                        for combo in combinations(keywords, 2):
                            combo_search = ' '.join(combo)
                            results = api.dataset_list(search=combo_search, max_size=100, sort_by='hottest')
                            for result in results:
                                if result.ref not in seen_refs:
                                    all_results.append(result)
                                    seen_refs.add(result.ref)
                    
                    # Strategy 2: Try all 3-word combinations if we have enough keywords
                    if len(keywords) >= 3 and len(all_results) < 20:
                        for combo in combinations(keywords, 3):
                            combo_search = ' '.join(combo)
                            results = api.dataset_list(search=combo_search, max_size=50, sort_by='hottest')
                            for result in results:
                                if result.ref not in seen_refs:
                                    all_results.append(result)
                                    seen_refs.add(result.ref)
                    
                    # Strategy 3: Try individual important keywords (prioritize longer/more specific ones)
                    if len(all_results) < 10:
                        # Sort keywords by length (longer = more specific)
                        sorted_keywords = sorted(keywords, key=len, reverse=True)
                        for keyword in sorted_keywords[:3]:  # Try top 3 most specific keywords
                            results = api.dataset_list(search=keyword, max_size=50, sort_by='hottest')
                            for result in results:
                                if result.ref not in seen_refs:
                                    all_results.append(result)
                                    seen_refs.add(result.ref)
                            if len(all_results) >= 20:
                                break
                    
                    # Use collected results if we found any
                    if all_results:
                        search_results = all_results
                        # Mark as fallback if we only got results from single keywords
                        if len(keywords) > 1 and len([r for r in all_results if any(kw in r.title.lower() or kw in r.ref.lower() for kw in keywords)]) < len(all_results) * 0.3:
                            is_fallback_search = True
                    elif keywords:
                        # Last resort: try just the first (most specific) keyword
                        first_word = sorted(keywords, key=len, reverse=True)[0]
                        search_results = api.dataset_list(search=first_word, max_size=1000, sort_by='hottest')
                        is_fallback_search = True
                
                if not search_results:
                    return f"Error: No datasets found matching '{dataset}'. Please use search_kaggle_datasets('{dataset}') to see all available options, or provide the full identifier in format 'username/dataset-name'."
                
                # Filter results to find the best match - prioritize results that contain search keywords
                original_lower = original_dataset.lower()
                search_keywords = [w for w in original_lower.split() 
                                 if len(w) > 2 and w not in ['data', 'dataset', 'datasets', 'the', 'a', 'an', 'for', 'and', 'or']]
                
                # Score each result based on how many keywords match in title/ref/description/tags
                scored_results = []
                for result in search_results:
                    title_lower = result.title.lower()
                    ref_lower = result.ref.lower()
                    description_lower = (result.description or "").lower()
                    
                    # Check tags too
                    tags_text = ""
                    if hasattr(result, 'tags') and result.tags:
                        if isinstance(result.tags, list):
                            tags_text = ' '.join([str(tag.get('name', '')) if isinstance(tag, dict) else str(tag) for tag in result.tags]).lower()
                        else:
                            tags_text = str(result.tags).lower()
                    
                    # Count keyword matches in multiple fields (weighted)
                    title_matches = sum(1 for keyword in search_keywords if keyword in title_lower)
                    ref_matches = sum(1 for keyword in search_keywords if keyword in ref_lower)
                    desc_matches = sum(1 for keyword in search_keywords if keyword in description_lower)
                    tag_matches = sum(1 for keyword in search_keywords if keyword in tags_text)
                    
                    # Weight: title and ref are most important, then description, then tags
                    total_matches = (title_matches * 3) + (ref_matches * 2) + desc_matches + (tag_matches * 0.5)
                    
                    scored_results.append((total_matches, result))
                
                # Sort by match score (descending), then by download count if available
                scored_results.sort(key=lambda x: (
                    -x[0],  # More matches first
                    -(getattr(x[1], 'download_count', 0) or 0)  # Then by popularity
                ), reverse=False)
                
                best_match = scored_results[0][1] if scored_results else search_results[0]
                match_score = scored_results[0][0] if scored_results else 0
                downloads = getattr(best_match, 'download_count', 'unknown')
                
                dataset = best_match.ref
                
                # Check dataset size before downloading
                dataset_size_bytes = getattr(best_match, 'total_bytes', None)
                
                # Safety check: if match quality is too low, don't download - return error instead
                # Since we're using weighted scores, convert back to approximate keyword count
                # The weighted score is roughly: title_matches*3 + ref_matches*2 + desc + tags*0.5
                # For a good match, we want at least 2-3 keywords in title/ref
                # If this was a fallback search (only first word), be even stricter
                if is_fallback_search:
                    min_required_score = max(4.0, len(search_keywords) * 1.5)  # Require higher score for fallback
                else:
                    min_required_score = max(3.0, len(search_keywords) * 1.0)  # Require at least 1 keyword match in title/ref
                if match_score < min_required_score:
                    return f"Error: No good matches found for '{original_dataset}'. The closest result was '{best_match.title}' ({dataset}) but it only matched {match_score}/{len(search_keywords)} keywords and doesn't seem related.\n\n" \
                           f"Please try:\n" \
                           f"1. Use search_kaggle_datasets('{original_dataset}') to see all available options\n" \
                           f"2. Provide the full dataset identifier in format 'username/dataset-name'\n" \
                           f"3. Try a different search term"
                
                # Warn if the match seems weak but still proceed (less than 70% match)
                if match_score < len(search_keywords) * 0.7:
                    result_prefix = f"WARNING: Found dataset '{dataset}' ({best_match.title}) but it may not match your search '{original_dataset}' perfectly. Only {match_score}/{len(search_keywords)} keywords matched. Proceeding with download...\n\n"
                else:
                    # Inform user which dataset will be downloaded
                    if downloads != 'unknown':
                        result_prefix = f"Found multiple datasets matching '{original_dataset}'. Using the best match: {dataset} ({best_match.title}, {downloads:,} downloads).\n\n"
                    else:
                        result_prefix = f"Found multiple datasets matching '{original_dataset}'. Using: {dataset} ({best_match.title}).\n\n"
            except Exception as e:
                error_str = str(e)
                # If search fails, return error asking for full identifier
                # Include the actual error for debugging
                return f"Error searching for dataset '{original_dataset}': {error_str}\n\n" \
                       f"Please try:\n" \
                       f"1. Use search_kaggle_datasets('{original_dataset}') to see available datasets\n" \
                       f"2. Provide the full identifier in format 'username/dataset-name' (e.g., 'heptapod/titanic')\n" \
                       f"3. Check your Kaggle credentials are set up correctly"
        
        # Check dataset size before downloading
        max_size_mb = 500  # Default limit: 500MB
        max_size_bytes = max_size_mb * 1024 * 1024
        
        # Try to get dataset size from API if we don't have it from search
        if 'dataset_size_bytes' not in locals() or dataset_size_bytes is None:
            dataset_size_bytes = None
            if '/' in dataset:
                try:
                    from kaggle.api.kaggle_api_extended import KaggleApi
                    api = KaggleApi()
                    api.authenticate()
                    # Get dataset metadata by searching for the exact ref
                    dataset_info = api.dataset_list(search=dataset.split('/')[1], user=dataset.split('/')[0])
                    if dataset_info:
                        for ds in dataset_info:
                            if ds.ref == dataset:
                                dataset_size_bytes = getattr(ds, 'total_bytes', None)
                                break
                except Exception:
                    pass  # If we can't get size, continue with download but check after
        
        # Check size before downloading (if we have size info)
        if dataset_size_bytes and dataset_size_bytes > max_size_bytes:
            size_mb = dataset_size_bytes / (1024 * 1024)
            size_gb = size_mb / 1024
            return f"Error: Dataset '{dataset}' is too large ({size_gb:.2f} GB / {size_mb:.1f} MB). Maximum allowed size is {max_size_mb} MB.\n\n" \
                   f"If you need to download this large dataset, you can:\n" \
                   f"1. Increase the size limit in the code (edit max_size_mb in download_kaggle_dataset function)\n" \
                   f"2. Download it manually from Kaggle\n" \
                   f"3. Use a different, smaller dataset"
        
        # Ensure kaggle_data directory exists (only if we haven't already)
        KAGGLE_DATA_DIR.mkdir(exist_ok=True)
        
        # Download dataset using kagglehub (downloads to cache, then we copy to kaggle_data/)
        download_path = kagglehub.dataset_download(dataset)
        
        # Copy files from download_path to kaggle_data/
        download_path_obj = Path(download_path)
        files_copied = []
        
        # Check actual downloaded size and copy files
        total_size = 0
        for item in download_path_obj.iterdir():
            if item.is_file():
                file_size = item.stat().st_size
                total_size += file_size
                # Check size before copying
                if total_size > max_size_bytes:
                    # Clean up: remove any files we already copied
                    for copied_file in files_copied:
                        copied_path = KAGGLE_DATA_DIR / copied_file
                        if copied_path.exists():
                            if copied_path.is_file():
                                copied_path.unlink()
                            else:
                                shutil.rmtree(copied_path)
                    size_mb = total_size / (1024 * 1024)
                    return f"Error: Dataset '{dataset}' is too large ({size_mb:.1f} MB downloaded so far). Maximum allowed size is {max_size_mb} MB.\n\n" \
                           f"Download stopped to prevent using too much disk space."
                dest = KAGGLE_DATA_DIR / item.name
                shutil.copy2(item, dest)
                files_copied.append(item.name)
            elif item.is_dir():
                # Calculate directory size
                dir_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                total_size += dir_size
                # Check size before copying
                if total_size > max_size_bytes:
                    # Clean up: remove any files we already copied
                    for copied_file in files_copied:
                        copied_path = KAGGLE_DATA_DIR / copied_file
                        if copied_path.exists():
                            if copied_path.is_file():
                                copied_path.unlink()
                            else:
                                shutil.rmtree(copied_path)
                    size_mb = total_size / (1024 * 1024)
                    return f"Error: Dataset '{dataset}' is too large ({size_mb:.1f} MB). Maximum allowed size is {max_size_mb} MB.\n\n" \
                           f"Download stopped to prevent using too much disk space."
                dest_dir = KAGGLE_DATA_DIR / item.name
                shutil.copytree(item, dest_dir, dirs_exist_ok=True)
                files_copied.append(item.name + "/")
        
        if not files_copied:
            return f"Dataset '{dataset}' downloaded but no files found in {download_path}"
        
        # Warn if dataset is large (but still downloaded)
        final_size_mb = total_size / (1024 * 1024)
        size_warning = ""
        if final_size_mb > 100:  # Warn if over 100MB
            size_warning = f"\n⚠️  Note: This dataset is {final_size_mb:.1f} MB in size."
        
        # Build result message (include prefix if we searched)
        result = result_prefix + f"Successfully downloaded dataset '{dataset}' to kaggle_data/\n"
        result += f"Files downloaded: {', '.join(files_copied[:10])}"  # Show first 10 files
        if len(files_copied) > 10:
            result += f" (and {len(files_copied) - 10} more files)"
        result += size_warning
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return f"Error: Kaggle authentication failed. Please set up your Kaggle API credentials using one of these methods:\n\n" \
                   f"Method 1 (Environment Variables - Recommended):\n" \
                   f"1. Go to https://www.kaggle.com/settings and create an API token\n" \
                   f"2. Add to your .env file:\n" \
                   f"   KAGGLE_USERNAME=your_username\n" \
                   f"   KAGGLE_KEY=your_api_key\n\n" \
                   f"Method 2 (kaggle.json file):\n" \
                   f"1. Go to https://www.kaggle.com/settings and create an API token\n" \
                   f"2. Place kaggle.json in ~/.kaggle/ directory\n" \
                   f"3. Set permissions: chmod 600 ~/.kaggle/kaggle.json"
        elif "403" in error_msg or "Forbidden" in error_msg:
            return f"Error: Permission denied (403 Forbidden) for dataset '{dataset}'. This usually means:\n\n" \
                   f"1. You need to accept the dataset's terms on Kaggle first:\n" \
                   f"   - Go to https://www.kaggle.com/datasets/{dataset if '/' in dataset else 'username/' + dataset}\n" \
                   f"   - Click 'New Notebook' or 'Download' to accept the terms\n" \
                   f"   - Then try downloading again\n\n" \
                   f"2. Make sure the dataset identifier is correct (format: 'username/dataset-name')\n" \
                   f"   Example: 'mmeyer/nfl-stats-2012-2024' instead of 'NFL Stats 2012-2024'"
        elif "404" in error_msg or "not found" in error_msg.lower():
            return f"Error: Dataset '{dataset}' not found. Make sure the dataset name is correct (format: 'username/dataset-name')\n" \
                   f"Example: 'mmeyer/nfl-stats-2012-2024' instead of 'NFL Stats 2012-2024'"
        else:
            return f"Error downloading dataset '{dataset}': {error_msg}\n\n" \
                   f"Tip: Make sure the dataset identifier is in format 'username/dataset-name' (e.g., 'mmeyer/nfl-stats-2012-2024')"





def plot_data(
    filename: str,
    chart_type: str,
    x_column: str,
    y_column: str = None,
    title: str = None,
    output_filename: str = None,
) -> str:
    """
    Generate a chart from a dataset and save it as a PNG image to the plots/ directory.

    Args:
        filename: Dataset file to plot (looks in raw_data/, clean_data/, or kaggle_data/)
        chart_type: Type of chart - "bar", "line", "scatter", "histogram", "pie"
        x_column: Column to use for the x-axis (or the value column for histogram/pie)
        y_column: Column to use for the y-axis (not needed for histogram or pie)
        title: Chart title. If not provided, one is generated automatically.
        output_filename: Name of the output PNG file. If not provided, one is generated automatically.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Resolve file path - bare names are searched across all data directories
    file_path = None
    try:
        if "/" in filename or "\\" in filename:
            file_path = _resolve_data_path(filename)
        else:
            for d in DATA_DIRS:
                candidate = d / filename
                if candidate.exists():
                    file_path = _resolve_data_path(filename, default_dir=d)
                    break
    except ValueError as e:
        return str(e)

    if file_path is None or not file_path.exists():
        return f"File '{filename}' not found. Checked raw_data/, clean_data/, kaggle_data/."

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            return "Unsupported file type. Use CSV or Excel files."
    except Exception as e:
        return f"Could not read file: {e}"

    # Validate columns
    if x_column not in df.columns:
        return f"Column '{x_column}' not found. Available columns: {', '.join(df.columns)}"
    if y_column and y_column not in df.columns:
        return f"Column '{y_column}' not found. Available columns: {', '.join(df.columns)}"

    chart_type = chart_type.lower().strip()
    supported = ["bar", "line", "scatter", "histogram", "pie"]
    if chart_type not in supported:
        return f"Unsupported chart type '{chart_type}'. Choose from: {', '.join(supported)}"

    # Auto-generate title and output filename
    base_name = file_path.stem
    if title is None:
        if chart_type == "histogram":
            title = f"Histogram of {x_column}"
        elif chart_type == "pie":
            title = f"Pie Chart of {x_column}"
        else:
            title = f"{chart_type.title()} Chart: {x_column} vs {y_column}"

    if output_filename is None:
        safe_title = title.lower().replace(" ", "_").replace(":", "").replace("/", "_")[:50]
        output_filename = f"{safe_title}.png"

    try:
        output_path = _safe_output_path(PLOTS_DIR, output_filename)
    except ValueError as e:
        return str(e)

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            if y_column is None:
                plt.close(fig)
                return "Bar chart requires a y_column."
            if not pd.api.types.is_numeric_dtype(df[y_column]):
                plt.close(fig)
                return f"Bar chart requires a numeric y_column. '{y_column}' is not numeric."
            plot_df = df.groupby(x_column)[y_column].mean()
            if len(plot_df) > 30:
                plot_df = plot_df.head(20)  # keep the chart readable
            ax.bar(plot_df.index.astype(str), plot_df.values)
            ax.set_xlabel(x_column)
            ax.set_ylabel(f"{y_column} (mean)")
            plt.xticks(rotation=45, ha="right")

        elif chart_type == "line":
            if y_column is None:
                plt.close(fig)
                return "Line chart requires a y_column."
            ax.plot(df[x_column], df[y_column], marker="o", linewidth=1, markersize=3)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

        elif chart_type == "scatter":
            if y_column is None:
                plt.close(fig)
                return "Scatter chart requires a y_column."
            ax.scatter(df[x_column], df[y_column], alpha=0.5, s=20)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

        elif chart_type == "histogram":
            if not pd.api.types.is_numeric_dtype(df[x_column]):
                plt.close(fig)
                return f"Histogram requires a numeric column. '{x_column}' is not numeric."
            ax.hist(df[x_column].dropna(), bins=30, edgecolor="black")
            ax.set_xlabel(x_column)
            ax.set_ylabel("Frequency")

        elif chart_type == "pie":
            counts = df[x_column].value_counts().head(10)
            ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%", startangle=140)
            ax.axis("equal")

        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        return f"Chart saved to plots/{output_filename}"

    except Exception as e:
        plt.close("all")
        return f"Error generating chart: {e}"
