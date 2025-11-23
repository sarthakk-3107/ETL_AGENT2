# 🤖 ETL_AGENT2: AI-Powered ETL Pipeline Generator

> **Transform messy CSVs into production-ready PostgreSQL tables with zero manual coding**

An intelligent 2-agent system that automatically analyzes your data, generates production-ready ETL code, and loads it to PostgreSQL - all powered by NVIDIA Nemotron LLM.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-316192.svg)](https://www.postgresql.org/)

---

## 🎯 What It Does

One command turns your messy CSV into a clean PostgreSQL table:

```bash
python etl_agent.py your_data.csv
```

**What happens:**
1. 🔍 **Schema Analyst Agent** examines your data for quality issues
2. ⚙️ **Code Generator Agent** writes a complete ETL pipeline
3. ✅ **Auto-validator** catches and fixes AI mistakes
4. 📊 **Result:** Working Python pipeline + data loaded to PostgreSQL

---

## ✨ Key Features

### 🧠 Intelligent Analysis
- Detects missing values, duplicates, and data quality issues
- Identifies data types and patterns
- Recommends specific cleaning transformations

### 🔧 Production-Ready Code Generation
- **Robust encoding handling** (UTF-8, Latin1, CP1252)
- **SQL injection prevention** with parameterized queries
- **Bulk loading optimization** using `psycopg2.extras.execute_values()`
- **Automatic schema generation** (pandas dtypes → PostgreSQL types)
- **Column name sanitization** (handles special characters like dots, spaces)

### 🛡️ Self-Healing Validation Layer
- Catches syntax errors and auto-fixes them
- Detects invalid SQL column names and sanitizes them
- Ensures cursor initialization and proper error handling
- Falls back to validated template if AI makes critical mistakes

### 🔒 Security Features
- Password masking with `getpass`
- API key management
- Secure database connections

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 13+
- NVIDIA API key ([Get one free here](https://build.nvidia.com/))

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sarthakk-3107/ETL_AGENT2.git
cd ETL_AGENT2
```

2. **Install dependencies:**
```bash
pip install openai pandas httpx psycopg2-binary
```

3. **Set up your NVIDIA API key:**

**Windows (CMD):**
```bash
set NVIDIA_API_KEY=nvapi-your-key-here
```

**Windows (PowerShell):**
```powershell
$env:NVIDIA_API_KEY='nvapi-your-key-here'
```

**Linux/Mac:**
```bash
export NVIDIA_API_KEY='nvapi-your-key-here'
```

4. **Set up PostgreSQL:**
- Install PostgreSQL locally or use a cloud instance
- Create a database for your data
- Ensure your user has CREATE TABLE permissions

---

## 📖 Usage

### Basic Usage

```bash
python etl_agent.py your_data.csv
```

The agent will:
1. Analyze your CSV structure
2. Generate a complete ETL pipeline
3. Save the code to `etl_output/generated_pipeline.py`
4. Save analysis to `etl_output/analysis.txt`

### Run the Generated Pipeline

```bash
cd etl_output
python generated_pipeline.py ../your_data.csv
```

You'll be prompted for:
- Database host (default: localhost)
- Database name
- Username (default: postgres)
- Password (hidden input)
- Target table name (default: etl_data)

### Example with Sample Data

```bash
# Run with built-in sample data
python etl_agent.py
```

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Input CSV     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Schema Analyst Agent   │  ← Analyzes data quality
│  (NVIDIA Nemotron)      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Code Generator Agent   │  ← Writes ETL code
│  (NVIDIA Nemotron)      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Validation Layer      │  ← Auto-fixes mistakes
│   (Rule-based)          │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Generated Pipeline     │  ← Ready to run!
│  (Python + PostgreSQL)  │
└─────────────────────────┘
```

---

## 🔧 Configuration

Edit `etl_agent.py` to customize:

### Change the LLM Model

```python
# Line 30-33
NEMOTRON_MODEL = "meta/llama-3.1-8b-instruct"  # Current (fast, free)
# NEMOTRON_MODEL = "meta/llama-3.1-70b-instruct"  # More powerful
# NEMOTRON_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"  # Best quality
```

### Adjust Timeouts

```python
# Line 36-37
API_TIMEOUT = 120.0  # Increase if requests time out
MAX_RETRIES = 3      # Number of retry attempts
```

---

## 📂 Project Structure

```
ETL_AGENT2/
│
├── etl_agent.py              # Main agent orchestrator
├── etl_output/               # Generated files
│   ├── generated_pipeline.py # Your ETL code
│   ├── analysis.txt          # Data analysis report
│   └── validation_report.txt # Issues found & fixed
│
└── README.md                 # This file
```

---

## 🎓 How It Works

### 1. Schema Analysis Phase
The **Schema Analyst Agent** examines your data:
- Checks column types and distributions
- Identifies missing values and duplicates
- Detects data quality issues
- Recommends cleaning steps

### 2. Code Generation Phase
The **Code Generator Agent** writes Python code with:
- `extract(file_path)`: Robust CSV reading with encoding detection
- `transform(df)`: Implements analyst recommendations
- `load(df, db_params)`: Bulk loads to PostgreSQL
- `get_db_config_from_user()`: Interactive database configuration

### 3. Validation Phase
The **Auto-validator** checks for:
- ✅ Syntax errors (compiles the code)
- ✅ Common LLM mistakes (locals() iteration, cursor scope)
- ✅ Invalid SQL identifiers (dots in column names)
- ✅ Security issues (password handling)

If issues are found, it either auto-fixes or uses a validated template.

---

## 🛠️ Generated Pipeline Features

The ETL code you get includes:

### Extract Function
```python
def extract(file_path):
    """Handles multiple encodings automatically"""
    encodings = ['utf-8', 'latin1', 'cp1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
            return df, encoding
        except UnicodeDecodeError:
            continue
```

### Transform Function
```python
def transform(df):
    """Implements analyst recommendations"""
    # Handles missing values
    # Removes duplicates
    # Type conversions
    # Custom cleaning logic
    return df
```

### Load Function
```python
def load(df, db_params):
    """Optimized PostgreSQL bulk loading"""
    # Creates optimized schema
    # Sanitizes column names
    # Bulk insert with execute_values()
    # Proper error handling & cleanup
```

---

## 🐛 Troubleshooting

### API Connection Issues

**Error:** `Connection failed`

**Solutions:**
1. Check your API key is valid
2. Test network access: `https://integrate.api.nvidia.com/v1/models`
3. Try disabling VPN/antivirus temporarily
4. Try different model (edit `NEMOTRON_MODEL`)

### Database Connection Issues

**Error:** `psycopg2.OperationalError`

**Solutions:**
1. Verify PostgreSQL is running: `psql -h localhost -U postgres`
2. Check credentials (host, database, username, password)
3. Ensure database exists: `CREATE DATABASE mydatabase;`
4. Grant permissions: `GRANT CREATE ON DATABASE mydatabase TO your_user;`

### Unicode/Encoding Errors

**Error:** `'charmap' codec can't encode character`

**Solutions:**
- Already handled! The system uses UTF-8 encoding for file writes
- If you see this in generated code, regenerate with latest version

### Column Name Errors

**Error:** `syntax error at or near "."`

**Solutions:**
- Already handled! Validator sanitizes column names automatically
- Special characters (dots, spaces, parentheses) are replaced with underscores

---

## 📊 Example Output

### Input CSV
```csv
Customer Id,First Name,Price ($),title_year.1
C001,John,99.99,2024
C002,Jane,149.50,2024
```

### Generated SQL Schema
```sql
CREATE TABLE etl_data (
    customer_id VARCHAR(255),
    first_name VARCHAR(255),
    price_ NUMERIC,
    title_year_1 INTEGER
);
```

### Data Loaded
```
✅ Successfully loaded 2 rows into etl_data
```

---

## 🔮 Roadmap

### Coming Soon
- [ ] **Guardrails Integration** - Enhanced AI safety and output validation
- [ ] **Streaming Data Support** - Handle real-time data ingestion
- [ ] **Data Quality Monitors** - Automated data validation rules
- [ ] **Query Optimization** - Performance tuning suggestions
- [ ] **Multi-database Support** - MySQL, SQL Server, Snowflake
- [ ] **Incremental Loads** - Update existing tables efficiently
- [ ] **Scheduling** - Automated pipeline execution

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🐛 **Report bugs** - Open an issue with details
2. 💡 **Suggest features** - Share your ideas
3. 🔧 **Submit PRs** - Fix bugs or add features
4. 📖 **Improve docs** - Help others understand the project
5. ⭐ **Star the repo** - Show your support!

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NVIDIA** for providing free API access to Nemotron models
- **PostgreSQL** for the robust open-source database
- **pandas** for powerful data manipulation
- **psycopg2** for PostgreSQL connectivity

---

## 📧 Contact

**Sarthak** - [@sarthakk-3107](https://github.com/sarthakk-3107)

**Project Link:** [https://github.com/sarthakk-3107/ETL_AGENT2](https://github.com/sarthakk-3107/ETL_AGENT2)

---

## 🌟 Show Your Support

If you found this project helpful, please consider:
- ⭐ **Starring the repository**
- 🐦 **Sharing on Twitter/LinkedIn**
- 🔄 **Forking and trying it with your data**
- 💬 **Opening issues with feedback**

---

**Built with ❤️ and 🤖 AI**
