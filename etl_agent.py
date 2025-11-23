"""
2-Agent ETL Pipeline Generator - WITH AUTO-FIX
Uses NVIDIA Nemotron Models with automatic code validation and correction

Setup:
    1. Get API key from: https://build.nvidia.com/nvidia/nemotron-4-340b-instruct
    2. Set: set NVIDIA_API_KEY=nvapi-your-key (Windows)
    3. Install: pip install openai pandas httpx

Usage:
    python etl_agent.py
    python etl_agent.py your_data.csv
"""

import pandas as pd
import json
import os
from openai import OpenAI
from datetime import datetime
import httpx
import time
import sys
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

NEMOTRON_MODEL = "meta/llama-3.1-8b-instruct"
API_TIMEOUT = 120.0
MAX_RETRIES = 3

# ============================================================================
# CODE VALIDATOR AND AUTO-FIXER
# ============================================================================

class CodeValidator:
    """Validates and auto-fixes common issues in generated ETL code"""

    @staticmethod
    def validate_and_fix(code):
        """
        Validates generated code and applies automatic fixes

        Returns:
            tuple: (fixed_code, issues_found, fixes_applied)
        """
        issues = []
        fixes = []
        fixed_code = code

        # Issue 1: Check for locals() iteration bug
        if re.search(r'for\s+\w+\s+in\s+locals\(\):', fixed_code):
            issues.append("[CRITICAL] Using locals() instead of df.columns")
            fixed_code = re.sub(
                r'for\s+(\w+)\s+in\s+locals\(\):',
                r'for \1 in df.columns:',
                fixed_code
            )
            fixes.append("[FIXED] Changed locals() to df.columns")

        # Issue 2: Ensure cursor is defined at function start
        if 'def load(' in fixed_code:
            # Check if cursor is initialized properly
            load_func_match = re.search(r'(def load\([^)]*\):.*?)(?=\n(?:def |if __name__|$))', fixed_code, re.DOTALL)
            if load_func_match:
                load_func = load_func_match.group(1)

                # Check if cursor = None initialization exists
                if 'cursor = None' not in load_func:
                    issues.append("[CRITICAL] cursor not initialized at function start")

                    # Find the function definition line
                    func_def_match = re.search(r'(def load\([^)]*\):)\s*\n', fixed_code)
                    if func_def_match:
                        func_def = func_def_match.group(1)
                        # Add proper initialization after function definition
                        replacement = f'''{func_def}
    """Load transformed data into PostgreSQL"""
    conn = None
    cursor = None
    
    try:'''
                        fixed_code = fixed_code.replace(func_def, replacement, 1)
                        fixes.append("[FIXED] Added cursor = None initialization")

        # Issue 3: Ensure proper try-finally for cursor cleanup
        if 'def load(' in fixed_code and 'cursor.close()' in fixed_code:
            load_func_match = re.search(r'(def load\([^)]*\):.*?)(?=\n(?:def |if __name__|$))', fixed_code, re.DOTALL)
            if load_func_match:
                load_func = load_func_match.group(1)

                # Check if finally block exists
                if 'finally:' not in load_func:
                    issues.append("[WARNING] No finally block for cursor cleanup")

        # Issue 4: Check for deprecated error_bad_lines
        if 'error_bad_lines' in fixed_code:
            issues.append("[CRITICAL] Using deprecated error_bad_lines parameter")
            fixed_code = fixed_code.replace('error_bad_lines=False', "on_bad_lines='skip'")
            fixes.append("[FIXED] Replaced error_bad_lines with on_bad_lines")

        # Issue 5: Ensure df.columns is used, not COLUMN_MAPPING.values()
        if re.search(r'for\s+\w+\s+in\s+COLUMN_MAPPING\.values\(\):', fixed_code):
            issues.append("[CRITICAL] Iterating over COLUMN_MAPPING.values() instead of df.columns")
            fixes.append("[WARNING] MANUAL FIX NEEDED: Change iteration to df.columns")

        # Issue 6: Ensure execute_values is called with proper cursor
        if 'execute_values(' in fixed_code:
            # Check if cursor is passed correctly
            if not re.search(r'execute_values\(\s*cursor\s*,', fixed_code):
                issues.append("[WARNING] execute_values may not be using cursor correctly")

        # Issue 7: Check for empty try blocks (indentation errors)
        if re.search(r'try:\s*\n\s*try:', fixed_code):
            issues.append("[CRITICAL] Empty try block detected - indentation error")
            fixes.append("[WARNING] Manual fix needed for try block indentation")

        # Issue 8: Check for basic Python syntax validity
        try:
            compile(fixed_code, '<string>', 'exec')
        except SyntaxError as e:
            issues.append(f"[CRITICAL] Syntax error: {e}")
            fixes.append("[ACTION] Using fallback template due to syntax errors")

        return fixed_code, issues, fixes

    @staticmethod
    def generate_template_code(df):
        """Generate a complete, working ETL template as fallback"""

        # Build COLUMN_MAPPING from dataframe
        column_mapping = {col: col.lower().replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '').replace(')', '').replace('/', '_').replace('\\', '_')
                         for col in df.columns}

        template = f'''"""
Generated ETL Pipeline
Auto-generated with validated template
"""

import sys
import os
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import extras
import getpass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Column mapping: DataFrame columns -> SQL columns
COLUMN_MAPPING = {json.dumps(column_mapping, indent=4)}


def get_sql_type(dtype):
    """Convert pandas dtype to PostgreSQL type"""
    dtype_str = str(dtype).lower()
    
    if 'int' in dtype_str:
        return 'INTEGER'
    elif 'float' in dtype_str:
        return 'NUMERIC'
    elif 'datetime' in dtype_str:
        return 'TIMESTAMP'
    elif 'bool' in dtype_str:
        return 'BOOLEAN'
    else:
        return 'VARCHAR(255)'


def extract(file_path):
    """Extract data from CSV file with robust encoding handling"""
    logging.info(f"Extracting data from {{file_path}}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {{file_path}}")
    
    # Try multiple encodings
    encodings = ['utf-8', 'latin1', 'cp1252']
    df = None
    successful_encoding = None
    
    for encoding in encodings:
        try:
            logging.info(f"Trying encoding: {{encoding}}")
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
            successful_encoding = encoding
            logging.info(f"Successfully read CSV with {{encoding}} encoding")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.error(f"Error with {{encoding}}: {{e}}")
            continue
    
    if df is None:
        raise Exception("Failed to read CSV with common encodings")
    
    logging.info(f"Extracted {{len(df)}} rows, {{len(df.columns)}} columns")
    return df, successful_encoding


def transform(df):
    """Transform the data"""
    logging.info("Transforming data...")
    
    # Iterate over COLUMN_MAPPING keys (original DataFrame column names)
    for original_col in COLUMN_MAPPING.keys():
        if original_col in df.columns:
            # Handle missing values
            if df[original_col].isnull().any():
                if df[original_col].dtype in ['int64', 'float64']:
                    df[original_col].fillna(0, inplace=True)
                else:
                    df[original_col].fillna('', inplace=True)
    
    logging.info("Transformation complete")
    return df


def load(df, db_params):
    """Load transformed data into PostgreSQL"""
    conn = None
    cursor = None
    
    # Define column order from COLUMN_MAPPING
    df_cols = list(COLUMN_MAPPING.keys())
    sql_cols = list(COLUMN_MAPPING.values())
    
    logging.info(f"Loading {{len(df)}} rows into {{db_params['table_name']}}")
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=db_params['host'],
            database=db_params['database'],
            user=db_params['user'],
            password=db_params['password']
        )
        cursor = conn.cursor()
        logging.info("Database connection established")
        
        # Build CREATE TABLE statement
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {{db_params['table_name']}} ("
        
        # CRITICAL: Iterate over DataFrame columns (original names)
        for column in df.columns:
            sql_col = COLUMN_MAPPING[column]
            sql_type = get_sql_type(df[column].dtype)
            create_table_sql += f"{{sql_col}} {{sql_type}}, "
        
        create_table_sql = create_table_sql.rstrip(', ') + ")"
        
        logging.info(f"Creating table: {{create_table_sql}}")
        cursor.execute(create_table_sql)
        
        # Extract data in correct column order
        data_to_insert = df[df_cols].values.tolist()
        
        # Build INSERT statement
        insert_sql = f"INSERT INTO {{db_params['table_name']}} ({{','.join(sql_cols)}}) VALUES %s"
        
        # Bulk insert
        logging.info(f"Inserting {{len(data_to_insert)}} rows...")
        extras.execute_values(cursor, insert_sql, data_to_insert, page_size=1000)
        
        conn.commit()
        logging.info(f"[OK] Successfully loaded {{len(df)}} rows")
        
        return True
        
    except psycopg2.errors.InsufficientPrivilege as e:
        logging.error(f"Permission error: {{e}}")
        logging.error("Grant CREATE TABLE privileges or use 'postgres' user")
        if conn:
            conn.rollback()
        raise
    
    except Exception as e:
        logging.error(f"Load error: {{e}}")
        if conn:
            conn.rollback()
        raise
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        logging.info("Database connection closed")


def get_db_config_from_user():
    """Prompt user for database configuration"""
    print("\\n" + "="*60)
    print("Database Configuration")
    print("="*60)
    
    host = input("Enter database host (default: 'localhost'): ").strip() or 'localhost'
    database = input("Enter database name: ").strip()
    user = input("Enter database username (default: 'postgres'): ").strip() or 'postgres'
    
    # Use getpass to hide password input
    import getpass
    password = getpass.getpass("Enter database password: ")
    
    table_name = input("Enter target table name (default: 'etl_data'): ").strip() or 'etl_data'
    
    print("Note: User needs CREATE TABLE permissions on the target schema")
    
    return {{
        'host': host,
        'database': database,
        'user': user,
        'password': password,
        'table_name': table_name
    }}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generated_pipeline.py <csv_file>")
        print("Example: python generated_pipeline.py data.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Validate file exists using os.path.exists (not Path.exists)
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {{csv_file}}")
        sys.exit(1)
    
    try:
        # ETL Pipeline
        df, encoding = extract(csv_file)
        print(f"Successful encoding: {{encoding}}")
        
        df = transform(df)
        
        db_params = get_db_config_from_user()
        
        load(df, db_params)
        
        print("\\n" + "="*60)
        print("[OK] ETL Pipeline Completed Successfully!")
        print("="*60)
        
    except Exception as e:
        logging.error(f"Pipeline failed: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''

        return template

# ============================================================================
# AGENT CLASS
# ============================================================================

class ETLAgent:
    """AI-powered agent using NVIDIA Nemotron with enhanced error handling"""

    def __init__(self, name, role, api_key):
        self.name = name
        self.role = role

        http_client = httpx.Client(
            verify=False,
            timeout=httpx.Timeout(API_TIMEOUT, connect=30.0)
        )

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            http_client=http_client
        )
        print(f"[OK] {name} initialized")

    def think(self, task, context):
        """Use Nemotron AI to complete the task with retry logic"""
        print(f"\n🤖 {self.name}: {task}")

        context_str = self._prepare_context(context)

        output_format_instruction = ""
        if self.name == "Code Generator":
             output_format_instruction = (
                "The output MUST be raw Python code only. "
                "DO NOT include any introductory sentences, descriptions, or Markdown code fences (```python). "
                "The very first line of your response MUST be a Python import or a comment (#). "
            )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"   Attempt {attempt}/{MAX_RETRIES}...", end=" ")

                response = self.client.chat.completions.create(
                    model=NEMOTRON_MODEL,
                    messages=[
                        {"role": "system", "content": self.role},
                        {"role": "user", "content": f"{task}\n{output_format_instruction}\n\nContext:\n{context_str}"}
                    ],
                    temperature=0.3,
                    max_tokens=4000,
                    timeout=API_TIMEOUT
                )

                result = response.choices[0].message.content

                if result.strip().startswith('```') and result.strip().endswith('```'):
                    lines = result.strip().split('\n')
                    result = '\n'.join(lines[1:-1]).strip()

                print(f"[OK] Completed ({len(result)} chars)")
                return result

            except Exception as e:
                print(f"[X] Error: {type(e).__name__}")
                if attempt < MAX_RETRIES:
                    time.sleep(5 * attempt)
                else:
                    raise

    def _prepare_context(self, context):
        """Convert context to string for API"""
        if 'dataframe' in context:
            df = context['dataframe']
            sample_data = df.head(5).astype(str).to_dict()

            info = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'sample_data': sample_data,
                'null_counts': df.isnull().sum().to_dict(),
                'unique_counts': {col: int(df[col].nunique()) for col in df.columns}
            }
            return json.dumps(info, indent=2)
        elif 'analysis' in context:
            result = "=== SCHEMA ANALYST RECOMMENDATIONS ===\n" + context['analysis'] + "\n\n"

            if 'dataframe' in context:
                df = context['dataframe']
                result += "=== ACTUAL DATA STRUCTURE ===\n"
                result += f"Columns ({len(df.columns)}): {list(df.columns)}\n"
                result += f"\nSample:\n{df.head(3).to_string()}\n"

            return result
        else:
            return json.dumps(context, indent=2, default=str)

# ============================================================================
# ORCHESTRATOR
# ============================================================================

class ETLOrchestrator:
    """Coordinates the two agents with auto-validation"""

    def __init__(self, api_key):
        print("\n" + "="*60)
        print("Initializing 2-Agent ETL System with Auto-Fix")
        print("="*60)

        self.analyst = ETLAgent(
            name="Schema Analyst",
            role="You are a data analyst. Analyze the dataframe and provide specific cleaning recommendations.",
            api_key=api_key
        )

        self.generator = ETLAgent(
            name="Code Generator",
            role="""You are a Python ETL expert. Generate production-ready ETL code.

CRITICAL RULES:
1. NEVER use 'for column in locals():' - ALWAYS use 'for column in df.columns:'
2. Initialize cursor at start: cursor = None
3. Use try-finally for cleanup
4. Use on_bad_lines='skip' not error_bad_lines
5. Access DataFrame with original column names
6. Create COLUMN_MAPPING for DataFrame->SQL mapping
7. Iterate over df.columns for schema creation
8. Use df[list(COLUMN_MAPPING.keys())].values.tolist() for data extraction
9. NEVER prompt for file path - it comes from sys.argv[1]
10. File validation: Use os.path.exists(csv_file) NOT Path(csv_file).exists()
11. CRITICAL: Use getpass.getpass() for password input to hide it
12. Import getpass at the top: import getpass
13. DO NOT use pathlib.Path - use os.path for file operations
14. CRITICAL: Clean column names for SQL - replace ALL special chars:
    - Replace spaces, dots, dashes, slashes with underscores
    - Remove parentheses
    - Example: 'title_year.1' -> 'title_year_1'
    - Example: 'Price ($)' -> 'price_'
    - Use: col.lower().replace(' ', '_').replace('.', '_').replace('-', '_').replace('(', '').replace(')', '').replace('/', '_')

Structure: extract(), transform(), load() functions + get_db_config_from_user()
In get_db_config_from_user(), MUST use: password = getpass.getpass("Enter database password: ")
File check MUST be: if not os.path.exists(csv_file): raise error""",
            api_key=api_key
        )

        self.validator = CodeValidator()

    def run(self, dataframe):
        """Run the complete ETL pipeline generation with validation"""
        print("\n" + "="*60)
        print("Running ETL Pipeline Generation")
        print("="*60)

        # Step 1: Analysis
        print("\n[Step 1/3] Schema Analysis...")
        analysis = self.analyst.think(
            task="Analyze this dataframe. Provide specific cleaning recommendations.",
            context={'dataframe': dataframe}
        )

        # Step 2: Code Generation
        print("\n[Step 2/3] Code Generation...")
        code = self.generator.think(
            task=f"""Generate complete ETL code. Must include:
- COLUMN_MAPPING dictionary
- extract(file_path) returning (df, encoding)
- transform(df) implementing recommendations  
- load(df, db_params) with proper cursor handling
- get_db_config_from_user() for interactive input
- Use df.columns for iteration, NOT locals()

Analysis: {analysis[:500]}...""",
            context={'analysis': analysis, 'dataframe': dataframe}
        )

        # Step 3: Validate and Fix
        print("\n[Step 3/3] Code Validation...")
        fixed_code, issues, fixes = self.validator.validate_and_fix(code)

        if issues:
            print("\n[!] Issues detected in generated code:")
            for issue in issues:
                print(f"   {issue}")

        if fixes:
            print("\n[OK] Auto-fixes applied:")
            for fix in fixes:
                print(f"   {fix}")

        # If critical issues remain OR syntax errors, use template
        if any("CRITICAL" in issue or "Syntax error" in issue for issue in issues):
            print("\n[!] Critical issues or syntax errors - using validated template")
            fixed_code = self.validator.generate_template_code(dataframe)
            print("[OK] Generated code from validated template")

        return {
            'analysis': analysis,
            'code': fixed_code,
            'validation': {
                'issues': issues,
                'fixes': fixes
            }
        }

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    print("""
╔══════════════════════════════════════════════════════╗
║  2-Agent ETL Pipeline Generator (AUTO-FIX)           ║
║  Powered by NVIDIA Nemotron                          ║
╚══════════════════════════════════════════════════════╝
    """)

    api_key = os.environ.get('NVIDIA_API_KEY')
    if not api_key:
        print("[X] Error: NVIDIA_API_KEY not found")
        print("\nSet it with:")
        print("  Windows: set NVIDIA_API_KEY=nvapi-...")
        print("  Linux/Mac: export NVIDIA_API_KEY='nvapi-...'")
        return

    print(f"[OK] API Key found")

    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"Loading: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
        except UnicodeDecodeError:
            print("Trying alternative encodings...")
            for encoding in ['latin1', 'cp1252']:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    print(f"[OK] Success with {encoding}")
                    break
                except:
                    continue
        except Exception as e:
            print(f"[X] Error: {e}")
            return
    else:
        print("Using sample data")
        df = create_sample_data()

    print(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns")

    # Run pipeline
    try:
        orchestrator = ETLOrchestrator(api_key)
        results = orchestrator.run(df)

        # Save results
        print("\n" + "="*60)
        print("Saving Results")
        print("="*60)

        os.makedirs('etl_output', exist_ok=True)

        with open('etl_output/analysis.txt', 'w') as f:
            f.write(results['analysis'])
        print("[OK] Saved: etl_output/analysis.txt")

        with open('etl_output/generated_pipeline.py', 'w') as f:
            f.write(results['code'])
        print("[OK] Saved: etl_output/generated_pipeline.py")

        # Show validation results
        if results['validation']['issues'] or results['validation']['fixes']:
            with open('etl_output/validation_report.txt', 'w', encoding='utf-8') as f:
                f.write("VALIDATION REPORT\n")
                f.write("="*60 + "\n\n")
                f.write("Issues Found:\n")
                for issue in results['validation']['issues']:
                    f.write(f"  {issue}\n")
                f.write("\nFixes Applied:\n")
                for fix in results['validation']['fixes']:
                    f.write(f"  {fix}\n")
            print("[OK] Saved: etl_output/validation_report.txt")

        print("\n" + "="*60)
        print("[SUCCESS]")
        print("="*60)
        print("\nReady to run:")
        print("  cd etl_output")
        print("  python generated_pipeline.py your_data.csv")

    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()

def create_sample_data():
    """Create sample data"""
    import numpy as np
    np.random.seed(42)

    data = {
        'order_id': range(1, 101),
        'customer_name': [f'Customer_{i}' for i in range(100)],
        'product': np.random.choice(['Laptop', 'Phone', 'Tablet'], 100),
        'quantity': np.random.randint(1, 5, 100),
        'price': np.random.uniform(100, 2000, 100).round(2),
        'order_date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'region': np.random.choice(['North', 'South', 'East', None], 100),
    }

    return pd.DataFrame(data)

if __name__ == "__main__":
    main()
