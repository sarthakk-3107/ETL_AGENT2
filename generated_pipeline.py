"""
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
COLUMN_MAPPING = {
    "movie_title": "movie_title",
    "num_critic_for_reviews": "num_critic_for_reviews",
    "duration": "duration",
    "DIRECTOR_facebook_likes": "director_facebook_likes",
    "actor_3_facebook_likes": "actor_3_facebook_likes",
    "ACTOR_1_facebook_likes": "actor_1_facebook_likes",
    "gross": "gross",
    "num_voted_users": "num_voted_users",
    "Cast_Total_facebook_likes": "cast_total_facebook_likes",
    "facenumber_in_poster": "facenumber_in_poster",
    "num_user_for_reviews": "num_user_for_reviews",
    "budget": "budget",
    "title_year": "title_year",
    "ACTOR_2_facebook_likes": "actor_2_facebook_likes",
    "imdb_score": "imdb_score",
    "title_year.1": "title_year_1"
}


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
    logging.info(f"Extracting data from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Try multiple encodings
    encodings = ['utf-8', 'latin1', 'cp1252']
    df = None
    successful_encoding = None
    
    for encoding in encodings:
        try:
            logging.info(f"Trying encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
            successful_encoding = encoding
            logging.info(f"Successfully read CSV with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.error(f"Error with {encoding}: {e}")
            continue
    
    if df is None:
        raise Exception("Failed to read CSV with common encodings")
    
    logging.info(f"Extracted {len(df)} rows, {len(df.columns)} columns")
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
    
    logging.info(f"Loading {len(df)} rows into {db_params['table_name']}")
    
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
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {db_params['table_name']} ("
        
        # CRITICAL: Iterate over DataFrame columns (original names)
        for column in df.columns:
            sql_col = COLUMN_MAPPING[column]
            sql_type = get_sql_type(df[column].dtype)
            create_table_sql += f"{sql_col} {sql_type}, "
        
        create_table_sql = create_table_sql.rstrip(', ') + ")"
        
        logging.info(f"Creating table: {create_table_sql}")
        cursor.execute(create_table_sql)
        
        # Extract data in correct column order
        data_to_insert = df[df_cols].values.tolist()
        
        # Build INSERT statement
        insert_sql = f"INSERT INTO {db_params['table_name']} ({','.join(sql_cols)}) VALUES %s"
        
        # Bulk insert
        logging.info(f"Inserting {len(data_to_insert)} rows...")
        extras.execute_values(cursor, insert_sql, data_to_insert, page_size=1000)
        
        conn.commit()
        logging.info(f"[OK] Successfully loaded {len(df)} rows")
        
        return True
        
    except psycopg2.errors.InsufficientPrivilege as e:
        logging.error(f"Permission error: {e}")
        logging.error("Grant CREATE TABLE privileges or use 'postgres' user")
        if conn:
            conn.rollback()
        raise
    
    except Exception as e:
        logging.error(f"Load error: {e}")
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
    print("\n" + "="*60)
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
    
    return {
        'host': host,
        'database': database,
        'user': user,
        'password': password,
        'table_name': table_name
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generated_pipeline.py <csv_file>")
        print("Example: python generated_pipeline.py data.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Validate file exists using os.path.exists (not Path.exists)
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    try:
        # ETL Pipeline
        df, encoding = extract(csv_file)
        print(f"Successful encoding: {encoding}")
        
        df = transform(df)
        
        db_params = get_db_config_from_user()
        
        load(df, db_params)
        
        print("\n" + "="*60)
        print("[OK] ETL Pipeline Completed Successfully!")
        print("="*60)
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
