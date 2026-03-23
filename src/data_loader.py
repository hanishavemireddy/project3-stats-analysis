# src/data_loader.py
# ──────────────────────────────────────────────────────────────
# Loads Olist Brazilian E-Commerce dataset into SQLite
# ──────────────────────────────────────────────────────────────

import pandas as pd
from sqlalchemy import create_engine
import os

# ── Constants ─────────────────────────────────────────────────
# ── Constants ─────────────────────────────────────────────────
# Build absolute paths relative to this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'ecommerce.db')
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw')

# ── File mapping ──────────────────────────────────────────────
FILES = {
    'customers':   'olist_customers_dataset.csv',
    'orders':      'olist_orders_dataset.csv',
    'order_items': 'olist_order_items_dataset.csv',
    'payments':    'olist_order_payments_dataset.csv',
    'reviews':     'olist_order_reviews_dataset.csv',
    'products':    'olist_products_dataset.csv',
    'sellers':     'olist_sellers_dataset.csv',
    'geolocation': 'olist_geolocation_dataset.csv'
}

# ──────────────────────────────────────────────────────────────
# LOAD AND CLEAN
# ──────────────────────────────────────────────────────────────
def load_and_clean():
    dfs = {}

    for name, filename in FILES.items():
        path = f'{RAW_PATH}/{filename}'
        dfs[name] = pd.read_csv(path)
        print(f'✅ Loaded {name}: {len(dfs[name]):,} rows, {dfs[name].shape[1]} columns')

    # Fix datetime columns
    date_cols = {
        'orders': [
            'order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]
    }

    for table, cols in date_cols.items():
        for col in cols:
            dfs[table][col] = pd.to_datetime(dfs[table][col], errors='coerce')

    print('\n✅ Datetime columns fixed!')
    return dfs

# ──────────────────────────────────────────────────────────────
# SAVE TO DATABASE
# ──────────────────────────────────────────────────────────────
def save_to_db(dfs):
    engine = create_engine(f'sqlite:///{DB_PATH}')

    for name, df in dfs.items():
        df.to_sql(name, engine, if_exists='replace', index=False)
        print(f'✅ Saved {name} to database')

    print(f'\n✅ Database created at {DB_PATH}')
    return engine

# ──────────────────────────────────────────────────────────────
# MAIN FUNCTION — call this from the notebook
# ──────────────────────────────────────────────────────────────
def load_data():
    '''
    Loads Olist CSVs, saves to SQLite, returns DataFrames and engine.
    Call from notebook: from src.data_loader import load_data
    '''
    if not os.path.exists(DB_PATH):
        print('── Loading raw CSVs ──')
        dfs    = load_and_clean()
        engine = save_to_db(dfs)
    else:
        print('── Loading existing database ──')
        engine = create_engine(f'sqlite:///{DB_PATH}')
        dfs    = {}
        for name in FILES.keys():
            dfs[name] = pd.read_sql(f'SELECT * FROM {name}', engine)
            print(f'✅ Loaded {name}: {len(dfs[name]):,} rows')

    return dfs, engine