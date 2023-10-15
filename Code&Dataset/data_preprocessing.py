# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 02:13:03 2023

@author: yjia_tan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load datasets
customers = pd.read_csv("olist_customers_dataset.csv")
geolocation = pd.read_csv('olist_geolocation_dataset.csv')
order_items = pd.read_csv('olist_order_items_dataset.csv')
payments = pd.read_csv('olist_order_payments_dataset.csv')
reviews = pd.read_csv('olist_order_reviews_dataset.csv')
orders = pd.read_csv('olist_orders_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
sellers = pd.read_csv('olist_sellers_dataset.csv')
product_category_translation = pd.read_csv('product_category_name_translation.csv')


# Create a dictionary for mapping from 'product_category_name' to 'product_category_name_english'
translation_map = dict(zip(product_category_translation.product_category_name, 
                           product_category_translation.product_category_name_english))

# a function to replace the 'product_category_name' in Portuguese to English
def replace_category_with_translation(df, translation_map):
    if 'product_category_name' in product_category_translation.columns:
        product_category_translation['product_category_name'].replace(
            translation_map, inplace=True)
    return product_category_translation

# ------- 1 customers
print("\nCustomer")
# Check the shape of the dataframe
print(customers.shape)
# View the first few rows of the dataframe
print(customers.head())
# Check for missing data
print(customers.isna().sum())
# Remove missing data
customers = customers.dropna()
# View descriptive statistics of the dataframe
print(customers.describe())

# ------- 2 geolocation
print("\nLocation")
print(geolocation.shape)
print(geolocation.head())
print(geolocation.isna().sum())
geolocation = geolocation.dropna()
# finding number of redundant rows
print(geolocation.duplicated().sum())
# removing redundant rows
geolocation = geolocation.drop_duplicates()

print(geolocation.describe())

# ------- 3 order_items
print("\nOrder_items")
print(order_items.shape)
print(order_items.head())
print(order_items.isna().sum())
order_items = order_items.dropna()
print(order_items.describe())

# ------- 4 payments
print("\nPayments")
print(payments.shape)
print(payments.head())
print(payments.isna().sum())
payments = payments.dropna()
print(payments.describe())

# ------- 5 reviews
print("\nReviews")
print(reviews.shape)
print(reviews.head())
print(reviews.isna().sum())
reviews = reviews.dropna()
print(reviews.describe())

# ------- 6 orders
print("\nOrders")
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
print(orders.shape)
print(orders.head())
print(orders.isna().sum())
orders = orders.dropna()
print(orders.describe())

# ------- 7 products
# Replace the 'product_category_name' values with the English translation
products['product_category_name'].replace(translation_map, inplace=True)
print("\nProduct")
print(products.shape)
print(products.head())
print(products.isna().sum())
products = products.dropna()
print(products.describe())
print(products['product_category_name'])

# ------- 8 sellers
print("\nSeller")
print(sellers.shape)
print(sellers.head())
print(sellers.isna().sum())
sellers = sellers.dropna()
print(sellers.describe())

# ------- 9 product_category_translation
print("\nproduct_category_translation")
print(product_category_translation.shape)
print(product_category_translation.head())
print(product_category_translation.isna().sum())
product_category_translation = product_category_translation.dropna()
print(product_category_translation.describe())

