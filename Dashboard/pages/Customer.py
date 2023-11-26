import squarify
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from tensorflow import keras
# to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set title for the Page
st.title('Customer Analysis')

# Load file
customers = pd.read_csv("olist_customers.csv")
geolocation = pd.read_csv('olist_geolocation.csv')
order_items = pd.read_csv('olist_order_items.csv')
payments = pd.read_csv('olist_payments.csv')
reviews = pd.read_csv('olist_reviews.csv')
orders = pd.read_csv('olist_orders.csv')
products = pd.read_csv('olist_products.csv')
sellers = pd.read_csv('olist_sellers.csv')

# merge dataset 
df_cus_seg= pd.merge(customers, orders, on="customer_id", how='inner')
df_cus_seg= df_cus_seg.merge(reviews, on="order_id", how='inner')
df_cus_seg= df_cus_seg.merge(order_items, on="order_id", how='inner')
df_cus_seg= df_cus_seg.merge(products, on="product_id", how='inner')
df_cus_seg= df_cus_seg.merge(payments, on="order_id", how='inner')
df_cus_seg= df_cus_seg.merge(sellers, on='seller_id', how='inner')

# Recategorize Product Category
def classify_cat(x):

    if x in ['office_furniture', 'furniture_decor', 'furniture_living_room', 'kitchen_dining_laundry_garden_furniture', 'bed_bath_table', 'home_comfort', 'home_comfort_2', 'home_construction', 'garden_tools', 'furniture_bedroom', 'furniture_mattress_and_upholstery']:
        return 'Furniture'
    
    elif x in ['auto', 'computers_accessories', 'musical_instruments', 'consoles_games', 'watches_gifts', 'air_conditioning', 'telephony', 'electronics', 'fixed_telephony', 'tablets_printing_image', 'computers', 'small_appliances_home_oven_and_coffee', 'small_appliances', 'audio', 'signaling_and_security', 'security_and_services']:
        return 'Electronics'
    
    elif x in ['fashio_female_clothing', 'fashion_male_clothing', 'fashion_bags_accessories', 'fashion_shoes', 'fashion_sport', 'fashion_underwear_beach', 'fashion_childrens_clothes', 'baby', 'cool_stuff', ]:
        return 'Fashion'
    
    elif x in ['housewares', 'home_confort', 'home_appliances', 'home_appliances_2', 'flowers', 'costruction_tools_garden', 'garden_tools', 'construction_tools_lights', 'costruction_tools_tools', 'luggage_accessories', 'la_cuisine', 'pet_shop', 'market_place']:
        return 'Home & Garden'
    
    elif x in ['sports_leisure', 'toys', 'cds_dvds_musicals', 'music', 'dvds_blu_ray', 'cine_photo', 'party_supplies', 'christmas_supplies', 'arts_and_craftmanship', 'art']:
        return 'Entertainment'
    
    elif x in ['health_beauty', 'perfumery', 'diapers_and_hygiene']:
        return 'Beauty & Health'
    
    elif x in ['food_drink', 'drinks', 'food']:
        return 'Food & Drinks'
    
    elif x in ['books_general_interest', 'books_technical', 'books_imported', 'stationery']:
        return 'Books & Stationery'
    
    elif x in ['construction_tools_construction', 'construction_tools_safety', 'industry_commerce_and_business', 'agro_industry_and_commerce']:
        return 'Industry & Construction'

df_cus_seg['product_category'] = df_cus_seg.product_category_name.apply(classify_cat)

df_cus_seg.product_category.value_counts()

# Convert Datetime features from Object to Datetime
df_cus_seg['order_purchase_timestamp'] = pd.to_datetime(df_cus_seg['order_purchase_timestamp'])
df_cus_seg['order_delivered_customer_date'] = pd.to_datetime(df_cus_seg['order_delivered_customer_date'])
df_cus_seg['order_estimated_delivery_date'] = pd.to_datetime(df_cus_seg['order_estimated_delivery_date'])
df_cus_seg['shipping_limit_date'] = pd.to_datetime(df_cus_seg['shipping_limit_date'])
df_cus_seg['order_delivered_carrier_date'] =pd.to_datetime(df_cus_seg['order_delivered_carrier_date'])

# Create new features to the dataframe
df_cus_seg['estimated_days'] = (df_cus_seg['order_estimated_delivery_date'].dt.date - df_cus_seg['order_purchase_timestamp'].dt.date)
df_cus_seg['arrival_days'] = (df_cus_seg['order_delivered_customer_date'].dt.date - df_cus_seg['order_purchase_timestamp'].dt.date)
df_cus_seg['shipping_days'] = (df_cus_seg['order_delivered_customer_date'].dt.date - df_cus_seg['order_delivered_carrier_date'].dt.date)
negative_shipping_days = df_cus_seg[df_cus_seg['shipping_days'] < pd.Timedelta(0)]
indices_to_drop = negative_shipping_days.index
df_cus_seg.drop(indices_to_drop, inplace=True)
df_cus_seg['seller_to_carrier_status'] = (df_cus_seg['shipping_limit_date'].dt.date - df_cus_seg['order_delivered_carrier_date'].dt.date)
df_cus_seg['seller_to_carrier_status'] = df_cus_seg['seller_to_carrier_status'].apply(lambda x: 'OnTime/Early' if x >= pd.Timedelta(0) else 'Late')
df_cus_seg['arrival_status'] = (df_cus_seg['order_estimated_delivery_date'].dt.date - df_cus_seg['order_delivered_customer_date'].dt.date)
df_cus_seg['arrival_status'] = df_cus_seg['arrival_status'].apply(lambda x: 'OnTime/Early' if x >= pd.Timedelta(0) else 'Late')

df_cus_seg[['estimated_days', 'arrival_days', 'shipping_days']].describe()

#  Remove Outliers in both features ( More than 60 days )
sixty_days = pd.Timedelta(days=60)

outlier_indices = df_cus_seg[(df_cus_seg['estimated_days'] > sixty_days) | 
                             (df_cus_seg['arrival_days'] > sixty_days) | 
                             (df_cus_seg['shipping_days'] > sixty_days)].index
df_cus_seg.drop(outlier_indices, inplace= True)
df_cus_seg.reset_index(inplace= True, drop= True)

# Function to rate the delivery time
def rates(x):

    if x in range(0, 8):
        return 'Very Fast'
    
    elif x in range(8, 16):
        return 'Fast'
    
    elif x in range(16, 25):
        return 'Neutral'
    
    elif x in range(25, 40):
        return 'Slow'
    
    else:
        return 'Very Slow'

# Create new features for rating of the delivery time
df_cus_seg['estimated_delivery_rate'] = df_cus_seg.estimated_days.apply(rates)
df_cus_seg['arrival_delivery_rate'] = df_cus_seg.arrival_days.apply(rates)
df_cus_seg['shipping_delivery_rate'] = df_cus_seg.shipping_days.apply(rates)

# ========================== EDA ===========================================
# ================================= Top 25 Customers Capacity Cities ============================
top_10_cities = df_cus_seg['customer_city'].value_counts().head(10)

# Drop unnecessary columns
orders = orders[['order_id', 'customer_id', 'order_status']] # Include only necessary columns
order_items = order_items[['order_id', 'order_item_id', 'product_id', 'price', 'freight_value']]
geolocation = geolocation[['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng' ]]  

# Merge datasets with geolocation
geo_segment= pd.merge(customers, orders, on="customer_id", how='inner')
geo_segment= geo_segment.merge(order_items, on="order_id", how='inner')
geo_segment= geo_segment.merge(products, on="product_id", how='inner')

# group customer by city
city_segments = geo_segment.groupby('customer_city')

city_metrics = city_segments.agg({
    'customer_unique_id': 'nunique',  # Count the number of unique customers
    'order_id': 'count',  # Count the number of orders
    'price': 'sum',  # Sum of the 'price' column
}).reset_index()

# Rename the columns for clarity
city_metrics.columns = ['city', 'number_of_customers', 'number_of_orders', 'total_price']
# calculate the average price per order
city_metrics['avg_order_value'] = city_metrics['total_price'] / city_metrics['number_of_orders']
# Sort data in decending order
city_metrics_sorted = city_metrics.sort_values(by='number_of_customers', ascending=False)

import plotly.express as px
import random

# Generate a list of random colors
num_categories = len(city_metrics_sorted[:25])  # Number of categories
random_colors = [random.choice(px.colors.qualitative.Plotly) for _ in range(num_categories)]

fig = px.bar(city_metrics_sorted[:25], x='city', y='number_of_customers', 
             labels={'number_of_customers': 'Number of Customers', 'city': 'City'},
             title='Top 25 Cities by Number of Customers',
             color='city',  # Set the color to the 'city' column
             color_discrete_map=dict(zip(city_metrics_sorted[:25]['city'], random_colors)))  # Assign random colors to each city
fig.update_layout(xaxis=dict(tickangle=45))

st.plotly_chart(fig)

# =================================== Customer distribution by state =============================================
import plotly.express as px

bins = [0, 50, 100, float('inf')]  # Define the bins for segmentation
labels = ['Low Value', 'Medium Value', 'High Value']  # Define labels for each segment

# Create a new column 'customer_segment' in the city_metrics DataFrame
city_metrics['customer_segment'] = pd.cut(city_metrics['avg_order_value'], bins=bins, labels=labels)

correlation_matrix = city_metrics[['number_of_customers', 'number_of_orders', 'total_price']].corr()
# group customer by state
state_segments = geo_segment.groupby('customer_state')

state_metrics = state_segments.agg({
    'customer_unique_id': 'nunique',  # Count the number of unique customers
    'order_id': 'count',  # Count the number of orders
    'price': 'sum',  # Sum of the 'price' column
}).reset_index()

state_metrics.columns = ['state', 'number_of_customers', 'number_of_orders', 'total_price']

state_metrics['avg_order_value'] = state_metrics['total_price'] / state_metrics['number_of_orders']

state_metrics_sorted = state_metrics.sort_values(by='number_of_customers', ascending=False)

fig = px.bar(state_metrics_sorted[:], x='state', y='number_of_customers', 
             labels={'number_of_customers': 'Number of Customers', 'city': 'State'},
             title='Number of Customers according State',
             color='state',  # Set the color to the 'city' column
             color_discrete_map=dict(zip(state_metrics_sorted[:25]['state'], random_colors)))
fig.update_layout(xaxis=dict(tickangle=45))

st.plotly_chart(fig)

# ============================================ Customer behaviour Segmentation =============================
from datetime import datetime 

df_segmentation = df_cus_seg.copy()

max_trans_date = max(df_segmentation.order_purchase_timestamp).date()

df_cluster = df_segmentation[['freight_value', 'price', 'payment_value', 'payment_installments']]

rfm_table = df_segmentation.groupby('customer_unique_id').agg({'order_purchase_timestamp': lambda x:(datetime.strptime(str(max_trans_date),'%Y-%m-%d') - x.max()).days,
                                                                'order_id': lambda x:len(x),
                                                             'payment_value': lambda x:sum(x)})

rfm_table.rename(columns={'order_purchase_timestamp':'Recency','order_id':'Frequancy','payment_value':'Monetary'}, inplace=True)

rfm_table['r_score'] = pd.qcut(rfm_table['Recency'], 4, ['4','3','2','1'])
rfm_table['f_score'] = pd.qcut(rfm_table['Frequancy'].rank(method= 'first'), 4, ['1','2','3','4'])
rfm_table['m_score'] = pd.qcut(rfm_table['Monetary'], 4, ['1','2','3','4'])

rfm_table['rfm_score'] = 100 * rfm_table['r_score'].astype(int) + 10 * rfm_table['f_score'].astype(int)+ rfm_table['m_score'].astype(int)

def customer_segmenation(rfm_score):
  if rfm_score == 444:
    return 'VIP'
  
  elif  rfm_score >= 433 and rfm_score < 444:
    return 'Very Loyal Customer'
  
  elif   rfm_score >=421 and rfm_score< 433:
    return 'Potential Loyalist Customer'
  
  elif rfm_score>=344 and rfm_score < 421:
    return 'New Customer'
  
  elif rfm_score>=323 and rfm_score<344:
    return 'Potential Customer'
  
  elif rfm_score>=224 and rfm_score<311:
    return 'At-Risk Customers' 
  
  else:
    return 'Lost customers'       
  
rfm_table['customer_segmentation'] = rfm_table['rfm_score'].apply(customer_segmenation)


for i in [0, 2]:
    outlier_indices = []
    col = rfm_table.columns[i]
    percentile_5 = np.percentile(rfm_table[col], 5)
    percentile_95 = np.percentile(rfm_table[col], 95)
    outlier_indices.append(rfm_table[(rfm_table[col] < percentile_5) | (rfm_table[col] > percentile_95)].index)

rfm_table.drop(outlier_indices[0][:], inplace= True)
rfm_table.reset_index(inplace= True, drop= True)

# Assuming Sizes and labels are defined
plt.rc('font', size=10)
Sizes = rfm_table.groupby('customer_segmentation')[['Monetary']].count()

# Choose a color palette
color_palette = sns.color_palette("turbo_r", n_colors=len(Sizes))

squarify.plot(sizes=Sizes.values, label=Sizes.index, color=color_palette, alpha=.55)
plt.suptitle("Customer Segmentation Grid", fontsize=15)
plt.axis('off')  # Turn off axis labels
plt.show()

st.pyplot(plt)