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
st.title('Sales Analysis')

# load dataset 
customers = pd.read_csv("olist_customers.csv")
geolocation = pd.read_csv('olist_geolocation.csv')
order_items = pd.read_csv('olist_order_items.csv')
payments = pd.read_csv('olist_payments.csv')
reviews = pd.read_csv('olist_reviews.csv')
orders = pd.read_csv('olist_orders.csv')
products = pd.read_csv('olist_products.csv')
sellers = pd.read_csv('olist_sellers.csv')

# merge dataset
order_customer_geolocation = customers.merge(orders, on='customer_id').merge(geolocation, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='inner')
order_product_info = order_items.merge(products, on='product_id').merge(order_customer_geolocation, on='order_id').merge(payments, on='order_id', how='inner')

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
    
order_product_info['product_category'] = order_product_info.product_category_name.apply(classify_cat)


orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

# Extract year and month from the order date
orders['year_month'] = orders['order_purchase_timestamp'].dt.to_period('M')

# Group by year and month and extract monthly sales data
monthly_sales = orders.groupby('year_month').size()

orders['order_year'] = orders['order_purchase_timestamp'].dt.year

# Extract month from the order date
orders['order_month'] = orders['order_purchase_timestamp'].dt.month

# Group by year and month and extract monthly sales data
monthly_sales_by_year = orders.groupby(['order_year', 'order_month']).size().unstack()

# ================================= Comparative line graph for sales trend =========================
# Plotting the line chart
plt.figure(figsize=(10, 6))

# Plot each year separately
for year in monthly_sales_by_year.index:
    plt.plot(monthly_sales_by_year.columns, monthly_sales_by_year.loc[year], marker='o', linestyle='-', label=f'Year {year}')

# Adding labels and title
plt.xlabel('Month')
plt.ylabel('Number of Sales')
plt.title('Monthly Sales Trends for Different Years')
# Adding legend
plt.legend()
# Display the plot
plt.show()

st.pyplot(plt)

# ======================================= Line graph for sales trend ==========================
# Plotting the line chart
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index.astype(str), monthly_sales.values, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Year-Month')
plt.ylabel('Number of Sales')
plt.title('Overall Monthly Sales Trends')

# Display the plot
plt.xticks(rotation=45)
plt.show()

st.pyplot(plt)

# =========================== Product Category Total Sales =============================
category_sales = order_product_info.groupby('product_category')['payment_value'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x = order_product_info.product_category.value_counts().index, y = category_sales.values, palette="turbo_r")
plt.title('Total Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45, ha='right')
plt.show()

st.pyplot(plt)

# ============== Total Orders by Day of the Week ======================================

orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["purchase_day"] = orders["order_purchase_timestamp"].dt.strftime('%A')

orders['purchase_hour'] = orders['order_purchase_timestamp'].apply(lambda x: x.hour)
hours_bins = [-0.1, 6, 12, 18, 23]
hours_labels = ['Midnight', 'Morning', 'Afternoon', 'Night']
orders['purchase_hour'] = pd.cut(orders['purchase_hour'], hours_bins, labels=hours_labels)

purchase_day_count = orders.groupby('purchase_day')['order_id'].nunique().sort_values(ascending=False)
purchase_day_count = purchase_day_count.rename("total_orders")

plt.figure(figsize=(9, 6))
sns.barplot(x=purchase_day_count.index, y=purchase_day_count.values, palette='turbo_r')
plt.title('Total Orders by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Total Orders')
plt.show()

st.pyplot(plt)

# ============== Total Orders by Parts of the Day ======================================

purchase_hour_count = orders.groupby('purchase_hour')['order_id'].nunique().sort_values(ascending=False)
purchase_hour_count = purchase_hour_count.rename("total_orders")

# Plot the bar chart using sns.barplot
plt.figure(figsize=(9, 6))
sns.barplot(x=purchase_hour_count.index, y=purchase_hour_count.values, palette='turbo_r')
plt.title('Total Orders by Parts of the Day')
plt.xlabel('Parts of the Day')
plt.ylabel('Total Orders')
plt.show()

st.pyplot(plt)

# ============== Top Product Category by Cities (Overall) ======================================
top_category_per_location = order_product_info.groupby(['customer_city', 'customer_state', 'customer_zip_code_prefix', 'geolocation_lat', 'geolocation_lng', 'product_category'])['product_id'].count().reset_index()
top_category_per_location = top_category_per_location.sort_values('product_category', ascending=False).groupby('customer_city').head(1)

import plotly.express as px

order_product_info['order_purchase_timestamp'] = pd.to_datetime(order_product_info['order_purchase_timestamp'])

# Extract year from 'order_purchase_timestamp'
order_product_info['order_year'] = order_product_info['order_purchase_timestamp'].dt.year
# convert to datetime format
order_product_info['order_year'] = order_product_info['order_purchase_timestamp'].dt.year

order_product_info_2016 = order_product_info[order_product_info['order_year'] == 2016]

order_product_info_2017 = order_product_info[order_product_info['order_year'] == 2017]

order_product_info_2018 = order_product_info[order_product_info['order_year'] == 2018]

category_sales = order_product_info.groupby(['order_year', 'product_category'])['payment_value'].sum().reset_index()

plt.figure(figsize=(16, 8))
sns.barplot(x='product_category', y='payment_value', hue='order_year', data=category_sales, palette='Set2')
plt.title('Category-wise Total Sales by Year')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Year')
plt.show()

st.pyplot(plt)

# ============== Top 10 Most Bought Product Categories by Year ======================================
most_product = order_product_info.groupby(['order_year', 'product_category']).size().reset_index(name='order_count')
most_product = most_product.groupby(['order_year', 'product_category'])['order_count'].sum().reset_index()
most_product = most_product.groupby('product_category').apply(lambda x: x.nlargest(10, 'order_count')).reset_index(drop=True)

plt.figure(figsize=(16, 8))
sns.barplot(x='product_category', y='order_count', hue='order_year', data=most_product, palette='Set2')
plt.title('Top 10 Most Bought Product Categories by Year')
plt.xlabel('Product Category')
plt.ylabel('Total Number of Orders')
plt.xticks(rotation='vertical')
plt.legend(title='Year')
plt.show()

st.pyplot(plt)


# ===================== Distribution of Sales Across Product Categories ================
order_product_info['broad_category'] = order_product_info['product_category_name'].apply(classify_cat)
category = order_product_info.broad_category.value_counts().index
count = order_product_info.broad_category.value_counts().values

plt.figure(figsize=(12, 6))

# Use the value_counts DataFrame directly
sns.barplot(x=category, y=count, palette='turbo_r')

plt.xlabel('Product Category')
plt.ylabel('Number of Sales')
plt.title('Distribution of Sales Across Product Categories')
plt.xticks(rotation=45)
plt.show()

st.pyplot(plt)

# ===================== Distribution of Sales Across Product Categories In Pie Chart ================
# Plotting Pie Chart
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
plt.axis('equal')

# Using a colormap for colors
colors = plt.cm.turbo_r(range(len(order_product_info)))

ax.pie(count, labels=category, radius=1, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('turbo_r'), labeldistance=1.20, pctdistance=1.1, wedgeprops=dict(width=1, edgecolor='w'))
plt.title('Distribution of Sales Across Product Categories')
plt.legend(category, loc='upper right', bbox_to_anchor=(1.3, 1))
plt.show()

st.pyplot(plt)

# ===================== Top Product Categories by Sales in Each City (Year-wise) ================
# Group the data for each year
top_category_per_location_per_year = order_product_info.groupby(['order_year', 'customer_city', 'customer_state', 'customer_zip_code_prefix', 'geolocation_lat', 'geolocation_lng', 'product_category'])['product_id'].count().reset_index()
top_category_per_location_per_year = top_category_per_location_per_year.sort_values('product_category', ascending=False).groupby(['order_year', 'customer_city']).head(1)

# Display the result

fig = px.scatter_geo(top_category_per_location_per_year,
                     lat='geolocation_lat',
                     lon='geolocation_lng',
                     color='product_category',
                     custom_data=['customer_city', 'product_category'],
                     animation_frame='order_year',  # Animation by year
                     projection="natural earth",
                     title='Top Product Categories by Sales in Each City (Year-wise)')

# Customize the layout and show the plot
fig.update_geos(showland=True, landcolor="#ECECEC", center=dict(lat=-14.235, lon=-51.9253), scope="south america", 
                projection_scale=1, showcoastlines=True)
# Adjust hover label appearance
fig.update_traces(textfont=dict(color='black', size=10), 
                  hovertemplate='<br>'.join([
                      '<b>%{customdata[0]}</b><br>',
                      'Latitude: %{lat:.2f}',
                      'Longitude: %{lon:.2f}',
                      'Product Category: %{customdata[1]}']),
                  hoverlabel=dict(align='left', bgcolor="white", 
                                  font=dict(color="black", size=12)))

fig.update_layout(title_text='Top Product Categories by Sales in Each City (Year-wise)', margin=dict(l=0, r=10, t=40, b=40), height=600, width=700)

st.plotly_chart(fig)

# ===================== Top Product Categories by Sales in Each State (Year-wise) ================
top_category_per_state_per_year = order_product_info.groupby(['order_year', 'customer_state', 'customer_zip_code_prefix', 'geolocation_lat', 'geolocation_lng', 'product_category'])['product_id'].count().reset_index()
top_category_per_state_per_year = top_category_per_location_per_year.sort_values('product_category', ascending=False).groupby(['order_year', 'customer_state']).head(1)

fig = px.scatter_geo(top_category_per_state_per_year,
                     lat='geolocation_lat',
                     lon='geolocation_lng',
                     color='product_category',
                     custom_data=['customer_state', 'product_category'],
                     animation_frame='order_year',  # Animation by year
                     projection="natural earth",
                     title='Top Product Categories by Sales in Each State (Year-wise)')

# Customize the layout and show the plot
fig.update_geos(showland=True, landcolor="#ECECEC", center=dict(lat=-14.235, lon=-51.9253), scope="south america", 
                projection_scale=1, showcoastlines=True)
# Adjust hover label appearance
fig.update_traces(textfont=dict(color='black', size=10), 
                  hovertemplate='<br>'.join([
                      '<b>%{customdata[0]}</b><br>',
                      'Latitude: %{lat:.2f}',
                      'Longitude: %{lon:.2f}',
                      'Product Category: %{customdata[1]}']),
                  hoverlabel=dict(align='left', bgcolor="white", 
                                  font=dict(color="black", size=12)))

fig.update_layout(title_text='Top Product Categories by Sales in Each State (Year-wise)', margin=dict(l=0, r=10, t=40, b=40), height=600, width=700)


st.plotly_chart(fig)