import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('https://github.com/greinald/Visual_Data_Science/blob/main/Final_Data.csv')

# Streamlit app layout
st.title("Interactive Dashboard for European Data")
st.sidebar.header("Filters")

# Sidebar inputs
selected_indicator = st.sidebar.selectbox(
    "Select an Indicator",
    options=df['Indicator'].unique(),
    index=0
)

selected_year = st.sidebar.slider(
    "Select Year",
    min_value=int(df['Year'].min()),
    max_value=int(df['Year'].max()),
    value=int(df['Year'].min()),
    step=1
)

# Choropleth Map
st.subheader(f"Choropleth Map: {selected_indicator} ({selected_year})")
filtered_df = df[(df['Indicator'] == selected_indicator) & (df['Year'] == selected_year)]

if not filtered_df.empty:
    choropleth_fig = px.choropleth(
        filtered_df,
        locations="Country",
        locationmode="country names",
        color="VALUE",
        scope='europe',
        color_continuous_scale="YlGnBu",
        title=f"Choropleth Map for {selected_indicator} ({selected_year})"
    )
    choropleth_fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(choropleth_fig)
else:
    st.write("No data available for the selected indicator and year.")

# Bar Chart
st.subheader(f"Bar Chart: {selected_indicator} ({selected_year})")
bar_filtered_df = df[
    (df['Dimension'] == 'by situational context') &
    (df['Unit of measurement'] == 'Counts') &
    (df['Indicator'] == selected_indicator) &
    (df['Year'] == selected_year)
]

if not bar_filtered_df.empty:
    category_means = bar_filtered_df.groupby('Category')['VALUE'].mean().reset_index()
    category_means_sorted = category_means.sort_values(by='VALUE', ascending=True)

    bar_fig = px.bar(
        category_means_sorted,
        x='VALUE',
        y='Category',
        orientation='h',
        title=f"Average Rates by Category ({selected_indicator}) in {selected_year}",
        text_auto=True
    )
    st.plotly_chart(bar_fig)
else:
    st.write("No data available for the bar chart.")

# Scatter Plot
st.subheader(f"Scatter Plot: Unemployment Rate vs. Homicide Rate ({selected_year})")
scatter_filtered_df = df[
    (df['Unit of measurement'] == 'Rate per 100,000 population') &
    (df['Region'] == 'Europe') &
    (df['Year'] == selected_year) &
    (df['Dimension'] == 'Total') &
    (df['Indicator'] == selected_indicator)
]

if not scatter_filtered_df.empty:
    category_means = scatter_filtered_df.groupby('Country').agg(
        {'VALUE': 'mean', 'Unemployment in %': 'mean'}
    ).reset_index()

    scaler = MinMaxScaler()
    category_means['Norm_Homicide'] = scaler.fit_transform(category_means[['VALUE']])
    category_means['Norm_Unemployment'] = scaler.fit_transform(category_means[['Unemployment in %']])
    category_means['Combined_Score'] = (
        0.5 * category_means['Norm_Homicide'] +
        0.5 * category_means['Norm_Unemployment']
    )

    scatter_fig = px.scatter(
        category_means,
        x='Unemployment in %',
        y='VALUE',
        color='Country',
        size='VALUE',
        title=f"Scatter Plot for {selected_indicator} ({selected_year})"
    )
    st.plotly_chart(scatter_fig)
else:
    st.write("No data available for the scatter plot.")

# Femicide Bar Chart
st.subheader(f"Femicide Rates by Country ({selected_year})")
femicide_filtered_df = df[
    (df['Region'] == 'Europe') &
    (df['Sex'] == 'Female') &
    (df['Unit of measurement'] == 'Counts') &
    (df['Year'] == selected_year) &
    (df['Age'] == 'Total') &
    (df['Dimension'] == 'Total') &
    (df['Category'] == 'Total') &
    (df['Indicator'] == selected_indicator)
]

if not femicide_filtered_df.empty:
    femicide_totals = femicide_filtered_df.groupby('Country')['VALUE'].sum().reset_index()
    top_countries = femicide_totals.sort_values(by='VALUE', ascending=False).head(10)

    if 'Austria' not in top_countries['Country'].values:
        austria_data = femicide_totals[femicide_totals['Country'] == 'Austria']
        top_countries = pd.concat([top_countries, austria_data])

    femicide_fig = px.bar(
        top_countries,
        x='Country',
        y='VALUE',
        title=f"Femicide Rates in Top European Countries ({selected_year})",
        text_auto=True
    )
    st.plotly_chart(femicide_fig)
else:
    st.write("No data available for femicide rates.")

