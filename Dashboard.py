import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/greinald/Visual_Data_Science/refs/heads/main/Final_Data.csv')

# Streamlit layout
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
    min_value=1990,
    max_value=2020,
    value=2005,
    step=1
)

# Filter data based on selections
filtered_df = df[(df['Indicator'] == selected_indicator) & (df['Year'] == selected_year)]

# Ensure 'VALUE' is numeric
filtered_df['VALUE'] = pd.to_numeric(filtered_df['VALUE'], errors='coerce')
filtered_df = filtered_df.dropna(subset=['VALUE'])

# Check for data availability
if not filtered_df.empty:
    # Normalize values
    scaler = MinMaxScaler()
    filtered_df['Normalized_Value'] = scaler.fit_transform(filtered_df[['VALUE']])

    # Check for country mismatches (debugging)
    missing_countries = filtered_df[~filtered_df['Country'].isin(px.data.gapminder()['country'])]['Country'].unique()
    if len(missing_countries) > 0:
        st.write("Warning: The following countries may not be mapped:", missing_countries)

    # Create the map
    choropleth_fig = px.choropleth(
        filtered_df,
        locations="Country",
        locationmode="country names",
        color="Normalized_Value",
        scope="europe",
        color_continuous_scale="YlGnBu",
        title=f"Choropleth Map for {selected_indicator} ({selected_year})",
        range_color=[0, 1],  # Force range to [0, 1]
        hover_data={"Country": True, "Normalized_Value": True, "VALUE": True},  # Debugging
    )

    # Update appearance
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
    best_country = category_means.loc[category_means['Combined_Score'] == category_means['Combined_Score'].min(), 'Country'].values[0]
    worst_country = category_means.loc[category_means['Combined_Score'] == category_means['Combined_Score'].max(), 'Country'].values[0]

    # Step 8: Extract Austria's data for annotation
    austria = category_means.loc[category_means['Country'] == 'Austria'].iloc[0] if 'Austria' in category_means['Country'].values else None

    scatter_fig = px.scatter(
        category_means,
        x='Unemployment in %',
        y='VALUE',
        color='Country',
        size='VALUE',
        title=f"Scatter Plot for {selected_indicator} ({selected_year})"
    )
    
    if austria is not None:
        scatter_fig.add_annotation(
            x=austria['Unemployment in %'],
            y=austria['VALUE'],
            text=f'Austria: {austria["Country"]}',
            showarrow=True,
            arrowhead=2,
            ax=-100,
            ay=-50,
            font=dict(size=12, color="blue"),
        )

    scatter_fig.add_annotation(
        x=category_means.loc[category_means['Country'] == best_country, 'Unemployment in %'].values[0],
        y=category_means.loc[category_means['Country'] == best_country, 'VALUE'].values[0],
        text=f'Best Performer: {best_country}',
        showarrow=True,
        arrowhead=2,
        ax=-100,
        ay=-50,
        font=dict(size=12, color="green"),
    )

    scatter_fig.add_annotation(
        x=category_means.loc[category_means['Country'] == worst_country, 'Unemployment in %'].values[0],
        y=category_means.loc[category_means['Country'] == worst_country, 'VALUE'].values[0],
        text=f'Worst Performer: {worst_country}',
        showarrow=True,
        arrowhead=2,
        ay=100,
        ax=50,
        font=dict(size=10, color="red"),
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
