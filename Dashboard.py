import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('https://raw.githubusercontent.com/greinald/Visual_Data_Science/refs/heads/main/Final_Data.csv')


st.title("Interactive Dashboard for European Data")
st.sidebar.header("Filters")


selected_indicator = st.sidebar.selectbox(
    "Select an Indicator",
    options=df['Indicator'].unique(),
    index=0
)

selected_year = st.sidebar.slider(
    "Select Year",
    min_value=int(2000),
    max_value=int(2020),
    value=int(df['Year'].min()),
    step=1
)

filtered_data = df.loc[
    (df.Region == 'Europe') &
    (df['Unit of measurement'] == 'Rate per 100,000 population') &
    (df['Age'] == 'Total') &
    (df['Sex'] == 'Total') &
    (df['Dimension'] == 'Total') &
    
    (df['Indicator']!='Persons arrested/suspected for intentional homicide') 
]


filtered_df = filtered_data[(df['Indicator'] == selected_indicator) & (filtered_data['Year'] == selected_year)]


filtered_df['VALUE'] = filtered_df['VALUE'].astype(int)



filtered_df = filtered_df.dropna(subset=['VALUE'])


if not filtered_df.empty:

    min_value = filtered_df['VALUE'].min()
    max_value = filtered_df['VALUE'].max()


    choropleth_fig = px.choropleth(
        filtered_df,
        locations="Country",
        locationmode="country names",
        color=filtered_df['VALUE'],
        scope='europe',
        color_continuous_scale="YlGnBu",
        title=f"Choropleth Map for {selected_indicator} ({selected_year})",
        range_color=[min_value, max_value]
        
    )


    choropleth_fig.update_geos(fitbounds="locations", visible=False)
    st.write(filtered_df.head())
    st.plotly_chart(choropleth_fig)
else:
    st.write("No data available for the selected indicator and year.")


# Bar Chart for Situational Context
st.subheader(f"Bar Chart: {selected_indicator} ({selected_year})")


bar_filtered_df = df[
    (df['Dimension'] == 'by situational context') &
    (df['Unit of measurement'] == 'Counts') &
    (df['Indicator'] == selected_indicator) &
    (df['Year'] == selected_year)
]


bar_filtered_df['VALUE'] = pd.to_numeric(bar_filtered_df['VALUE'], errors='coerce')
bar_filtered_df = bar_filtered_df.dropna(subset=['VALUE'])

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

    
    bar_fig.update_traces(
        texttemplate='%{x:,.0f}k',
        textposition='outside',
        marker_color='darkblue'
    )
    bar_fig.update_yaxes(title_text='Situational Context', showgrid=False)
    bar_fig.update_xaxes(title_text='Counts (Tsd.)', showgrid=True, gridwidth=0.5, gridcolor='LightGrey')
    bar_fig.update_layout(
        title_font=dict(size=18, family='Arial, sans-serif', color='darkblue'),
        xaxis_tickangle=-45,
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=50, r=50, t=80, b=150),
        height=600,
        bargap=0.15,
        bargroupgap=0.1,
        plot_bgcolor='white'
    )

    
    st.plotly_chart(bar_fig)
else:
    
    st.write(f"No data available for {selected_indicator} in {selected_year}.")
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

st.subheader(f"Femicide Rates by Country ({selected_year})")

# Filter the data for the selected year and indicator
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

# Convert VALUE to numeric and drop rows with NaN
femicide_filtered_df['VALUE'] = pd.to_numeric(femicide_filtered_df['VALUE'], errors='coerce')
femicide_filtered_df = femicide_filtered_df.dropna(subset=['VALUE'])
femicide_filtered_df['VALUE'] = femicide_filtered_df['VALUE'].astype(int)


if not femicide_filtered_df.empty:
    # Group by country and sum values
    femicide_totals = femicide_filtered_df.groupby('Country')['VALUE'].sum().reset_index()

    # Sort by VALUE and get the top 10 countries
    top_countries = femicide_totals.sort_values(by='VALUE', ascending=False).head(10)

    # Add Austria if not in the top 10
    if 'Austria' not in top_countries['Country'].values:
        austria_data = femicide_totals[femicide_totals['Country'] == 'Austria']
        top_countries = pd.concat([top_countries, austria_data])

    # Add a 'Color' column to differentiate Austria
    top_countries['Color'] = 'Other Countries'
    top_countries.loc[top_countries['Country'] == 'Austria', 'Color'] = 'Austria'

    # Create the bar chart
    femicide_fig = px.bar(
        top_countries,
        x='Country',
        y='VALUE',
        title=f"Victims of Femicide by Count in Top 10 European Countries and Austria ({selected_year})",
        color='Color',
        color_discrete_map={'Austria': 'red', 'Other Countries': 'darkblue'},
        template='plotly_white',
        text_auto=True
    )

    # Update the bar chart appearance
    femicide_fig.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
    femicide_fig.update_yaxes(title_text='Total Femicide Rates', showgrid=True, gridwidth=0.5, gridcolor='LightGrey')
    femicide_fig.update_xaxes(title_text='Country', showgrid=False)
    femicide_fig.update_layout(
        title_font=dict(size=18, family='Arial, sans-serif', color='darkblue'),
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=50, r=50, t=80, b=150),
        height=600,
        bargap=0.15,
        bargroupgap=0.1,
        plot_bgcolor='white'
    )

    # Display the bar chart
    st.plotly_chart(femicide_fig)
else:
    st.write("No data available for femicide rates.")
