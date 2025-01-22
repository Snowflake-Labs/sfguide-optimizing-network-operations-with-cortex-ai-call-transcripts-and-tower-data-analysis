from snowflake.snowpark import Session
import streamlit as st
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

secrets = st.secrets["snowflake"]
session = Session.builder.configs(secrets).create()

st.markdown("## Cell Tower Performance: Failure and Success Rate Analysis")

data = session.sql("""
    SELECT
    cell_id,
    ROUND(cell_latitude, 2) AS cell_latitude, 
    ROUND(cell_longitude, 2) AS cell_longitude, 
    SUM(CASE WHEN call_release_code = 0 THEN 1 ELSE 0 END) AS total_success, 
    COUNT(*) AS total_calls, 
    ROUND((SUM(CASE WHEN call_release_code != 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) AS failure_rate, 
    ROUND((SUM(CASE WHEN call_release_code = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) AS success_rate
FROM TELCO_NETWORK_OPTIMIZATION_PROD.RAW.CELL_TOWER
GROUP BY cell_id, cell_latitude, cell_longitude;

""").to_pandas()

# Function to generate color based on failure rate
def get_color(failure_rate):
    """Map failure rate to a color"""
    if failure_rate >= 90:
        return [255, 0, 0, 160]  # Red
    elif failure_rate >= 60:
        return [255, 255, 0, 160]  # Yellow
    else:
        return [0, 255, 0, 160]  # Green  

# Apply the color function to create a color column
data['COLOR'] = data['FAILURE_RATE'].apply(get_color)

# Find the average failure rate location
avg_failure = data.groupby(['CELL_LATITUDE', 'CELL_LONGITUDE']).agg({'FAILURE_RATE': 'mean'}).reset_index()
highest_avg_failure = avg_failure.loc[avg_failure['FAILURE_RATE'].idxmax()]

# Define Pydeck GridLayer
grid_layer = pdk.Layer(
    "GridLayer",
    id="cell_tower_grid",
    data=data,
    get_position=["CELL_LONGITUDE", "CELL_LATITUDE"],
    cell_size=2000,  # Adjust size for visual clarity (in meters)
    extruded=True,
    pickable=True,
    elevation_scale=20,  # Use failure rate for height
    get_elevation="FAILURE_RATE",
    get_fill_color="COLOR",
)

# Define the initial view state
view_state = pdk.ViewState(
    latitude=highest_avg_failure['CELL_LATITUDE'],  # Center map on highest failure rate
    longitude=highest_avg_failure['CELL_LONGITUDE'],
    zoom=8,
    pitch=50,
)

# Display in Streamlit
st.session_state.event = st.pydeck_chart(
    pdk.Deck(
        layers=[grid_layer],
        initial_view_state=view_state,
    ), on_select="rerun", selection_mode="single-object"
)

cell_tower_objects = st.session_state.event.selection.get("objects", {}).get("cell_tower_grid", [])
data = []

for obj in cell_tower_objects:
    points = obj.get("points", [])
    for point in points:
        source = point.get("source", {})
        data.append({
            "Cell ID": source.get("CELL_ID"),
            "Latitude": source.get("CELL_LATITUDE"),
            "Longitude": source.get("CELL_LONGITUDE"),
            "Failure Rate (%)": source.get("FAILURE_RATE"),
            "Success Rate (%)": source.get("SUCCESS_RATE"),
            "Total Calls": source.get("TOTAL_CALLS"),
            "Total Successful Calls": source.get("TOTAL_SUCCESS"),
        })

df = pd.DataFrame(data)

if data:
  prompt = f"""
    You are a network engineer analyzing multiple failed cells in a cell tower. 
    Provide a concise summary of the failed cells using the following data:

    {points}

    Start your response directly with: "The selected grid has". 
    Format the response in Markdown with proper bullet points. For each failed cell, use a bullet point and display the details as sub-bullets, like this example:

    - **Cell ID: 123456**
      - **Location:** Latitude 12.34, Longitude -56.78; **Failure Rate:** 10%; **Success Rate:** 90%; **Total Calls:** 1000; **Total Successful Calls:** 900

    Do not include phrases like "Based on the provided data".
    """
  
  prompt = prompt.replace("'", "''")

  selection_text = session.sql(f"select snowflake.cortex.complete('mistral-large', '{prompt}') as res").to_pandas()
  st.write("#### Selected Grid Cells")
  st.markdown(selection_text["RES"][0])

  st.write("")
  col1, col2, col3 = st.columns(3)
  # Plot 1: Bar Chart of Failure Rates

  fig1, ax1 = plt.subplots()
  df.plot(kind="bar", x="Cell ID", y="Failure Rate (%)", color="orange", ax=ax1)
  ax1.set_ylabel("Failure Rate (%)")
  ax1.set_title("Failure Rate for Each Cell")
  col1.pyplot(fig1)


  cell_ids_list = df["Cell ID"].to_list()
  cell_ids_str = ','.join(map(str, cell_ids_list))
  loyalty_data = session.sql(f"""SELECT 
        c.cell_id,
        COUNT(CASE WHEN cl.status = 'Bronze' THEN 1 END) AS bronze_count,
        COUNT(CASE WHEN cl.status = 'Silver' THEN 1 END) AS silver_count,
        COUNT(CASE WHEN cl.status = 'Gold' THEN 1 END) AS gold_count
    FROM 
        TELCO_NETWORK_OPTIMIZATION_PROD.raw.customer_loyalty cl
    JOIN 
        TELCO_NETWORK_OPTIMIZATION_PROD.RAW.CELL_TOWER c
    ON 
        cl.phone_number = c.msisdn
    WHERE 
        c.call_release_code != 0
        AND c.cell_id IN ({cell_ids_str})
    GROUP BY 
        c.cell_id;
    """).to_pandas()
  # Set 'cell_id' as the index for better visualization
  loyalty_data.set_index('CELL_ID', inplace=True)

  # Plotting the loyalty status counts
  fig2, ax2 = plt.subplots()
  loyalty_data.plot(kind='bar', stacked=True, ax=ax2, color=['#cd7f32', '#c0c0c0', '#ffd700'])

  # Customize plot
  ax2.set_title('Loyalty Status Count by Cell', fontsize=16)
  ax2.set_xlabel('Cell ID', fontsize=12)
  ax2.set_ylabel('Customer Count', fontsize=12)
  ax2.set_xticklabels(loyalty_data.index, rotation=45)
  ax2.legend(title="Loyalty Status", labels=["Bronze", "Silver", "Gold"])

  # Show the plot in Streamlit
  col2.pyplot(fig2)

  sentiment_score = session.sql(f"""SELECT 
      cell_id,
      AVG(sentiment_score) + 20 AS avg_sentiment_score
  FROM 
      TELCO_NETWORK_OPTIMIZATION_PROD.RAW.SUPPORT_TICKETS
  WHERE cell_id IN ({cell_ids_str})
  GROUP BY 
      cell_id
  ORDER BY 
      avg_sentiment_score DESC;
  """).to_pandas()

  # Create the figure and axes for plotting
  fig3, ax3 = plt.subplots()

  # Plotting the data on ax3 (not ax1, as you might have mixed it up)
  sentiment_score.plot(kind="bar", x="CELL_ID", y="AVG_SENTIMENT_SCORE", color="orange", ax=ax3)

  # Set plot labels and title
  ax3.set_ylabel("Avg Sentiment Score")
  ax3.set_title("Call Center Transcripts Sentiment Score by Cell")

  # Show the plot in Streamlit
  col3.pyplot(fig3)

  prompt = f"""
    You are a network engineer tasked with improving customer experience, adoption, and reducing call failures. \
    Based on the following data, provide recommendations on which cell to prioritize for improvements:

    1. Failure Rate for Each Cell: {df}
    2. Customer Loyalty Status by Cell (Bronze, Silver, Gold): {loyalty_data}
    3. Call Center Transcripts Sentiment Score by Cell: {sentiment_score}

    Consider the following when making recommendations:
    - Cells with high failure rates and low loyalty status may require immediate attention.
    - Cells with negative sentiment scores may indicate issues with customer experience that need addressing.
    - Focus on cells with high adoption potential and lower failure rates for optimal impact on customer experience.

    Based on this data, suggest which cell should be prioritized for fixes, the reasons for that choice.
  """

  prompt = prompt.replace("'", "''")

  suggestion = session.sql(f"select snowflake.cortex.complete('mistral-large', '{prompt}') as res").to_pandas()
  st.write("#### Suggestion from LLM:")
  st.markdown(suggestion["RES"][0])
