# New Python file
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Set random seed for reproducibility
np.random.seed(42)

# Initialize Dash App
app = Dash(__name__)

# Expose the underlying Flask server for Render (Gunicorn WSGI)
server = app.server

# =========================================================
# 1. SYNTHETIC DATA GENERATION (Lahore GIS Datasets)
# =========================================================

# EPSG Projection Constants
WGS84_CRS = "EPSG:4326"
UTM_CRS = "EPSG:32643"  # UTM Zone 43N for Punjab, Pakistan

# --- A. Candidate Helipad Sites ---
lons_hp = np.random.uniform(74.20, 74.45, 50)
lats_hp = np.random.uniform(31.40, 31.60, 50)
helipads_gdf = gpd.GeoDataFrame(
    {'site_id': [f"helipad_{i}" for i in range(50)]},
    geometry=[Point(lon, lat) for lon, lat in zip(lons_hp, lats_hp)],
    crs=WGS84_CRS
)

# --- B. Highways & Urban Zones ---
m2_motorway = LineString([(74.20, 31.40), (74.25, 31.50), (74.30, 31.62)])
ring_road = LineString([(74.25, 31.42), (74.42, 31.48), (74.40, 31.58)])
highways_gdf = gpd.GeoDataFrame(
    {'name': ['M-2 Motorway', 'Lahore Ring Road']},
    geometry=[m2_motorway, ring_road],
    crs=WGS84_CRS
)

walled_city = Polygon([(74.30, 31.55), (74.36, 31.55), (74.36, 31.60), (74.30, 31.60)])
urban_gdf = gpd.GeoDataFrame(
    {'zone': ['Walled City Urban Core']},
    geometry=[walled_city],
    crs=WGS84_CRS
)

# --- C. Crime Incidents Data ---
lons_crime = np.random.uniform(74.22, 74.42, 150)
lats_crime = np.random.uniform(31.42, 31.58, 150)
crime_types = np.random.choice(['Vehicle Theft', 'Robbery', 'Burglary', 'Assault'], size=150)
severity = np.random.choice([1, 2, 3, 4, 5], size=150)
crimes_gdf = gpd.GeoDataFrame(
    {'incident_id': [f"crime_{i}" for i in range(150)], 'crime_type': crime_types, 'severity': severity},
    geometry=[Point(lon, lat) for lon, lat in zip(lons_crime, lats_crime)],
    crs=WGS84_CRS
)

# --- D. Hospitals & Emergency Facilities ---
lons_hosp = np.random.uniform(74.25, 74.40, 15)
lats_hosp = np.random.uniform(31.43, 31.57, 15)
hospitals_gdf = gpd.GeoDataFrame(
    {'hospital_name': [f"Hospital_{i}" for i in range(15)], 'beds': np.random.randint(50, 500, 15)},
    geometry=[Point(lon, lat) for lon, lat in zip(lons_hosp, lats_hosp)],
    crs=WGS84_CRS
)

# --- E. Industrial Candidate Zones ---
ind_poly_1 = Polygon([(74.20, 31.40), (74.26, 31.40), (74.26, 31.45), (74.20, 31.45)])
ind_poly_2 = Polygon([(74.38, 31.50), (74.44, 31.50), (74.44, 31.56), (74.38, 31.56)])
industrial_gdf = gpd.GeoDataFrame(
    {'zone_name': ['Sundar Industrial Estate Region', 'Quaid-e-Azam Industrial Park Region']},
    geometry=[ind_poly_1, ind_poly_2],
    crs=WGS84_CRS
)

# =========================================================
# 2. SPATIAL PROCESSING FUNCTIONS
# =========================================================

def get_helipad_analysis():
    helipads_proj = helipads_gdf.to_crs(UTM_CRS)
    highways_proj = highways_gdf.to_crs(UTM_CRS)
    urban_proj = urban_gdf.to_crs(UTM_CRS)

    highway_buffer = highways_proj.geometry.buffer(5000).union_all()
    urban_buffer = urban_proj.geometry.buffer(1000).union_all()

    near_hw = helipads_proj.geometry.within(highway_buffer)
    out_urban = ~helipads_proj.geometry.within(urban_buffer)

    valid = helipads_proj[near_hw & out_urban].to_crs(WGS84_CRS)
    rejected = helipads_proj[~(near_hw & out_urban)].to_crs(WGS84_CRS)

    valid['lon'] = valid.geometry.x
    valid['lat'] = valid.geometry.y
    valid['Status'] = 'Suitable Site'

    rejected['lon'] = rejected.geometry.x
    rejected['lat'] = rejected.geometry.y
    rejected['Status'] = 'Excluded Site'

    df = pd.concat([valid, rejected])
    
    fig = px.scatter_mapbox(
        df, lat="lat", lon="lon", color="Status", hover_name="site_id",
        color_discrete_map={'Suitable Site': 'blue', 'Excluded Site': 'gray'},
        zoom=10, center={"lat": 31.5204, "lon": 74.3587}, mapbox_style="open-street-map"
    )
    fig.update_traces(marker=dict(size=10))

    # Add Exclusion Polygon
    for _, row in urban_gdf.iterrows():
        lons, lats = row.geometry.exterior.xy
        fig.add_trace(go.Scattermapbox(
            mode="lines", lon=list(lons), lat=list(lats), fill="toself",
            fillcolor="rgba(255, 0, 0, 0.25)", line=dict(width=2, color="red"),
            name="Urban Exclusion Core"
        ))

    # Add Highway Lines
    for _, row in highways_gdf.iterrows():
        lons, lats = row.geometry.xy
        fig.add_trace(go.Scattermapbox(
            mode="lines", lon=list(lons), lat=list(lats),
            line=dict(width=4, color="black"), name=row['name']
        ))

    fig.update_layout(title="<b>Helipad Site Suitability Analysis</b>", margin=dict(r=0, l=0, b=0, t=40))
    metrics = {
        "total": len(df),
        "valid": len(valid),
        "rejected": len(rejected),
        "desc": "Sites within 5km of highways and outside urban buffer zones."
    }
    return fig, metrics


def get_crime_analysis():
    df = crimes_gdf.copy()
    df['lon'] = df.geometry.x
    df['lat'] = df.geometry.y

    fig = px.density_mapbox(
        df, lat='lat', lon='lon', z='severity', radius=15,
        center={"lat": 31.5204, "lon": 74.3587}, zoom=10,
        mapbox_style="open-street-map", title="Crime Density Heatmap"
    )
    fig.update_layout(margin=dict(r=0, l=0, b=0, t=40))

    metrics = {
        "total": len(df),
        "high_severity": len(df[df['severity'] >= 4]),
        "top_crime": df['crime_type'].mode()[0],
        "desc": "Density heatmap calculated from reported incidents weighted by severity."
    }
    return fig, metrics


def get_hospital_analysis():
    hosp_proj = hospitals_gdf.to_crs(UTM_CRS)
    buffers_3km = hosp_proj.geometry.buffer(3000).to_crs(WGS84_CRS)

    df = hospitals_gdf.copy()
    df['lon'] = df.geometry.x
    df['lat'] = df.geometry.y

    fig = px.scatter_mapbox(
        df, lat="lat", lon="lon", hover_name="hospital_name", size="beds",
        color_discrete_sequence=['green'], zoom=10, center={"lat": 31.5204, "lon": 74.3587},
        mapbox_style="open-street-map"
    )

    for poly in buffers_3km:
        lons, lats = poly.exterior.xy
        fig.add_trace(go.Scattermapbox(
            mode="lines", lon=list(lons), lat=list(lats), fill="toself",
            fillcolor="rgba(0, 255, 0, 0.15)", line=dict(width=1, color="green"),
            name="3km Coverage Buffer"
        ))

    fig.update_layout(title="<b>Hospital Emergency Access & Coverage</b>", margin=dict(r=0, l=0, b=0, t=40))

    metrics = {
        "total": len(df),
        "total_beds": int(df['beds'].sum()),
        "avg_beds": int(df['beds'].mean()),
        "desc": "3km emergency radius buffers plotted around healthcare centers."
    }
    return fig, metrics


def get_industrial_analysis():
    ind_wgs = industrial_gdf.copy()
    fig = go.Figure()

    for _, row in ind_wgs.iterrows():
        lons, lats = row.geometry.exterior.xy
        fig.add_trace(go.Scattermapbox(
            mode="lines", lon=list(lons), lat=list(lats), fill="toself",
            fillcolor="rgba(255, 165, 0, 0.35)", line=dict(width=2, color="orange"),
            name=row['zone_name']
        ))

    for _, row in highways_gdf.iterrows():
        lons, lats = row.geometry.xy
        fig.add_trace(go.Scattermapbox(
            mode="lines", lon=list(lons), lat=list(lats),
            line=dict(width=3, color="black"), name=row['name']
        ))

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=31.5204, lon=74.3587), zoom=10),
        title="<b>Industrial Zone Allocation & Highway Access</b>",
        margin=dict(r=0, l=0, b=0, t=40)
    )

    metrics = {
        "total": len(ind_wgs),
        "zones": ", ".join(ind_wgs['zone_name'].tolist()),
        "status": "Optimal Connectivity",
        "desc": "Approved industrial zones evaluated relative to freight highway routes."
    }
    return fig, metrics

# =========================================================
# 3. DASH APPLICATION LAYOUT & INTERACTIVE CALLBACKS
# =========================================================

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f6f9', 'padding': '20px'}, children=[
    
    # Title Banner
    html.Div(style={'textAlign': 'center', 'marginBottom': '20px', 'backgroundColor': '#1e293b', 'color': 'white', 'padding': '15px', 'borderRadius': '8px'}, children=[
        html.H1("Lahore Multi-Criteria GIS Analysis Dashboard", style={'margin': '0'}),
        html.P("Urban Planning, Emergency Response, and Site Suitability Intelligence System", style={'margin': '5px 0 0 0', 'color': '#94a3b8'})
    ]),

    # Controls Section
    html.Div(style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
        html.Label("Select GIS Analysis Module:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
        dcc.Dropdown(
            id='analysis-selector',
            options=[
                {'label': 'Helipad Site Suitability Analysis', 'value': 'helipad'},
                {'label': 'Crime Hotspot & Density Analysis', 'value': 'crime'},
                {'label': 'Hospital Access & Emergency Coverage', 'value': 'hospital'},
                {'label': 'Industrial Zone Allocation', 'value': 'industrial'}
            ],
            value='helipad',
            clearable=False,
            style={'marginTop': '10px'}
        )
    ]),

    # Main Content Area
    html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}, children=[
        
        # Interactive Map View
        html.Div(style={'flex': '3', 'minWidth': '600px', 'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
            dcc.Graph(id='main-gis-map', style={'height': '650px'})
        ]),

        # Side Panel Metrics & Summary
        html.Div(style={'flex': '1', 'minWidth': '300px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
            html.H3("Analysis Insights", style={'borderBottom': '2px solid #e2e8f0', 'paddingBottom': '10px', 'marginTop': '0'}),
            html.Div(id='metrics-panel'),
            html.Hr(),
            html.Div(id='analysis-description', style={'color': '#475569', 'fontSize': '14px', 'lineHeight': '1.5'})
        ])
    ])
])

@app.callback(
    [Output('main-gis-map', 'figure'),
     Output('metrics-panel', 'children'),
     Output('analysis-description', 'children')],
    [Input('analysis-selector', 'value')]
)
def update_dashboard(selected_module):
    if selected_module == 'helipad':
        fig, metrics = get_helipad_analysis()
        metrics_html = html.Div([
            html.P([html.Strong("Total Candidate Sites: "), f"{metrics['total']}"]),
            html.P([html.Strong("Suitable Sites: "), html.Span(f"{metrics['valid']}", style={'color': 'green', 'fontWeight': 'bold'})]),
            html.P([html.Strong("Rejected Sites: "), html.Span(f"{metrics['rejected']}", style={'color': 'red', 'fontWeight': 'bold'})])
        ])
    elif selected_module == 'crime':
        fig, metrics = get_crime_analysis()
        metrics_html = html.Div([
            html.P([html.Strong("Total Incidents Logged: "), f"{metrics['total']}"]),
            html.P([html.Strong("High Severity Incidents: "), html.Span(f"{metrics['high_severity']}", style={'color': 'red', 'fontWeight': 'bold'})]),
            html.P([html.Strong("Most Frequent Crime: "), f"{metrics['top_crime']}"])
        ])
    elif selected_module == 'hospital':
        fig, metrics = get_hospital_analysis()
        metrics_html = html.Div([
            html.P([html.Strong("Total Hospitals: "), f"{metrics['total']}"]),
            html.P([html.Strong("Total Bed Capacity: "), html.Span(f"{metrics['total_beds']}", style={'color': 'blue', 'fontWeight': 'bold'})]),
            html.P([html.Strong("Average Beds per Hospital: "), f"{metrics['avg_beds']}"])
        ])
    elif selected_module == 'industrial':
        fig, metrics = get_industrial_analysis()
        metrics_html = html.Div([
            html.P([html.Strong("Designated Industrial Zones: "), f"{metrics['total']}"]),
            html.P([html.Strong("Zone Names: "), f"{metrics['zones']}"]),
            html.P([html.Strong("Transport Readiness: "), html.Span(f"{metrics['status']}", style={'color': 'green', 'fontWeight': 'bold'})])
        ])

    description_html = html.Div([
        html.H4("Methodology Summary", style={'marginBottom': '5px'}),
        html.P(metrics['desc'])
    ])

    return fig, metrics_html, description_html

# =========================================================
# 4. EXECUTION BLOCK FOR LOCAL TESTING & RENDER DEPLOYMENT
# =========================================================

if __name__ == '__main__':
    # Render binds the port dynamically via environmental variables
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)
