import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, no_update
import dash_leaflet as dl
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import plotly.express as px
import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


try:
    from dash_leaflet.plugins.markercluster import MarkerClusterGroup
except ImportError:
    MarkerClusterGroup = None

def cluster_group(markers):
    if MarkerClusterGroup:
        return [MarkerClusterGroup(children=markers)]
    return markers

# --- Data loading & preparation -------
def load_data():
    try:
        if os.path.exists("cleaned_airbnb_selected_cities.csv"):
            airbnb_df = pd.read_csv("cleaned_airbnb_selected_cities.csv")
        else:
            np.random.seed(42)
            airbnb_df = pd.DataFrame({
                'price': np.random.uniform(30,300,500),
                'room_type': np.random.choice(['Entire home/apt','Private room','Shared room'],500),
                'guest_rating': np.random.uniform(3.5,5.0,500),
                'cleanliness': np.random.uniform(3.0,5.0,500),
                'capacity': np.random.randint(1,8,500),
                'latitude': np.random.uniform(45,55,500),
                'longitude': np.random.uniform(-5,20,500)
            })
        if os.path.exists("cleaned_unesco_sites.csv"):
            unesco_df = pd.read_csv("cleaned_unesco_sites.csv")
        else:
            unesco_df = pd.DataFrame({
                'site_name': [f'Historic Site {i}' for i in range(50)] + [f'Cathedral {i}' for i in range(25)],
                'type': np.random.choice(['Cultural','Natural','Mixed'],75),
                'entrance_fee': np.random.uniform(0, 25, 75),  # Add entrance fee
                'latitude': np.random.uniform(45,55,75),
                'longitude': np.random.uniform(-5,20,75)
            })
        if os.path.exists("cleaned_hiking_trails.csv"):
            trails_df = pd.read_csv("cleaned_hiking_trails.csv")
        else:
            np.random.seed(42)
            trails_df = pd.DataFrame({
                'trail_name': [f'Trail {i}' for i in range(200)],
                'distance_km': np.random.uniform(1,30,200),
                'elev_gain_m': np.random.uniform(0,2000,200),
                'difficulty': np.random.choice(['Easy', 'Moderate', 'Hard'], 200, p=[0.4, 0.4, 0.2]),
                'latitude': np.random.uniform(45,55,200),
                'longitude': np.random.uniform(-5,20,200)
            })
        return airbnb_df, unesco_df, trails_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

airbnb_df, unesco_df, trails_df = load_data()
if airbnb_df is None:
    exit(1)

# Assign IDs
airbnb_df['airbnb_id'] = airbnb_df.index
unesco_df['unesco_id'] = unesco_df.index
trails_df['trail_id']  = trails_df.index

# Add culture category
unesco_df['culture_category'] = unesco_df['site_name'].apply(
    lambda x: 'Architecture' if 'cathedral' in x.lower() else 'Mythology'
)

# Add estimated time for trails
trails_df['estimated_time'] = trails_df['distance_km'] * 20  # 20 min per km

# --- Embeddings & clustering for trails ---------
def prepare_adventure_embeddings():
    np.random.seed(0)
    sample = trails_df.copy().reset_index(drop=True)
    sample['avg_airbnb_price'] = np.random.uniform(40,180,len(sample))
    sample['unesco_score']     = np.random.uniform(0,1,len(sample))
    feats  = sample[['distance_km','elev_gain_m','avg_airbnb_price','unesco_score']]
    scaled = StandardScaler().fit_transform(feats)
    perp   = min(30, len(sample)-1)
    tsne_xy= TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(scaled)
    sample[['x','y']] = tsne_xy
    ncl    = min(4,len(sample))
    sample['cluster'] = KMeans(n_clusters=ncl, random_state=0).fit_predict(scaled) if ncl>1 else 0
    return sample, scaled

tsne_df, knn_scaled = prepare_adventure_embeddings()
knn_model           = NearestNeighbors(n_neighbors=min(4,len(tsne_df))).fit(knn_scaled)

# --- Ranking model -------------
model = None
def train_ranking_model(all_listings, feedback):
    global model
    X, y = [], []
    for recs in all_listings.values():
        for l in recs:
            lid = str(l['id'])
            fb  = feedback.get(lid, {'likes':0,'dislikes':0})
            X.append([l['price'],l['guest_rating'],l['cleanliness'],l['capacity']])
            y.append(fb['likes'] - fb['dislikes'])
    if len(y) >= 5:
        model = GradientBoostingRegressor(random_state=42).fit(X, y)

def rank_airbnb_listings(listings, feedback):
    if model and listings:
        X     = [[l['price'],l['guest_rating'],l['cleanliness'],l['capacity']] for l in listings]
        preds = model.predict(X)
        return [l for _,l in sorted(zip(preds,listings), key=lambda x:-x[0])]
    scored=[]
    for l in listings:
        lid=str(l['id'])
        fb=feedback.get(lid,{'likes':0,'dislikes':0})
        sc=l['guest_rating'] + fb['likes'] - fb['dislikes']
        scored.append((sc,l))
    return [l for _,l in sorted(scored,key=lambda x:-x[0])]

# --- Build complete itinerary (enhanced) -----------------------------------
def build_itinerary(trail_fb, unesco_fb, listings, airbnb_fb, days=3):
    # 1) Trails user 👍-ed, in feedback order
    liked_tids = [int(tid) for tid, fb in trail_fb.items() if fb['likes'] > fb['dislikes']]
    liked_trails = (
        trails_df[trails_df.trail_id.isin(liked_tids)]
        .set_index('trail_id').loc[liked_tids]
        .reset_index().to_dict('records')
    )

    # 2) Sites user 👍-ed, in feedback order
    liked_uids = [int(uid) for uid, fb in unesco_fb.items() if fb['likes'] > fb['dislikes']]
    liked_sites = (
        unesco_df[unesco_df.unesco_id.isin(liked_uids)]
        .set_index('unesco_id').loc[liked_uids]
        .reset_index().to_dict('records')
    )

    # 3) Stays: first 👍-ed, then fill to 5
    liked_lids = [int(lid) for lid, fb in airbnb_fb.items() if fb['likes'] > fb['dislikes']]
    stays = [l for lid in liked_lids for l in listings if l['id']==lid]
    ranked = rank_airbnb_listings(listings, airbnb_fb)
    for l in ranked:
        if l['id'] not in liked_lids and len(stays) < 5:
            stays.append(l)

    # 4) Build days with combined activities
    itinerary = []
    for i in range(days):
        day = {"day": i+1, "activities": []}
        
        if i < len(liked_trails):
            trail = liked_trails[i]
            trail['type'] = 'trail'
            day["activities"].append(trail)
            
        if i < len(liked_sites):
            site = liked_sites[i]
            site['type'] = 'site'
            day["activities"].append(site)
            
        if i < len(stays):
            stay = stays[i]
            stay['type'] = 'stay'
            day["activities"].append(stay)
            
        itinerary.append(day)
        
    return itinerary

# --- PDF generation --------------------------------------------------------
def create_pdf(itinerary, total_budget):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Title style
    style_title = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        alignment=1,
        spaceAfter=20
    )
    
    # Day header style
    style_day = ParagraphStyle(
        'Day',
        parent=styles['Heading2'],
        spaceAfter=10
    )
    
    # Item style
    style_item = ParagraphStyle(
        'Item',
        parent=styles['BodyText'],
        spaceAfter=5
    )
    
    # Budget style
    style_budget = ParagraphStyle(
        'Budget',
        parent=styles['Heading2'],
        textColor=colors.darkgreen,
        alignment=1,
        spaceBefore=20,
        spaceAfter=10
    )
    
    content = []
    
    # Title
    content.append(Paragraph("Your Adventure Itinerary", style_title))
    
    # Itinerary days
    for day in itinerary:
        content.append(Paragraph(f"Day {day['day']}:", style_day))
        
        for activity in day["activities"]:
            if activity['type'] == 'trail':
                content.append(Paragraph(f"🥾 <b>{activity['trail_name']}</b> - {activity['distance_km']:.1f} km ({activity['estimated_time']/60:.1f} hrs)", style_item))
            
            elif activity['type'] == 'site':
                fee = activity.get('entrance_fee', 0.0)
                content.append(Paragraph(f"🏛 <b>{activity['site_name']}</b> ({activity['type']}) – Entrance: €{fee:.1f}",style_item))
            
            elif activity['type'] == 'stay':
                content.append(Paragraph(f"🏠 <b>€{int(activity['price'])} {activity['room_type']}</b> · ⭐{activity['guest_rating']:.1f}", style_item))
        
        content.append(Spacer(1, 15))
    
    # Budget summary
    content.append(Paragraph(f"Total Budget: €{total_budget:.2f}", style_budget))
    
    # Budget breakdown table
    budget_data = [["Category", "Cost (€)"]]
    categories = {
        "Accommodation": 0,
        "Activities": 0,
        "Transportation": 0,
        "Food": 0
    }
    
    # Calculate costs
    for day in itinerary:
        for activity in day["activities"]:
            if activity['type'] == 'stay':
                categories["Accommodation"] += activity['price']
            elif activity['type'] == 'site':
                categories["Activities"] += activity.get('entrance_fee', 0)
            elif activity['type'] == 'trail':
                categories["Activities"] += 5  # Small fee for trail access
    
    # Add estimated costs
    categories["Transportation"] = len(itinerary) * 20  # €20 per day for transport
    categories["Food"] = len(itinerary) * 40  # €40 per day for food
    
    for category, cost in categories.items():
        budget_data.append([category, f"{cost:.2f}"])
    
    budget_table = Table(budget_data)
    budget_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    
    content.append(budget_table)
    
    doc.build(content)
    buf.seek(0)
    return buf.read()

# --- Dash App setup --------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.config.suppress_callback_exceptions = True

room_types   = sorted(airbnb_df.room_type.unique())
site_types   = sorted(unesco_df.type.unique())
culture_cats = sorted(unesco_df.culture_category.unique())

app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dcc.Download(id='download-pdf'),

    html.H1("Plan Your Perfect Hike + Stay in 3 Clicks", className="text-center my-3"),
    html.H5("Select a trail → rate stays or click an Airbnb → click Generate Itinerary",
            className="text-center mb-4 text-muted"),
    
    # Budget Tracker
    dbc.Card([
        dbc.CardHeader("Budget Tracker", className="h4"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div("Estimated Total Budget:", className="h5"),
                    html.Div(id="total-budget", className="h2 text-success mb-3")
                ], className="text-center"),
                
                dbc.Col([
                    dbc.Progress(id="budget-progress", value=0, max=100, className="mb-2"),
                    html.Div(id="budget-breakdown", className="small")
                ])
            ]),
            
            # Trip Duration Selector
            dbc.Row([
                dbc.Col([
                    html.Label("Trip Duration (days):", className="mb-1"),
                    dcc.Slider(
                        id='trip-duration-slider',
                        min=1,
                        max=14,
                        value=3,
                        marks={i: str(i) for i in range(1, 15, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=12)
            ], className="mt-3")
        ])
    ], className="mb-4"),

    # Presets
    dbc.Row(dbc.Col([
        html.Label("Preset Mode:"),
        dcc.Dropdown(id='preset-dropdown', options=[
            {'label':'Weekend Getaway','value':'weekend'},
            {'label':'Family Trip','value':'family'}
        ], placeholder="Choose a preset")
    ], md=4), className="mb-4"),

    # Filters
    dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([
            html.Label("Max Airbnb Price (€):"),
            dcc.Slider(id='price-slider', min=20, max=500, step=10, value=150,
                       marks={100:'100',200:'200',300:'300',400:'400'})
        ], md=2),
        dbc.Col([
            html.Label("Room Type:"),
            dcc.Dropdown(id='room-type-dropdown',
                         options=[{'label':r,'value':r} for r in room_types],
                         placeholder="All")
        ], md=2),
        dbc.Col([
            html.Label("UNESCO Site Type:"),
            dcc.Dropdown(id='site-type-dropdown',
                         options=[{'label':s,'value':s} for s in site_types],
                         placeholder="All")
        ], md=2),
        dbc.Col([
            html.Label("Culture Category:"),
            dcc.Dropdown(id='culture-type-dropdown',
                         options=[{'label':c,'value':c} for c in culture_cats],
                         placeholder="All")
        ], md=2),
        dbc.Col([
            html.Label(f"Trail Distance (km): {trails_df.distance_km.min():.0f}–{trails_df.distance_km.max():.0f}"),
            dcc.RangeSlider(id='trail-distance-slider',
                            min=0, max=int(trails_df.distance_km.max())+1, step=1,
                            value=[5,25],
                            marks={i:str(i) for i in range(0,int(trails_df.distance_km.max())+1,5)})
        ], md=4),
    ])), className="mb-4"),

    # Map & sidebars
    html.Div([
        dl.Map(center=[50,10], zoom=5, style={"height":"70vh","margin":"10px"},
               children=[
                   dl.TileLayer(),
                   dl.LayerGroup(id="airbnb-layer"),
                   dl.LayerGroup(id="unesco-layer"),
                   dl.LayerGroup(id="trail-layer")
               ]),
        # Left: nearby trails & UNESCO
        html.Div([
            html.H5("Nearby Trails & UNESCO", className="mb-2"),
            html.Div(id="trail-unesco-recommendations-container")
        ], style={
            "position":"absolute","top":"100px","left":"30px","zIndex":"1000",
            "backgroundColor":"white","padding":"12px","borderRadius":"8px",
            "boxShadow":"0 4px 10px rgba(0,0,0,0.2)","width":"300px",
            "maxHeight":"80vh","overflowY":"auto"
        }),
        # Right: Airbnb recs
        html.Div([
            html.H5("Nearby Accommodations", className="mb-2"),
            html.Div(id="airbnb-recommendations-container"),
            html.Button("Generate Itinerary", id="done-button", n_clicks=0,
                        className="mt-3 w-100 btn btn-primary"),
            dbc.Button("Clear", id="clear-button", n_clicks=0,
                       className="mt-2 w-100 btn btn-warning btn-sm"),
            dbc.Modal([
                dbc.ModalHeader("🎉 Your Final Itinerary"),
                dbc.ModalBody(id="final-modal-body"),
                dbc.ModalFooter([
                    html.Button("Download PDF", id="download-button", n_clicks=0,
                                className="btn btn-secondary"),
                    dbc.Button("Close", id="close-modal",
                               className="ml-auto btn-secondary")
                ])
            ], id="final-modal", is_open=False, size="lg")
        ], style={
            "position":"absolute","top":"100px","right":"30px","zIndex":"1000",
            "backgroundColor":"white","padding":"12px","borderRadius":"8px",
            "boxShadow":"0 4px 10px rgba(0,0,0,0.2)","width":"420px",
            "maxHeight":"80vh","overflowY":"auto"
        })
    ], style={"position":"relative"}),

    html.Hr(),
    html.H3("Explore Adventure Clusters", className="text-center mb-3"),
    dcc.Graph(id="tsne-graph", config={"displayModeBar":True}),
    html.Div(id="click-output", className="text-center font-weight-bold my-2"),
    
    # Enhanced Multi-Day Itinerary Section
    dbc.Card([
        dbc.CardHeader("Multi-Day Itinerary Builder", className="h4"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div(id="itinerary-header", className="mb-3"),
                    dcc.Loading(
                        id="itinerary-loading",
                        type="circle",
                        children=html.Div(id="itinerary-display")
                    )
                ])
            ])
        ])
    ], className="mt-4", id="itinerary-section"),

    # Stores
    dcc.Store(id='filtered-trails-store'),
    dcc.Store(id='selected-trail-store'),
    dcc.Store(id='selected-airbnb-store'),
    dcc.Store(id='airbnb-listings-store'),
    dcc.Store(id='airbnb-feedback-store', data={}),
    dcc.Store(id='trail-feedback-store',   data={}),
    dcc.Store(id='unesco-feedback-store', data={}),
    dcc.Store(id='itinerary-store', data=[]),
    dcc.Store(id='budget-store', data={'total': 0, 'breakdown': {}})
], fluid=True, style={"minHeight": "100vh"})

# --- Marker factories -----------------------------------------------------
def make_trail_markers(df, highlights=None, selected=None):
    highlights = highlights or []
    markers = []
    for _,r in df.iterrows():
        sel = (r.trail_name == selected)
        hl  = (r.trail_name in highlights)
        color  = "#8B0000" if sel else "#FF8C00" if hl else "#4CAF50"
        radius = 15 if sel else 12 if hl else 6
        markers.append(dl.CircleMarker(
            center=[r.latitude,r.longitude], radius=radius,
            color=color, fillColor=color, fillOpacity=0.8,
            stroke=True, weight=2,
            children=dl.Tooltip(f"{r.trail_name} ({r.distance_km:.1f} km)"),
            id={"type":"trail-marker","trail_name":r.trail_name}
        ))
    return cluster_group(markers)

def make_airbnb_markers(df):
    markers = []
    for _,r in df.iterrows():
        markers.append(dl.Marker(
            position=[r.latitude,r.longitude],
            children=dl.Tooltip(f"€{r.price:.0f} | {r.room_type} | ⭐{r.guest_rating:.1f}"),
            id={"type":"airbnb-marker","airbnb_id":int(r.airbnb_id)},
            n_clicks=0
        ))
    return cluster_group(markers)

# --- Callbacks -------------------------------------------------------------

# 1) Unified selection (trail vs Airbnb)
@app.callback(
    Output('selected-trail-store','data'),
    Output('selected-airbnb-store','data'),
    Input({'type':'trail-marker','trail_name':ALL}, 'n_clicks'),
    Input({'type':'airbnb-marker','airbnb_id':ALL}, 'n_clicks'),
    Input('clear-button','n_clicks'),
    prevent_initial_call=True
)
def select_item(trail_clicks, airbnb_clicks, clear_all):
    trig = ctx.triggered_id
    if trig == 'clear-button':
        return None, None
    if isinstance(trig, dict) and trig.get('type')=='trail-marker':
        return {'trail_name':trig['trail_name']}, None
    if isinstance(trig, dict) and trig.get('type')=='airbnb-marker':
        aid = trig['airbnb_id']
        row = airbnb_df[airbnb_df.airbnb_id==aid].iloc[0].to_dict()
        return None, row
    return no_update, no_update

# 2) Presets → Filters
@app.callback(
    Output('price-slider','value'),
    Output('room-type-dropdown','value'),
    Output('site-type-dropdown','value'),
    Output('culture-type-dropdown','value'),
    Output('trail-distance-slider','value'),
    Input('preset-dropdown','value')
)
def apply_preset(p):
    if p=='weekend': return 200,None,None,None,[5,20]
    if p=='family':  return 300,'Entire home/apt','Cultural',None,[1,10]
    return no_update,no_update,no_update,no_update,no_update

# 3) Filters → Filtered trails
@app.callback(
    Output('filtered-trails-store','data'),
    Input('trail-distance-slider','value')
)
def filter_trails(dist_range):
    df = trails_df[
        (trails_df.distance_km>=dist_range[0]) &
        (trails_df.distance_km<=dist_range[1])
    ]
    return df.to_dict('records')

# 4 & 11) Combined: trail/t-SNE selection & feedback re-ranking
@app.callback(
    Output("click-output","children"),
    Output("trail-layer","children"),
    Output("airbnb-listings-store","data"),
    Input("tsne-graph","clickData"),
    Input("selected-trail-store","data"),
    Input("filtered-trails-store","data"),
    Input("airbnb-feedback-store","data"),
    State("airbnb-listings-store","data"),
    prevent_initial_call=True
)
def update_main_and_rerank(clickData, sel_data, filtered_data, feedback, current_listings):
    prop = ctx.triggered[0]["prop_id"]
    if prop.startswith("airbnb-feedback-store"):
        if not current_listings:
            return no_update, no_update, no_update
        train_ranking_model({'latest': current_listings}, feedback or {})
        return no_update, no_update, rank_airbnb_listings(current_listings, feedback or {})
    if not filtered_data:
        return "No trails match your criteria", [], []
    df = pd.DataFrame(filtered_data)
    selected, highlights, listings = None, [], []
    if clickData and clickData.get("points"):
        selected = clickData["points"][0]["hovertext"]
    elif sel_data:
        selected = sel_data.get("trail_name")
    if selected in tsne_df.trail_name.values:
        idx  = tsne_df[tsne_df.trail_name==selected].index[0]
        nbrs = knn_model.kneighbors([knn_scaled[idx]],
                                    n_neighbors=min(4,len(tsne_df)))[1][0]
        highlights = tsne_df.iloc[nbrs[1:]]['trail_name'].tolist() if len(nbrs)>1 else []
        lat, lon = tsne_df.iloc[idx][['latitude','longitude']]
        air = airbnb_df.copy()
        air['distance'] = np.hypot(air.latitude-lat, air.longitude-lon)
        top = air.nsmallest(10,'distance').reset_index().rename(columns={'index':'id'})
        listings = top.to_dict('records')
    return (
        f"Selected: {selected}" if selected else "Click a trail or graph point",
        make_trail_markers(df, highlights, selected),
        listings
    )

# 5) Map layers
@app.callback(
    Output("airbnb-layer","children"),
    Input("price-slider","value"),
    Input("room-type-dropdown","value")
)
def update_airbnb_layer(price,room):
    df = airbnb_df[airbnb_df.price<=price]
    if room:
        df = df[df.room_type==room]
    return make_airbnb_markers(df)

@app.callback(
    Output("unesco-layer","children"),
    Input("site-type-dropdown","value"),
    Input("culture-type-dropdown","value")
)
def update_unesco_layer(site,cult):
    df=unesco_df.copy()
    if site: df=df[df.type==site]
    if cult: df=df[df.culture_category==cult]
    markers=[
        dl.CircleMarker(center=[r.latitude,r.longitude],radius=5,
                        color="blue",fillColor="blue",fillOpacity=0.6,
                        children=dl.Tooltip(f"{r.site_name} ({r.type})"))
        for _,r in df.iterrows()
    ]
    return cluster_group(markers)

# 6) t-SNE plot
@app.callback(
    Output("tsne-graph","figure"),
    Input("filtered-trails-store","data"),
    Input("selected-trail-store","data")
)
def update_tsne(filtered_data, sel_data):
    if not filtered_data:
        return px.scatter(title="No trails available")
    vis      = tsne_df[tsne_df.trail_name.isin(pd.DataFrame(filtered_data).trail_name)].copy()
    sel_name = sel_data.get('trail_name') if sel_data else None
    vis['color'] = vis.trail_name.apply(
        lambda t: "Selected" if t==sel_name else f"Cluster {vis.loc[vis.trail_name==t,'cluster'].iloc[0]}"
    )
    fig=px.scatter(vis,x='x',y='y',color='color',hover_name='trail_name',
                   hover_data={'distance_km':':.1f','elev_gain_m':':.0f'},
                   title="Adventure Clusters")
    fig.update_traces(marker=dict(size=10)); fig.update_layout(height=500)
    return fig

# 7a) Right: Airbnb recs
@app.callback(
    Output("airbnb-recommendations-container","children"),
    Input('airbnb-listings-store','data'),
    Input('airbnb-feedback-store','data')
)
def update_airbnb_recs(listings, feedback):
    if not listings:
        return html.Div("Select a trail to see nearby accommodations")
    fb     = feedback or {}
    train_ranking_model({'latest':listings},fb)
    ranked = rank_airbnb_listings(listings,fb)
    rows   = []
    for l in ranked:
        lid=str(l['id'])
        la = fb.get(lid,{}).get('likes',0) > fb.get(lid,{}).get('dislikes',0)
        da = fb.get(lid,{}).get('dislikes',0) > fb.get(lid,{}).get('likes',0)
        like_btn=html.Button("👍",id={'type':'like-btn','index':lid},
                            style={"border":f"2px solid {'#28a745' if la else '#ccc'}",
                                   "background":"#28a745" if la else "white",
                                   "color":"white" if la else "#28a745","cursor":"pointer",
                                   "padding":"5px","borderRadius":"4px"})
        dislike_btn=html.Button("👎",id={'type':'dislike-btn','index':lid},
                                style={"border":f"2px solid {'#dc3545' if da else '#ccc'}",
                                       "background":"#dc3545" if da else "white",
                                       "color":"white" if da else "#dc3545","cursor":"pointer",
                                       "padding":"5px","borderRadius":"4px"})
        link=f"https://www.google.com/maps?q={l['latitude']},{l['longitude']}"
        rows.append(html.Tr([
            html.Td(f"€{int(l['price'])}"),html.Td(l['room_type']),
            html.Td(f"{l['guest_rating']:.1f}"),html.Td(f"{l['cleanliness']:.1f}"),
            html.Td(int(l['capacity'])),html.Td(html.A("📍 Map",href=link,target="_blank")),
            html.Td(like_btn),html.Td(dislike_btn),
        ]))
    header=html.Thead(html.Tr([
        html.Th("💶 Price"),html.Th("🛏️ Type"),html.Th("⭐ Rating"),
        html.Th("🧼 Cleanliness"),html.Th("👥 Guests"),
        html.Th("📍 Map"),html.Th("👍 Like"),html.Th("👎 Dislike"),
    ]))
    return dbc.Table([header,html.Tbody(rows)], bordered=True, hover=True, size="sm")

# 7b) Left: Trails & UNESCO around clicked Airbnb, with 👍👎
@app.callback(
    Output("trail-unesco-recommendations-container","children"),
    Input('selected-airbnb-store','data'),
    Input('trail-feedback-store','data'),
    Input('unesco-feedback-store','data')
)
def update_side_recs(sel_airbnb, trail_fb, unesco_fb):
    if not sel_airbnb:
        return html.Div("Click an Airbnb to see nearby trails & UNESCO sites")
    lat, lon = sel_airbnb['latitude'], sel_airbnb['longitude']
    # Trails
    df_tr = trails_df.copy()
    df_tr['distance'] = np.hypot(df_tr.latitude-lat, df_tr.longitude-lon)
    top_tr = df_tr.nsmallest(10,'distance')
    tr_rows = []
    for _, t in top_tr.iterrows():
        tid = str(t['trail_id'])
        likes    = (trail_fb or {}).get(tid,{}).get('likes',0)
        dislikes = (trail_fb or {}).get(tid,{}).get('dislikes',0)
        tr_rows.append(html.Tr([
            html.Td(t['trail_name']),
            html.Td(f"{t['distance']:.2f}"),
            html.Td(html.Button("👍", id={'type':'trail-like','index':tid},
                                style={"margin":"0 4px",
                                       "color": "white" if likes>dislikes else "#28a745",
                                       "background": "#28a745" if likes>dislikes else "white",
                                       "border": "1px solid #28a745"})),
            html.Td(html.Button("👎", id={'type':'trail-dislike','index':tid},
                                style={"margin":"0 4px",
                                       "color": "white" if dislikes>likes else "#dc3545",
                                       "background": "#dc3545" if dislikes>likes else "white",
                                       "border": "1px solid #dc3545"}))
        ]))
    # UNESCO
    df_un = unesco_df.copy()
    df_un['distance'] = np.hypot(df_un.latitude-lat, df_un.longitude-lon)
    top_un = df_un.nsmallest(10,'distance')
    un_rows = []
    for _, u in top_un.iterrows():
        uid = str(u['unesco_id'])
        likes    = (unesco_fb or {}).get(uid,{}).get('likes',0)
        dislikes = (unesco_fb or {}).get(uid,{}).get('dislikes',0)
        un_rows.append(html.Tr([
            html.Td(u['site_name']),
            html.Td(u['type']),
            html.Td(html.Button("👍", id={'type':'unesco-like','index':uid},
                                style={"margin":"0 4px",
                                       "color": "white" if likes>dislikes else "#28a745",
                                       "background": "#28a745" if likes>dislikes else "white",
                                       "border": "1px solid #28a745"})),
            html.Td(html.Button("👎", id={'type':'unesco-dislike','index':uid},
                                style={"margin":"0 4px",
                                       "color": "white" if dislikes>likes else "#dc3545",
                                       "background": "#dc3545" if dislikes>likes else "white",
                                       "border": "1px solid #dc3545"}))
        ]))
    return html.Div([
        html.H6("🥾 Nearby Trails"),
        dbc.Table([html.Thead(html.Tr([html.Th("Trail"),html.Th("Dist"),html.Th("👍"),html.Th("👎")])),
                   html.Tbody(tr_rows)], size="sm", bordered=True),
        html.H6("🏛 Nearby UNESCO Sites", className="mt-3"),
        dbc.Table([html.Thead(html.Tr([html.Th("Site"),html.Th("Type"),html.Th("👍"),html.Th("👎")])),
                   html.Tbody(un_rows)], size="sm", bordered=True)
    ])

# 8) Unified side‐feedback (clear + trail 👍👎 + UNESCO 👍👎) ----------------
@app.callback(
    Output('trail-feedback-store','data'),
    Output('unesco-feedback-store','data'),
    Input('selected-airbnb-store','data'),
    Input({'type':'trail-like',    'index':ALL}, 'n_clicks'),
    Input({'type':'trail-dislike', 'index':ALL}, 'n_clicks'),
    Input({'type':'unesco-like',    'index':ALL}, 'n_clicks'),
    Input({'type':'unesco-dislike', 'index':ALL}, 'n_clicks'),
    State('trail-feedback-store','data'),
    State('unesco-feedback-store','data'),
    prevent_initial_call=True
)
def handle_side_feedback(sel_airbnb,
                         tl_clicks, td_clicks,
                         ul_clicks, ud_clicks,
                         trail_store, unesco_store):
    trig = ctx.triggered_id

    # 1) Clear-all button
    if trig == 'clear-button':
        return {}, {}

    # 2) Trail 👍/👎
    if isinstance(trig, dict) and trig['type'].startswith('trail-'):
        tid = str(trig['index'])
        ts  = trail_store or {}
        ts.setdefault(tid, {'likes':0,'dislikes':0})
        if trig['type']=='trail-like':
            ts[tid]['likes'] += 1
        else:
            ts[tid]['dislikes'] += 1
        return ts, unesco_store or {}

    # 3) UNESCO 👍/👎
    if isinstance(trig, dict) and trig['type'].startswith('unesco-'):
        uid = str(trig['index'])
        us  = unesco_store or {}
        us.setdefault(uid, {'likes':0,'dislikes':0})
        if trig['type']=='unesco-like':
            us[uid]['likes'] += 1
        else:
            us[uid]['dislikes'] += 1
        return trail_store or {}, us

    # 4) Nothing else changes
    return no_update, no_update

# 8a) Airbnb feedback --------------------------------------------------------
@app.callback(
    Output("airbnb-feedback-store", "data"),
    Input({'type': 'like-btn',    'index': ALL}, "n_clicks"),
    Input({'type': 'dislike-btn', 'index': ALL}, "n_clicks"),
    State("airbnb-feedback-store", "data"),
    prevent_initial_call=True
)
def update_airbnb_feedback(likes, dislikes, store):
    if not ctx.triggered:
        return store or {}
    store = store or {}
    trig  = ctx.triggered_id
    lid   = str(trig["index"])
    if lid not in store:
        store[lid] = {"likes": 0, "dislikes": 0}
    if trig["type"] == "like-btn":
        store[lid]["likes"] += 1
    else:
        store[lid]["dislikes"] += 1
    return store

# 9) Final modal (enhanced for combined itinerary display)
@app.callback(
    Output("final-modal","is_open"),
    Output("final-modal-body","children"),
    Output("itinerary-store", "data"),
    Output("budget-store", "data"),
    Output("itinerary-header", "children"),
    Input("done-button","n_clicks"),
    Input("close-modal","n_clicks"),
    State("final-modal","is_open"),
    State("trail-feedback-store","data"),
    State("unesco-feedback-store","data"),
    State("airbnb-feedback-store","data"),
    State("airbnb-listings-store","data"),
    State("trip-duration-slider", "value"),
    prevent_initial_call=True
)
def toggle_final_modal(done_n, close_n, is_open, 
                       trail_fb, unesco_fb, airbnb_fb, listings, trip_duration):
    if ctx.triggered_id=="close-modal":
        return False, no_update, no_update, no_update, no_update
        
    if done_n and listings:
        # Build multi-day itinerary
        itinerary = build_itinerary(trail_fb or {}, unesco_fb or {}, listings, airbnb_fb or {}, trip_duration)
        
        # Calculate budget
        total_budget = 0
        breakdown = {
            "Accommodation": 0,
            "Activities": 0,
            "Transportation": trip_duration * 20,  # €20 per day
            "Food": trip_duration * 40  # €40 per day
        }
        
        for day in itinerary:
            for activity in day["activities"]:
                if activity['type'] == 'stay':
                    breakdown["Accommodation"] += activity.get('price', 0)
                    total_budget += activity.get('price', 0)

                elif activity['type'] == 'site':
                    fee = activity.get('entrance_fee', 0)
                    breakdown["Activities"] += fee
                    total_budget += fee

                elif activity['type'] == 'trail':
                    # you were hard-coding €5 for trail access
                    breakdown["Activities"] += 5
                    total_budget += 5
        
        total_budget += breakdown["Transportation"]
        total_budget += breakdown["Food"]
        
        # Create itinerary display with combined activities
        itinerary_display = []
        for day in itinerary:
            day_activities = []
            for activity in day["activities"]:
                if activity['type'] == 'trail':
                    day_activities.append(dbc.ListGroupItem([
                        html.Span("🥾 ", style={"fontSize": "1.5rem", "marginRight": "10px"}),
                        html.Div([
                            html.Strong(activity['trail_name'], style={"fontSize": "1.1rem"}),
                            html.Br(),
                            html.Span(f"{activity['distance_km']:.1f} km • ", className="text-muted"),
                            html.Span(f"⏱ {activity['estimated_time']/60:.1f} hrs • ", className="text-muted"),
                            html.Span(f"Difficulty: {activity.get('difficulty', 'Moderate')}", className="text-muted")
                        ])
                    ], className="d-flex align-items-center"))
                    
                elif activity['type'] == 'site':
                    day_activities.append(dbc.ListGroupItem([
                        html.Span("🏛 ", style={"fontSize": "1.5rem", "marginRight": "10px"}),
                        html.Div([
                            html.Strong(activity['site_name'], style={"fontSize": "1.1rem"}),
                            html.Br(),
                            html.Span(f"{activity['type']} • ", className="text-muted"),
                            html.Span(f"€{activity.get('entrance_fee', 0):.1f} entrance • ", className="text-muted"),
                            html.Span(f"Culture: {activity.get('culture_category', 'Cultural')}", className="text-muted")
                        ])
                    ], className="d-flex align-items-center"))
                    
                elif activity['type'] == 'stay':
                    day_activities.append(dbc.ListGroupItem([
                        html.Span("🏠 ", style={"fontSize": "1.5rem", "marginRight": "10px"}),
                        html.Div([
                            html.Strong(f"€{int(activity['price'])} {activity['room_type']}", style={"fontSize": "1.1rem"}),
                            html.Br(),
                            html.Span(f"⭐ {activity['guest_rating']:.1f} • ", className="text-muted"),
                            html.Span(f"🧼 {activity['cleanliness']:.1f} • ", className="text-muted"),
                            html.Span(f"👥 {activity['capacity']} guests", className="text-muted")
                        ])
                    ], className="d-flex align-items-center"))
            
            itinerary_display.append(dbc.Card(
                dbc.CardBody([
                    html.H4(f"Day {day['day']}", className="card-title mb-3 border-bottom pb-2"),
                    dbc.ListGroup(day_activities, flush=True)
                ]), className="mb-4 shadow-sm"
            ))
        
        # Add budget summary
        budget_display = dbc.Card(
            dbc.CardBody([
                html.H4("Budget Summary", className="card-title"),
                html.Div(f"Total: €{total_budget:.2f}", className="h4 text-success mb-3"),
                html.Div([
                    html.Div(f"Accommodation: €{breakdown['Accommodation']:.2f}"),
                    html.Div(f"Activities: €{breakdown['Activities']:.2f}"),
                    html.Div(f"Transportation: €{breakdown['Transportation']:.2f}"),
                    html.Div(f"Food: €{breakdown['Food']:.2f}")
                ])
            ]), className="mt-3"
        )
        
        # Create header showing number of activities
        num_trails = sum(1 for day in itinerary for act in day["activities"] if act['type'] == 'trail')
        num_sites = sum(1 for day in itinerary for act in day["activities"] if act['type'] == 'site')
        num_stays = sum(1 for day in itinerary for act in day["activities"] if act['type'] == 'stay')
        
        itinerary_header = html.Div([
            html.H4("Your Adventure Itinerary", className="mb-1"),
            html.P([
                html.Span(f"{trip_duration} days • ", className="text-primary"),
                html.Span(f"{num_trails} trails • ", className="text-success"),
                html.Span(f"{num_sites} UNESCO sites • ", className="text-info"),
                html.Span(f"{num_stays} accommodations", className="text-warning"),
            ], className="mb-0")
        ])
        
        body = html.Div([
            itinerary_header,
            *itinerary_display,
            budget_display
        ])
        
        return True, body, itinerary, {'total': total_budget, 'breakdown': breakdown}, itinerary_header
    
    return is_open, no_update, no_update, no_update, no_update

# 10) Download PDF
@app.callback(
    Output('download-pdf','data'),
    Input('download-button','n_clicks'),
    State("itinerary-store","data"),
    State("budget-store","data"),
    prevent_initial_call=True
)
def download_pdf(n, itinerary, budget_data):
    if not itinerary:
        return no_update
    return dcc.send_bytes(create_pdf(itinerary, budget_data['total']), "itinerary.pdf")

# 11) Update itinerary display (enhanced for combined view)
@app.callback(
    Output("itinerary-display", "children"),
    Input("itinerary-store", "data")
)
def update_itinerary_display(itinerary):
    if not itinerary:
        return dbc.Alert("Generate an itinerary to see your multi-day plan", color="secondary")
    
    days = []
    for day in itinerary:
        day_activities = []
        for activity in day["activities"]:
            if activity['type'] == 'trail':
                day_activities.append(dbc.ListGroupItem([
                    html.Span("🥾 ", style={"fontSize": "1.5rem", "marginRight": "10px"}),
                    html.Div([
                        html.Strong(activity['trail_name'], style={"fontSize": "1.1rem"}),
                        html.Br(),
                        html.Span(f"{activity['distance_km']:.1f} km • ", className="text-muted"),
                        html.Span(f"⏱ {activity['estimated_time']/60:.1f} hrs • ", className="text-muted"),
                        html.Span(f"Difficulty: {activity.get('difficulty', 'Moderate')}", className="text-muted")
                    ])
                ], className="d-flex align-items-center"))
                
            elif activity['type'] == 'site':
                day_activities.append(dbc.ListGroupItem([
                    html.Span("🏛 ", style={"fontSize": "1.5rem", "marginRight": "10px"}),
                    html.Div([
                        html.Strong(activity['site_name'], style={"fontSize": "1.1rem"}),
                        html.Br(),
                        html.Span(f"{activity['type']} • ", className="text-muted"),
                        html.Span(f"€{activity.get('entrance_fee', 0):.1f} entrance • ", className="text-muted"),
                        html.Span(f"Culture: {activity.get('culture_category', 'Cultural')}", className="text-muted")
                    ])
                ], className="d-flex align-items-center"))
                
            elif activity['type'] == 'stay':
                day_activities.append(dbc.ListGroupItem([
                    html.Span("🏠 ", style={"fontSize": "1.5rem", "marginRight": "10px"}),
                    html.Div([
                        html.Strong(f"€{int(activity['price'])} {activity['room_type']}", style={"fontSize": "1.1rem"}),
                        html.Br(),
                        html.Span(f"⭐ {activity['guest_rating']:.1f} • ", className="text-muted"),
                        html.Span(f"🧼 {activity['cleanliness']:.1f} • ", className="text-muted"),
                        html.Span(f"👥 {activity['capacity']} guests", className="text-muted")
                    ])
                ], className="d-flex align-items-center"))
        
        days.append(dbc.Card(
            dbc.CardBody([
                html.H4(f"Day {day['day']}", className="card-title mb-3 border-bottom pb-2"),
                dbc.ListGroup(day_activities, flush=True)
            ]), className="mb-4 shadow-sm"
        ))
    
    return html.Div(days)

# 12) Budget tracker
@app.callback(
    Output("total-budget", "children"),
    Output("budget-progress", "value"),
    Output("budget-progress", "label"),
    Output("budget-breakdown", "children"),
    Input("budget-store", "data"),
    Input("trip-duration-slider", "value")
)
def update_budget_tracker(budget_data, trip_duration):
    if not budget_data or budget_data['total'] == 0:
        return "€0.00", 0, "0%", "No budget data available"
    
    total = budget_data['total']
    breakdown = budget_data['breakdown']
    
    # Calculate budget progress (as percentage of a reasonable max budget)
    max_budget = trip_duration * 300  # €300 per day
    progress = min(100, (total / max_budget) * 100)
    
    # Create budget breakdown
    breakdown_html = [
        html.Div([
            html.Span(f"{category}:", className="font-weight-bold"),
            html.Span(f" €{cost:.2f}", className="float-right")
        ]) for category, cost in breakdown.items()
    ]
    
    return f"€{total:.2f}", progress, f"{progress:.0f}%", breakdown_html

if __name__ == "__main__":
    app.run(debug=True, port=8051)
