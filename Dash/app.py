import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import folium
import Itinerary_Functions
from pathlib import Path
import xml.etree.ElementTree as ET

# CREATING AND LOADING API STATIC DATA

file_content = Path("ontology.xml").read_text()
root = ET.fromstring(file_content)
df_cat_en_fr = Itinerary_Functions.compute_df_cat_en_fr(root)
(master_categories,classes_to_be_removed_list,full_categories_chain_list,all_reversed_categories_chains_sorted,
    ) = Itinerary_Functions.get_classes_to_be_removed_list(root)
Itinerary_Functions.write_files(master_categories, classes_to_be_removed_list)
fixedFile = Itinerary_Functions.write_reg_reu_main_csv()
(df_clean,df_poi_3,df_food,df_outdooractivity,df_best,df_poi) = Itinerary_Functions.prepare_dataframes(df_cat_en_fr,
                            fixedFile,full_categories_chain_list,all_reversed_categories_chains_sorted)

# INITIALIZING API GLOBAL VARIABLES

checkbox_2_option=['Incontournable','POI','Activité','Restaurant']
reunion_coords=[-21.13,55.47]


# INITIZALIZE THE API

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)

app.layout=html.Div([
    dcc.Location(id='url',refresh=False),
    html.Div(id='page_content'),
    dcc.Store(id='itinerary_df'),
])


# LANDING PAGE CALLBACK

@app.callback(Output('page_content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/poi':
        return page_1
    elif pathname == '/itinerary':
        return page_2
    else:
        return index_page


# LANDING PAGE LAYOUT

index_page=html.Div(children=
                        [html.Div(children=[html.H2('La Réunion : mon séjour',style={'color':'black'}),
                                            html.Div([dbc.Button(children=dcc.Link("Points d'intérêt",href='/poi',style={'color':'white'}),color="primary", className="me-1"),
                                                    dbc.Button(children=dcc.Link("Mon Itinéraire",href='/itinerary',style={'color':'white'}),color="primary", className="me-1")
                                                    ])],
                                style={'display' : 'inline-block', 'verticalAlign' : 'middle', 'textAlign' : 'center','marginLeft':'27%','marginTop':'3%'}
                                )
                        ],
                    style={'background-image': 'url(/assets/la_reunion.jpg)','background-size': 'cover',      
                    'background-position': 'center','background-repeat': 'no-repeat','height': '100vh',               
                    'width': '100vw'}
                    )


# PAGE 1 STYLES

CONTROLS_STYLE_1 ={
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '15%',
    'padding': '20px 10px',
}

CONTENT_STYLE_1 = {
    'margin-left': '20%',
    'margin-right': '5%',
    'top': 0,
    'padding': '20px 10px'
}


# CREATE CONTROLS OF PAGE 1

controls_1 = html.Div(
    [   
        html.H4("Catégories", style={'textAlign': 'center','color': 'black'}),
        html.Hr(),
        dbc.Card([dbc.Checklist(
            id='check_list_1',
            options=sorted(df_clean.superclass_0_fr.unique()),
            value=[],
        )])
    ],
    style=CONTROLS_STYLE_1,
)


# CREATE MAP OF PAGE 1

@app.callback(Output('map_1', 'srcDoc'),Input('check_list_1', 'value'))
def map_update(value):
    m = folium.Map(location=reunion_coords,zoom_start=10.5,control_scale=True)
    m.add_child(folium.LatLngPopup())

    # Sorted list of the categories to be shown
    categories=sorted(value)

    # Lists of available colors and categories
    av_colors=['red', 'darkblue','green', 'black', 'orange', 'blue','lightblue', 'purple',
           'beige', 'cadetblue', 'darkred', 'gray', 'darkgreen', 'lightgreen', 'pink',
           'lightgray', 'lightred', 'white', 'darkpurple']
    av_categories=sorted(df_clean.superclass_0_fr.unique())

    # Color palette creation
    categories_index=[av_categories.index(category) for category in categories]
    colors={category:av_colors[index] for (category,index) in zip(categories,categories_index)}

    # Create the dataframe filtered for the categories to be shown
    df_filtered=df_clean[df_clean.superclass_0_fr.isin(categories)]

    # Add a marker for each POI
    for index, row in df_filtered.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        nom = row['Nom_du_POI']
        categorie = row['superclass_0_fr']
        color = colors[categorie]
        popup = folium.Popup(f"Nom: {nom}<br>Catégorie: {categorie}",max_width=len(f"Nom: {row.Nom_du_POI}"*20))
        icon = folium.Icon(color=color, icon_size='small')
        folium.Marker([lat, lon], icon=icon, popup=popup).add_to(m)

   # Add a map control (Groupby on Jour)
    folium.TileLayer('Stamen Terrain').add_to(m)
    folium.LayerControl(position='topright', collapsed=False).add_to(m) 

    # Add a legend to the map
    legend_html = f'<div style="position: fixed; bottom: 40px; left: 20px; width: 200px; height: {40+30*len(categories)}px; border:2px solid grey; z-index:9999; font-size:14px;background-color:white;">&nbsp;<b>Légende</b><br>'
    for category, color in colors.items():
        legend_html += f'&nbsp;<i class="fa fa-map-marker fa-2x" style="color:{color}"></i>&nbsp; {category} <br>'
    legend_html += "</div>"
    if categories!=[]:
        m.get_root().html.add_child(folium.Element(legend_html))
    m.save('map_1.html')
    return open('map_1.html', 'r', encoding='utf-8').read()


# PAGE 1 CONTENT DEFINITION

content_1=html.Div([
        html.H3("Principaux points d'intérêt de La Réunion", style={'color':'black','textAlign':'center'}),
        html.Iframe(id='map_1', srcDoc=None,width='100%',height='700')
        ],style=CONTENT_STYLE_1)


# PAGE 1 LAYOUT

page_1=html.Div([controls_1,content_1],style={'background-color': '#f8f9fa'})


# CREATE ITINERARY

@app.callback(Output('itinerary_df','data'),Output('cards_day','options'),Output('cards_day','value'),Output('check_list_2','value'),Input('submit_button', 'n_clicks'),[State('check_list_2', 'value'),
    State('ndays_dropdown', 'value'),State('npoi_dropdown', 'value'),State('nactivity_dropdown', 'value'),
    State('nfood_dropdown', 'value')])
def create_itinerary(n_clicks,value,ndays,npoi,nactivity,nfood):

    if not (None in [ndays,npoi,nactivity,nfood]):
        # Calculate the itinerary
        df_iti=Itinerary_Functions.get_itinerary(df_clean,df_poi_3,df_food,df_outdooractivity,
        df_best,df_poi,num_days=ndays, poi_per_day=npoi, food_per_day=nfood, activity_per_day=nactivity)

        return df_iti.to_json(date_format='iso', orient='split'),list(range(1,ndays+1,1)),1,checkbox_2_option
    else:
        return no_update,no_update,no_update,no_update


# CREATE MAP OF PAGE 2

@app.callback(Output('map_2', 'srcDoc'),Input('itinerary_df','data'),Input('submit_button', 'n_clicks'),Input('check_list_2', 'value'),Input('cards_day','options'),
    [State('ndays_dropdown', 'value'),State('npoi_dropdown', 'value'),State('nactivity_dropdown', 'value'),
    State('nfood_dropdown', 'value')])
def map_update(data,n_clicks,value,cardday,ndays,npoi,nactivity,nfood):

    if not None in [data,value]:
        #Load the data
        df_iti=pd.read_json(data, orient='split')

        # Initialize the map as m
        map_center = [df_iti.Latitude.mean(),df_iti.Longitude.mean()]
        m = folium.Map(location=map_center,zoom_start=10.5,control_scale=True)
        m.add_child(folium.LatLngPopup())

        # sorted list of categories to be displayed
        categories=sorted(value) 

        # Available colors and categories
        av_colors=['red', 'darkblue','green', 'black', 'orange', 'beige','lightblue', 'purple',
           'blue', 'cadetblue', 'darkred', 'gray', 'darkgreen', 'lightgreen', 'pink',
           'lightgray', 'lightred', 'white', 'darkpurple']
        av_categories=checkbox_2_option

        # Create color palette for categories to be displayed
        categories_index=[av_categories.index(category) for category in categories]
        colors={category:av_colors[index] for (category,index) in zip(categories,categories_index)}

        # Create the dataframe filtered for the categories to be displayed
        df_filtered=df_iti[df_iti.Type.isin(categories)]

        # Create a marker and for each poi of the itinerary
        for day, df_gb in df_filtered.groupby('Jour'):
            fg = folium.FeatureGroup(name=f"Jour: {day}")
            for row in df_gb.itertuples():
                popup =folium.Popup( f"Nom: {row.Nom_du_POI}<br>Catégorie: {row.Type}<br>Jour: {row.Jour}",max_width=len(f"Nom: {row.Nom_du_POI}"*20))
                folium.Marker(location=[row.Latitude, row.Longitude], popup = popup, 
                    icon=folium.Icon(color = colors[row.Type], iconsize='small')).add_to(fg)
                fg.add_to(m)
    
        # Add a map control (Groupby on Jour)
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.LayerControl(position='topright', collapsed=False).add_to(m) 

        # Add a legend to the map
        legend_html = f'<div style="position: fixed; bottom: 40px; left: 20px; width: 200px;height: {30+30*len(categories)}px; border:2px solid grey; z-index:9999; font-size:14px;background-color:white;">&nbsp;<b>Légende</b><br>'
 
        for category, color in colors.items():
            legend_html += f'&nbsp;<i class="fa fa-map-marker fa-2x" style="color:{color}"></i>&nbsp; {category} <br>'
        legend_html += "</div>"
        if categories!=[]:
            m.get_root().html.add_child(folium.Element(legend_html))
        m.save('map_2.html')

    else:
        # Initialize the map as m
        map_center = reunion_coords
        m = folium.Map(location=map_center,zoom_start=10.5,control_scale=True)
        m.add_child(folium.LatLngPopup())
        m.save('map_2.html')

    return open('map_2.html', 'r', encoding='utf-8').read()

# FUNCTION TO CREATE A DASH CARD REPRESENTING A POI

def create_card(row):
    card=dbc.Card(dbc.CardBody(
        [
            html.H5(row.Nom_du_POI, className="card-title"),
            html.H6(row.Type, className="card-subtitle"),
            html.Hr(),
            html.Label(f"Catégorie: {row.subclass_0_fr}",className="card-text"),
            html.Label(f"Commune: {row.commune}",className="card-text"),
            html.Label(f"Latitude: {row.Latitude}",className="card-text"),
            html.Label(f"Longitude: {row.Longitude}",className="card-text")
        ]
    ),
    style={"width": "15rem"},
    )
    return card


# UPDATE DISPLAY OF POI PER DAY

@app.callback(Output('Itinerary_cards','children'),Input('itinerary_df','data'),
                Input('cards_day','value'),Input('submit_button', 'n_clicks'))
def update_cards(data,day,click):

    if not None in [day,data]:
        #Load the data
        df_iti=pd.read_json(data, orient='split')
        df_day=df_iti[df_iti['Jour']==day]
        # Create a card for each POI in the Itinerary
        card_dict={}
        card_dict['poi']=[]
        card_dict['food']=[]
        card_dict['activity']=[]
        card_dict['incont']=[]
        
        for index,row in df_day.iterrows():
            if row.Type=='POI':
                card_dict['poi'].append(create_card(row))
            elif row.Type=='Restaurant':
                card_dict['food'].append(create_card(row))
            elif row.Type=='Incontournable':
                card_dict['incont'].append(create_card(row))
            else:
                card_dict['activity'].append(create_card(row))     

        incont_col_children=[html.H6("Incontournables"),html.Hr()]+card_dict['incont']
        poi_col_children=[html.H6("POI"),html.Hr()]+card_dict['poi']
        activity_col_children=[html.H6("Activités"),html.Hr()]+card_dict['activity']
        food_col_children=[html.H6("Restaurants"),html.Hr()]+card_dict['food']
        column_list=[dbc.Col(children=incont_col_children,md=3),
                dbc.Col(children=poi_col_children,md=3),
                dbc.Col(children=activity_col_children,md=3),
                dbc.Col(children=food_col_children,md=3)
                ]
        return column_list
    else:
        return []


# PAGE 2 STYLES

CONTROLS_STYLE_2 ={
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
}

CONTENT_STYLE_2 = {
    'margin-left': '25%',
    'margin-right': '5%',
    'top': 0,
    'padding': '20px 10px'
}


# PAGE 2 CONTROLS DEFINITION

controls_2 = html.Div(children=
        [html.Br(),
        html.Br(),
        html.H5("Préférences du séjour", style={'textAlign': 'center','color': 'black'}),
        html.Hr(),
        html.Label('Nombre de jours', style={'color' : 'black', 'fontSize': '16px'}),
        dcc.Dropdown(id = 'ndays_dropdown',options=list(range(3,21,1)),placeholder="Entrez un nombre entre 3 et 20"),
        html.Br(),
        html.Label('Nombre de POI par jour', style={'color' : 'black', 'fontSize': '16px'}),
        dcc.Dropdown(id = 'npoi_dropdown',options=list(range(2,11,1)),placeholder="Entrez un nombre entre 2 et 10"),
        html.Br(),
        html.Label("Nombre d'activités par jour", style={'color' : 'black', 'fontSize': '16px'}),
        dcc.Dropdown(id = 'nactivity_dropdown',options=list(range(0,11,1)),placeholder="Entrez un nombre entre 0 et 10"),
        html.Br(),
        html.Label('Nombre de restaurants', style={'color' : 'black', 'fontSize': '16px'}),
        dcc.Dropdown(id = 'nfood_dropdown',options=list(range(0,11,1)),placeholder="Entrez un nombre entre 0 et 10"),
        html.Br(),
        html.H5("Affichage", style={'textAlign': 'center','color': 'black'}),
        html.Hr(),
        dbc.Card([dbc.Checklist(options=checkbox_2_option,value=[],id='check_list_2',)],style={'textAlign':'left'}),
        html.Br(),
        html.Div(dbc.Button( id='submit_button',n_clicks=0,children='Calculer',color='primary'),className="d-grid gap-2"),
    ],style=CONTROLS_STYLE_2
)


# PAGE 2 CONTENT DEFINITION

content_2=html.Div([
        html.H3("Mon séjour à la Réunion", style={'color':'black','textAlign':'center'}),
        html.Iframe(id='map_2', srcDoc=None,width='100%',height='700'),
        html.Hr(),
        html.H3("Programme du séjour", style={'color':'black','textAlign':'left'}),
        html.Hr(),
        html.Label('Journée', style={'color' : 'black', 'fontSize': '16px','textAlign':'center'}),
        dcc.Dropdown(id = 'cards_day',options=list(range(1,8,1)),placeholder="Sélectionnez un jour",style={'width':'40%'}),
        html.Br(),
        dbc.Row(id='Itinerary_cards',children=[])
],style=CONTENT_STYLE_2)


# PAGE 2 LAYOUT

page_2=html.Div([controls_2,content_2],style={'background-color': '#f8f9fa'})


# LAUNCH API SERVER

if __name__ == '__main__':
    app.run_server(debug=True,host='0.0.0.0')


