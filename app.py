# NDP app always beta a lot
import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st
import shapely.speedups
shapely.speedups.enable()
import plotly.express as px
px.set_mapbox_access_token(st.secrets['MAPBOX_TOKEN'])
my_style = st.secrets['MAPBOX_STYLE']
from pathlib import Path
import h3pandas as h3
import json


# page setup ---------------------------------------------------------------
st.set_page_config(page_title="NDP App d1", layout="wide")
padding = 1
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

header = '<p style="font-family:sans-serif; color:grey; font-size: 12px;">\
        NDP data paper #4 V0.9\
        </p>'
st.markdown(header, unsafe_allow_html=True)
# plot size setup
#px.defaults.width = 600
px.defaults.height = 700

# page header
header_title = '''
**Naked Density Project**
'''
st.subheader(header_title)
header_text = '''
<p style="font-family:sans-serif; color:dimgrey; font-size: 10px;">
Naked Density Project is a PhD research project by <a href="https://research.aalto.fi/en/persons/teemu-jama" target="_blank">Teemu Jama</a> in Aalto University Finland.  
NDP project studies correlation between urban density and <a href="https://sdgs.un.org/goals" target="_blank">SDG-goals</a> by applying latest spatial data analytics and machine learning. \
</p>
'''
st.markdown(header_text, unsafe_allow_html=True)
st.markdown("----")
# content
st.title("Data Paper #4")
st.subheader("Correlation between urban density and amenities")
ingress = '''
<p style="font-family:sans-serif; color:Black; font-size: 14px;">
This data paper visualise the change in correlation between urban density and urban amenities.
Research quest here is to see how often used arguments of positive density impacts on local amenities in
urban planning works in different geographical scales. The research method is correlation calculations between
gross floor area (GFA) and urban amenities in different scales.
Research reference is highly influential article 
<a href="https://academic.oup.com/joeg/article-abstract/1/1/27/2964661" target="_blank">Consumer City</a>  
by Edward Glaeser and Jef Kolko from year 2001.
</p>
'''
st.markdown(ingress, unsafe_allow_html=True)
st.markdown("###")
# translate dict
eng_feat = {
    'kem_2000':'Total GFA in 2000',
    'askem_2000':'Residential GFA in 2000',
    'kem_2016':'Total GFA in 2016',
    'askem_2016':'Residential GFA in 2016',
    'kem_muutos':'GFA change 2000-2016',
    'askem_muutos':'Residential GFA change 2000-2016',
    'palv_pien_2000':'One person companies (OPC) in urban amenities in 2000',
    'palv_2000':'Urban amenities (OPC excluded) in 2000',
    'kaup_2000':'Wholesale and retail trade in 2000',
    'pt_2000':'Crocery stores and kiosks in 2000',
    'palv_pien_2016':'One person companies (OPC) in urban amenities in 2016',
    'palv_2016':'Urban amenities (OPC excluded) in 2016',
    'kaup_2016':'Wholesale and retail trade in 2016',
    'pt_2016':'Crocery stores and kiosks in 2016',
    'palv_pien_muutos':'Change in one person companies (OPC) in urban amenities 2000-2016',
    'palv_muutos':'Cange in Urban amenities (OPC excluded) 2000-2016',
    'kaup_muutos':'Change in wholesale and retail trade 2000-2016',
    'pt_muutos':'Change in Crocery stores and kiosks 2000-2016',
}

@st.cache(allow_output_mutation=True)
def load_data():
    path = Path(__file__).parent / 'h3_10_PKS.csv'
    with path.open() as f:
        data = pd.read_csv(f, index_col='h3_10', header=0)#.astype(str)
    # translate columns
    eng_data = data.rename(columns=eng_feat)
    return eng_data

gdf = load_data()
# filters
s1,s2 = st.columns(2)
pnolista = gdf['pno'].unique()
tapa = s1.selectbox('Select...',['By City','By Neighbourhood'])
if tapa == 'By City':
    kuntani = s2.selectbox(' ',['Helsinki','Espoo','Vantaa'])
    mygdf = gdf.loc[gdf.kunta == kuntani]
else:
    pnos = s2.multiselect(' ', pnolista,
                            default=['Tapiola','Pohjois-Tapiola','Otaniemi'])
    if pnos is not None:
        mygdf = gdf.loc[gdf.pno.isin(pnos)]
    else:
        st.warning('Select city or neighbourhoods.')
        st.stop()
# filters..
col_list = mygdf.drop(columns=['kunta','pno']).columns.to_list()
p1,p2 = st.columns(2)
color = p1.selectbox('Select feature..', col_list)
q_off = p2.slider('...and filter by setting low quantile cut (%)',10,90,20,10)
mygdf = mygdf.loc[mygdf[f'{color}'].astype(int) > mygdf[f'{color}'].astype(int).quantile(q_off/100)]
mapplace = st.empty()
l1,l2 = st.columns(2)
level = l1.slider('H3-resolution in map (H6-H9)',6,9,9,1)
l1.caption('https://h3geo.org/docs/core-library/restable/')
# map plot
if len(mygdf) > 1:
    plot = mygdf.h3.h3_to_parent_aggregate(level)
    # map plot
    lat = plot.unary_union.centroid.y
    lon = plot.unary_union.centroid.x
    range_min = plot[color].quantile(0.05)
    range_max = plot[color].quantile(0.95)
    fig = px.choropleth_mapbox(plot,
                            geojson=plot.geometry,
                            locations=plot.index,
                            color=color,
                            center={"lat": lat, "lon": lon},
                            mapbox_style=my_style,
                            range_color=(range_min, range_max),
                            color_continuous_scale=px.colors.sequential.Inferno[::-1],
                            #color_continuous_scale=px.colors.sequential.Blackbody[::-1],
                            #labels={'palv':'Palveluiden määrä'},
                            zoom=9,
                            opacity=0.5,
                            width=1200,
                            height=700
                            )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=700)
    fig.update_layout(coloraxis_showscale=True)
    with mapplace:
        st.plotly_chart(fig, use_container_width=True)
    
else:
    st.stop()

st.markdown('---')

# corr graphs
st.subheader('Correlation loss')

@st.cache(allow_output_mutation=True)
def corr_loss(df,h=10,corr_type='2000'):
    if corr_type == '2000':
        x_list=['Total GFA in 2000',
                'Residential GFA in 2000']
        y_list=['One person companies (OPC) in urban amenities in 2000',
                'Urban amenities (OPC excluded) in 2000',
                'Wholesale and retail trade in 2000',
                'Crocery stores and kiosks in 2000']
    elif corr_type == '2016':
        x_list=['Total GFA in 2016',
                'Residential GFA in 2016']
        y_list=['One person companies (OPC) in urban amenities in 2016',
                'Urban amenities (OPC excluded) in 2016',
                'Wholesale and retail trade in 2016',
                'Crocery stores and kiosks in 2016']
    else:
        x_list=['GFA change 2000-2016',
                'Residential GFA change 2000-2016']
        y_list=['Change in one person companies (OPC) in urban amenities 2000-2016',
                'Cange in Urban amenities (OPC excluded) 2000-2016',
                'Change in wholesale and retail trade 2000-2016',
                'Change in Crocery stores and kiosks 2000-2016',]
    frames = []
    for x in x_list:
        for y in y_list:
            corr_list = []
            for i in range(1,5):
                df_i = df.h3.h3_to_parent_aggregate(h-i,return_geometry=False)
                corr_i = df_i.corr()[x][y]
                corr_list.append(corr_i)
            corr_y = pd.DataFrame(corr_list, index=['h9','h8','h7','h6'], columns=[x+' VS '+y])
            frames.append(corr_y)
    corr_df = pd.concat(frames, axis=1, ignore_index=False)
    return corr_df

type = st.radio('Select year',('2000','2016'))
if tapa == 'By City':
    st.caption(f'Data filtered using {color} -value quantiles {q_off}-100% in {kuntani}')
    graph_title = kuntani
else:
    st.caption(f'Data filtered using {color} -value quantiles {q_off}-100% in neighbourhoods {pnos}')
    graph_title = pnos
    
if type == '2000':
    corr = corr_loss(mygdf,corr_type='2000')
elif type == '2016':
    corr = corr_loss(mygdf,corr_type='2016')
else:
    corr = corr_loss(mygdf,corr_type='muutos')

fig_corr = px.line(corr,
                   labels = {'index':'Geographical scale (H3-resolution)','value':'Correlation','variable':'Correlation pairs'},
                   title=f'Correlation loss in {graph_title} in {type}')
fig_corr.update_xaxes(autorange="reversed")
st.plotly_chart(fig_corr, use_container_width=True)

with st.expander('Correlation matrix', expanded=False):
    st.markdown(f'Correlation matrix on resolution H{level} in {graph_title}')
    st.caption('Adjust data in map for matrix')
    import matplotlib.pyplot as plt
    import seaborn as sns
    corr_mat = round(plot.corr(),2)
    fig, ax = plt.subplots()
    sns.set(font_scale=0.9)
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, linewidths=2, linecolor='white', cmap="Greens", cbar=False,
                vmin=0, vmax=1, annot = True, annot_kws={"size": 7}, ax=ax, square=True)
    st.write(fig)
        
with st.expander('Classification', expanded=False):        
    class_expl = """
    **Urban amenities** are all company business space locations which belong
    to the following finnish TOL-industry classes:  
    _Wholesale and retail_  
    _Accomondation and food service activites_  
    _Information and communication_  
    _Financial and insurance activities_  
    _Other service activities_  

    Original raw data is from
    <a href="https://research.aalto.fi/fi/projects/l%C3%A4hi%C3%B6iden-kehityssuunnat-ja-uudelleenkonseptointi-2020-luvun-segr " target="_blank">Re:Urbia</a>
    -research project data retrieved from the data products "SeutuCD 2002" and "SeutuCD 2018" by Statistical Finland.
    """
    st.markdown(class_expl, unsafe_allow_html=True)

#footer
footer_title = '''
---
**Naked Density Project**
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/ndp_app4/blob/main/LICENSE) 
'''
st.markdown(footer_title)