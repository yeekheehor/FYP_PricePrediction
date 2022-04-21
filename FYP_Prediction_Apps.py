import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from PIL import Image

#----------------------------------------------------------------------------------------------------------------------------------
def summary_table(df):
#Return a summary table with the descriptive statistics about the dataframe.
    summary = {
        "Number of Variables": [len(df.columns)],
        "Number of Observations": [df.shape[0]],
        "Missing Cells": [df.isnull().sum().sum()],
        "Missing Cells (%)": [round(df.isnull().sum().sum() / df.shape[0] * 100, 2)],
        "Duplicated Rows": [df.duplicated().sum()],
        "Duplicated Rows (%)": [round(df.duplicated().sum() / df.shape[0] * 100, 2)],
        "Categorical Variables": [len([i for i in df.columns if df[i].dtype==object])],
        "Numerical Variables": [len([i for i in df.columns if df[i].dtype!=object])],
    }
    return pd.DataFrame(summary).T.rename(columns={0: 'Values'})

@st.cache
def plot_histogram(data, x, nbins, height, margin, title_text=None):
    fig = px.histogram(data, x=x, nbins=nbins)
    fig.update_layout(bargap=0.05, height=height, width=1000, title_text=title_text, margin=dict(t=margin,b=margin)
    )
    return fig

@st.cache
def plot_scatter(data, x, y, height, margin, residual=False, title_text=None):
    fig = px.scatter(data, x=x, y=y)
    fig.update_layout(bargap=0.05, height=height, width=1000, title_text=title_text, margin=dict(t=margin,b=margin)
    )
    return fig

@st.cache(hash_funcs={pd.DataFrame: lambda x: x})
def plot_boxplot(data, x, y, height, margin, color=None, single_box=False, model_name=None, custom_feature=None, custom_target=None, title_text=None):
    fig = px.box(data, x=x, y=y, color=color)
    fig.update_layout(bargap=0.05, height=height, width=1000, title_text=title_text, margin=dict(t=margin,b=margin)
    )
    return fig

@st.cache
def plot_countplot(data, x, height, margin, title_text=None):
    fig = px.histogram(data, x=x, color=x)
    fig.update_layout(bargap=0.05, height=height, width=1000, title_text=title_text, margin=dict(t=margin,b=margin)
    )
    return fig

@st.cache
def plot_heatmap(corr_matrix, height, margin, title_text=None):
    fig = go.Figure(
        go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.index.values,
        y=corr_matrix.columns.values,
        colorscale='RdBu_R',
        zmax=1,
        zmin=-1
        )
    )
    fig.update_layout(bargap=0.05, height=height, width=1000, title_text=title_text, margin=dict(t=margin, b=margin)
    )

    return fig

@st.cache
def plot_bar(data, x, y, height,  margin, title_text=None):
    fig = px.bar(data, x=x, y=y, color=x)

    fig.update_layout(bargap=0.05, height=height, width=1000, title_text=title_text, margin=dict(t=margin,b=margin)
    )

    return fig


#----------------------------------------------------------------------------------------------------------------------------------
# set page title
st.set_page_config('Airbnb Price Prediction App')
image = Image.open('D:\Academic\Airbnb_data\Airbnb-logo.jpg')
st.image(image, width = 300)

data = pd.read_csv("D:/Academic/Airbnb_data/listings_transformed_enc.csv")

X = data[['host_response_rate','host_acceptance_rate', 'bedrooms','bathroom', 'beds','accommodates','neighborhood_enc','room_type_enc','host_response_time_enc','host_is_superhost_enc','instant_bookable_enc','has_availability_enc']]
Y = data['log_price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#----------------------------------------------------------------------------------------------------------------------------------
menu_list = ['Exploratory Data Analysis', 'Model Prediction']
menu = st.sidebar.selectbox("Menu", menu_list)

if menu == 'Exploratory Data Analysis':
    st.title('Exploratory Data Analysis of Airbnb Properties Price for Different Cities')
    data = pd.read_csv('D:/Academic/Airbnb_data/listings_new.csv')

    st.header('Descriptive Analysis')
    st.table(summary_table(data))

    st.header('Data Visualization')
    height, width, margin = 450, 1500, 10

    st.subheader('Airbnb Rental Price Distribution')
    select_country_eda = st.selectbox('Select Country',[i for i in data['area'].unique()])
    fig = plot_histogram(data = data.loc[data['area'] == select_country_eda], x="price", nbins=800, height=height,margin=margin)      
    st.plotly_chart(fig)

    st.subheader('Scatterplot')
    select_numerical = st.selectbox('Select Numerical Variable',
        ['number_of_reviews', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','host_response_rate','host_acceptance_rate']
    )
    fig = plot_scatter(data=data, x=select_numerical, y="price", height=height, margin=margin)
    st.plotly_chart(fig)

    st.subheader('Categorical Graphs')
    select_graph = st.radio(
        'Select the Type of Graph',
        ('Boxplot', 'Countplot')
    )
    select_categorical = st.selectbox(
        'Select Categorical Variable',
         ['host_is_superhost', 'neighborhood', 'room_type', 'accommodates','bedrooms','beds','bathroom','has_availability','instant_bookable','host_response_time']
    )
    if select_graph == 'Boxplot':
        fig = plot_boxplot(data=data, x=select_categorical, y="price", color=select_categorical, height=height,margin=margin)
    elif select_graph == 'Countplot':
        fig = plot_countplot(data=data, x=select_categorical, height=height, margin=margin)
    st.plotly_chart(fig)

    st.subheader('Correlation Matrix')
    corr_matrix = data.corr()
    fig = plot_heatmap(corr_matrix=corr_matrix, height=height, margin=margin)
    st.plotly_chart(fig)



elif menu == 'Model Prediction':
    country_list = ['London', 'New York', 'Singapore']
    country = st.sidebar.radio("Select Country", country_list) 
    
    if country == 'London':
        st.sidebar.header('Select Input')
        room_type_list = ['Entire home/apt','Hotel room','Private room','Shared room']
        instant_booking_list = ['t','f']
        has_availability_list = ['t','f']
        superhost_list = ['t','f']
        host_response_time_list = ['a few days or more','within a day','within a few hours','within an hour']
        london_neighborhood_list = ['Barking and Dagenham', 'Barnet',  'Bexley','Brent', 'Bromley', 'Camden', 'City of London', 
        'Croydon', 'Ealing', 'Enfield', 'Greenwich', 'Hackney', 'Hammersmith and Fulham', 'Haringey','Harrow',
        'Havering','Hillingdon','Hounslow','Islington','Kensington and Chelsea','Kingston upon Thames','Lambeth',
        'Lewisham','Merton','Newham','Redbridge','Richmond upon Thames','Southwark','Sutton','Tower Hamlets','Waltham Forest','Wandsworth','Westminster', 'Other']

        london_neighborhood_choice = st.sidebar.selectbox('Neighborhood', london_neighborhood_list)
        london_neighborhood = data.loc[data.neighborhood == london_neighborhood_choice, 'neighborhood_enc'].values[0]  

        room_type_choice = st.sidebar.selectbox(label='Room Type', options=room_type_list)
        room_type = data.loc[data.room_type == room_type_choice, 'room_type_enc'].values[0]

        accomodates = st.sidebar.slider('Number of guests', 0, 16, 7)

        bedroom = st.sidebar.slider('Number of bedrooms', 0, 22, 4)

        bathroom = st.sidebar.slider('Number of bathrooms', 0, 21, 10)

        bed = st.sidebar.slider('Number of beds', 0, 58, 7)

        instant_booking_choice = st.sidebar.selectbox(label='Instant_booking (t = True; f = False)', options=instant_booking_list)
        instant_booking = data.loc[data.instant_bookable == instant_booking_choice, 'instant_bookable_enc'].values[0]

        has_availability_choice = st.sidebar.selectbox(label='Has Availability (t = True; f = False)', options=has_availability_list)
        has_availability = data.loc[data.has_availability == has_availability_choice, 'has_availability_enc'].values[0]

        superhost_choice = st.sidebar.selectbox(label='Superhost (t = True; f = False)', options=superhost_list)
        superhost = data.loc[data.host_is_superhost == superhost_choice, 'host_is_superhost_enc'].values[0]

        host_response_time_choice = st.sidebar.selectbox(label='host_response_time', options=host_response_time_list)
        host_response_time = data.loc[data.host_response_time == host_response_time_choice, 'host_response_time_enc'].values[0]
        
        host_response_rate = st.sidebar.slider('Host Response Rate(%)', 0, 100, 80)   
        host_acceptance_rate = st.sidebar.slider('Host Acceptance Rate(%)', 0, 100, 75)

        user_input_num = {'host_response_rate': host_response_rate,
        'host_acceptance_rate':host_acceptance_rate, 
        'bedroom':bedroom,
        'bathroom':bathroom, 
        'bed':bed,
        'accomodates':accomodates}

        user_input_cat = {'london_neighborhood':london_neighborhood_choice,
        'room_type':room_type_choice,
        'instant_booking':instant_booking_choice,
        'has_availability':has_availability_choice,
        'is_superhost':superhost_choice,
        'host_response_time':host_response_time_choice}

        num_data, cat_data = st.columns(2)
        user_input_num = pd.DataFrame(user_input_num, index = ['Input Data'])
        user_input_num = user_input_num.T
        num_data.write(user_input_num)
        
        user_input_cat = pd.DataFrame(user_input_cat, index = ['Input Data'])
        user_input_cat = user_input_cat.T
        cat_data.write(user_input_cat)  

        predict_array = [host_response_rate,host_acceptance_rate, bedroom,bathroom, bed,accomodates,london_neighborhood,room_type,instant_booking,has_availability,superhost,host_response_time]

        value_to_predict = pd.DataFrame([predict_array], columns=X.columns)

        xgb_tuned = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                gamma=0, gpu_id=-1, importance_type=None,
                interaction_constraints='', learning_rate=1, max_delta_step=0,
                max_depth=2, min_child_weight=3,monotone_constraints='()', n_estimators=900, n_jobs=8,
                num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
                reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
                validate_parameters=1, verbosity=None)

        xgb_tuned.fit(X_train.values, Y_train.values)     
        xgb_tuned_pred = xgb_tuned.predict(X_test.values)
        xgb_score = xgb_tuned.score(X_train.values, Y_train.values)
        xgb_MSE = mean_squared_error(Y_test, xgb_tuned_pred)
        xgb_MAE = mean_absolute_error(Y_test, xgb_tuned_pred)

        if st.button('Predict'):
            st.write('R2', xgb_score)
            st.write('MSE', xgb_MSE)
            st.write('MAE', xgb_MAE)
            predicted_value = np.expm1(xgb_tuned.predict(value_to_predict)[0])
            st.success(f'The predicted value is $ {round(predicted_value, 2)}')

            #Plot feature importance
            ft_weights_xgb_tuned = pd.DataFrame(xgb_tuned.feature_importances_, columns=['weight'], index=X.columns)
            ft_weights_xgb_tuned.sort_values('weight', inplace=True)
            fig = px.bar(ft_weights_xgb_tuned, x = 'weight', y = ft_weights_xgb_tuned.index, title="Feature Importances")
            st.write(fig)


    
    elif country == 'New York':
        st.sidebar.header('Select Input')
        room_type_list = ['Entire home/apt','Hotel room','Private room','Shared room']
        instant_booking_list = ['t','f']
        has_availability_list = ['t','f']
        superhost_list = ['t','f']
        host_response_time_list = ['a few days or more','within a day','within a few hours','within an hour']
        ny_neighborhood_list = ['Allerton', 'Baychester', 'Belmont', 
        'Castle Hill','City Island', 'Claremont Village', 'Clason Point', 'Concourse', 'Concourse Village', 'Co-op City', 'Country Club', 
        'Eastchester', 'East Morrisania', 'Edenwald', 'Fieldston','Fordham', 'Highbridge','Hunts Point','Kingsbridge','Longwood',
        'Melrose','Morrisania','Morris Heights','Morris Park','Mott Haven','Mount Eden','Mount Hope',
        'North Riverdale','Norwood','Olinville','Parkchester','Pelham Bay','Pelham Gardens','Port Morris',
        'Riverdale','Schuylerville','Soundview','Spuyten Duyvil','Throgs Neck','Tremont','Unionport','University Heights',
        'Van Nest','Wakefield','Westchester Square','West Farms','Williamsbridge','Woodlawn','Bath Beach','Bay Ridge',
        'Bedford-Stuyvesant','Bensonhurst','Bergen Beach','Boerum Hill','Borough Park','Brighton Beach','Brooklyn Heights',
        'Brownsville','Bushwick','Canarsie','Carroll Gardens','Clinton Hill','Cobble Hill','Columbia St','Coney Island','Crown Heights',
        'Cypress Hills','Downtown Brooklyn','DUMBO','Dyker Heights','East Flatbush','East New York','Flatbush','Flatlands',
        'Fort Greene','Fort Hamilton','Gerritsen Beach','Gowanus','Gravesend','Greenpoint','Kensington','Manhattan Beach',
        'Midwood','Mill Basin','Navy Yard','Park Slope','Prospect Heights','Prospect-Lefferts Gardens','Red Hook','Sea Gate','Sheepshead Bay',
        'South Slope','Sunset Park','Vinegar Hill','Williamsburg','Windsor Terrace','Battery Park City','Chelsea',
        'Chinatown','Civic Center','East Harlem','East Village','Financial District','Flatiron District',
        'Gramercy','Greenwich Village','Harlem','Hell\'s Kitchen','Inwood','Kips Bay','Little Italy','Lower East Side',
        'Marble Hill','Midtown','Morningside Heights','Murray Hill','NoHo','Nolita','Roosevelt Island','SoHo','Stuyvesant Town',
        'Theater District','Tribeca','Two Bridges','Upper East Side','Upper West Side','Washington Heights','West Village','Arverne',
        'Astoria','Bayside','Bayswater','Bay Terrace','Belle Harbor','Bellerose','Breezy Point','Briarwood','Cambria Heights',
        'College Point','Corona','Ditmars Steinway','Douglaston','East Elmhurst','Edgemere','Elmhurst','Far Rockaway','Flushing',
        'Forest Hills','Fresh Meadows','Glendale','Glen Oaks','Hollis','Hollis Hills','Holliswood','Howard Beach','Jackson Heights',
        'Jamaica','Jamaica Estates','Jamaica Hills','Kew Gardens','Kew Gardens Hills','Laurelton','Little Neck','Long Island City',
        'Maspeth','Middle Village','Neponsit','Ozone Park','Queens Village','Rego Park','Richmond Hill','Ridgewood','Rockaway Beach',
        'Rosedale','South Ozone Park','Springfield Gardens','St. Albans','Sunnyside','Whitestone','Woodhaven','Woodside','Arden Heights',
        'Arrochar','Bay Terrace, Staten Island','Bloomfield','Bull\'s Head','Castleton Corners','Charleston','Chelsea, Staten Island','Clifton',
        'Concord','Dongan Hills','Eltingville','Emerson Hill','Fort Wadsworth','Graniteville','Grant City','Great Kills','Grymes Hill',
        'Howland Hook','Huguenot','Lighthouse Hill','Mariners Harbor','Midland Beach','New Brighton','New Dorp','New Dorp Beach','New Springville',
        'Oakwood','Pleasant Plains','Port Ivory','Port Richmond','Prince\'s Bay','Randall Manor','Richmondtown','Rosebank','Rossville',
        'Shore Acres','Silver Lake','South Beach','Stapleton','St. George','Todt Hill','Tompkinsville','Tottenville','West Brighton','Westerleigh',
        'Willowbrook','Woodrow','Other']

        ny_neighborhood_choice = st.sidebar.selectbox('Neighborhood', ny_neighborhood_list)
        ny_neighborhood = data.loc[data.neighborhood == ny_neighborhood_choice, 'neighborhood_enc'].values[0]  

        room_type_choice = st.sidebar.selectbox(label='Room Type', options=room_type_list)
        room_type = data.loc[data.room_type == room_type_choice, 'room_type_enc'].values[0]

        accomodates = st.sidebar.slider('Number of guests', 0, 16, 7)

        bedroom = st.sidebar.slider('Number of bedrooms', 0, 22, 4)

        bathroom = st.sidebar.slider('Number of bathrooms', 0, 21, 10)

        bed = st.sidebar.slider('Number of beds', 0, 58, 7)

        instant_booking_choice = st.sidebar.selectbox(label='Instant_booking (t = True; f = False)', options=instant_booking_list)
        instant_booking = data.loc[data.instant_bookable == instant_booking_choice, 'instant_bookable_enc'].values[0]

        has_availability_choice = st.sidebar.selectbox(label='Has Availability (t = True; f = False)', options=has_availability_list)
        has_availability = data.loc[data.has_availability == has_availability_choice, 'has_availability_enc'].values[0]

        superhost_choice = st.sidebar.selectbox(label='Superhost (t = True; f = False)', options=superhost_list)
        superhost = data.loc[data.host_is_superhost == superhost_choice, 'host_is_superhost_enc'].values[0]

        host_response_time_choice = st.sidebar.selectbox(label='host_response_time', options=host_response_time_list)
        host_response_time = data.loc[data.host_response_time == host_response_time_choice, 'host_response_time_enc'].values[0]
        
        host_response_rate = st.sidebar.slider('Host Response Rate(%)', 0, 100, 78)   
        host_acceptance_rate = st.sidebar.slider('Host Acceptance Rate(%)', 0, 100, 92)

        user_input_num = {'host_response_rate': host_response_rate,
        'host_acceptance_rate':host_acceptance_rate, 
        'bedroom':bedroom,
        'bathroom':bathroom, 
        'bed':bed,
        'accomodates':accomodates}

        user_input_cat = {'ny_neighborhood':ny_neighborhood_choice,
        'room_type':room_type_choice,
        'instant_booking':instant_booking_choice,
        'has_availability':has_availability_choice,
        'is_superhost':superhost_choice,
        'host_response_time':host_response_time_choice}

        num_data, cat_data = st.columns(2)
        user_input_num = pd.DataFrame(user_input_num, index = ['Input Data'])
        user_input_num = user_input_num.T
        num_data.write(user_input_num)
        
        user_input_cat = pd.DataFrame(user_input_cat, index = ['Input Data'])
        user_input_cat = user_input_cat.T
        cat_data.write(user_input_cat)  

        predict_array = [host_response_rate,host_acceptance_rate, bedroom,bathroom, bed,accomodates,ny_neighborhood,room_type,instant_booking,has_availability,superhost,host_response_time]

        value_to_predict = pd.DataFrame([predict_array], columns=X.columns)

        xgb_tuned = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                gamma=0, gpu_id=-1, importance_type=None,
                interaction_constraints='', learning_rate=1, max_delta_step=0,
                max_depth=2, min_child_weight=3,monotone_constraints='()', n_estimators=900, n_jobs=8,
                num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
                reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
                validate_parameters=1, verbosity=None)

        xgb_tuned.fit(X_train.values, Y_train.values)     
        xgb_tuned_pred = xgb_tuned.predict(X_test.values)
        xgb_score = xgb_tuned.score(X_train.values, Y_train.values)
        xgb_MSE = mean_squared_error(Y_test, xgb_tuned_pred)
        xgb_MAE = mean_absolute_error(Y_test, xgb_tuned_pred)

        if st.button('Predict'):
            st.write('R2', xgb_score)
            st.write('MSE', xgb_MSE)
            st.write('MAE', xgb_MAE)
            predicted_value = np.expm1(xgb_tuned.predict(value_to_predict)[0])
            st.success(f'The predicted value is $ {round(predicted_value, 2)}')

            #Plot feature importance
            ft_weights_xgb_tuned = pd.DataFrame(xgb_tuned.feature_importances_, columns=['weight'], index=X.columns)
            ft_weights_xgb_tuned.sort_values('weight', inplace=True)
            fig = px.bar(ft_weights_xgb_tuned, x = 'weight', y = ft_weights_xgb_tuned.index, title="Feature Importances")
            st.write(fig)

    elif country == 'Singapore':
        st.sidebar.header('Select Input')
        room_type_list = ['Entire home/apt','Hotel room','Private room','Shared room']
        instant_booking_list = ['t','f']
        has_availability_list = ['t','f']
        superhost_list = ['t','f']
        host_response_time_list = ['a few days or more','within a day','within a few hours','within an hour']
        sg_neighborhood_list = ['Bishan', 'Bukit Merah', 'Bukit Timah','Downtown Core','Geylang','Kallang','Marina East','Marina South',
        'Marine Parade','Museum','Newton','Novena','Orchard','Outram','Queenstown','River Valley','Rochor','Singapore River','Southern Islands',
        'Straits View','Tanglin','Toa Payoh','Bedok','Changi','Changi Bay','Pasir Ris','Paya Lebar','Tampines','Ang Mo Kio','Hougang',
        'North-Eastern Islands','Punggol','Seletar','Sengkang','Serangoon','Central Water Catchment','Lim Chu Kang','Mandai','Sembawang',
        'Simpang','Sungei Kadut','Woodlands','Yishun','Boon Lay','Bukit Batok','Bukit Panjang','Choa Chu Kang','Clementi','Jurong East',
        'Jurong West','Pioneer','Tengah','Tuas','Western Islands','Western Water Catchment','Other']

        sg_neighborhood_choice = st.sidebar.selectbox('Neighborhood', sg_neighborhood_list)
        sg_neighborhood = data.loc[data.neighborhood == sg_neighborhood_choice, 'neighborhood_enc'].values[0]  

        room_type_choice = st.sidebar.selectbox(label='Room Type', options=room_type_list)
        room_type = data.loc[data.room_type == room_type_choice, 'room_type_enc'].values[0]

        accomodates = st.sidebar.slider('Number of guests', 0, 16, 7)

        bedroom = st.sidebar.slider('Number of bedrooms', 0, 22, 4)

        bathroom = st.sidebar.slider('Number of bathrooms', 0, 21, 10)

        bed = st.sidebar.slider('Number of beds', 0, 58, 7)

        instant_booking_choice = st.sidebar.selectbox(label='Instant_booking (t = True; f = False)', options=instant_booking_list)
        instant_booking = data.loc[data.instant_bookable == instant_booking_choice, 'instant_bookable_enc'].values[0]

        has_availability_choice = st.sidebar.selectbox(label='Has Availability (t = True; f = False)', options=has_availability_list)
        has_availability = data.loc[data.has_availability == has_availability_choice, 'has_availability_enc'].values[0]

        superhost_choice = st.sidebar.selectbox(label='Superhost (t = True; f = False)', options=superhost_list)
        superhost = data.loc[data.host_is_superhost == superhost_choice, 'host_is_superhost_enc'].values[0]

        host_response_time_choice = st.sidebar.selectbox(label='host_response_time', options=host_response_time_list)
        host_response_time = data.loc[data.host_response_time == host_response_time_choice, 'host_response_time_enc'].values[0]
        
        host_response_rate = st.sidebar.slider('Host Response Rate(%)', 0, 100, 86)   
        host_acceptance_rate = st.sidebar.slider('Host Acceptance Rate(%)', 0, 100, 73)

        user_input_num = {'host_response_rate': host_response_rate,
        'host_acceptance_rate':host_acceptance_rate, 
        'bedroom':bedroom,
        'bathroom':bathroom, 
        'bed':bed,
        'accomodates':accomodates}

        user_input_cat = {'sg_neighborhood':sg_neighborhood_choice,
        'room_type':room_type_choice,
        'instant_booking':instant_booking_choice,
        'has_availability':has_availability_choice,
        'is_superhost':superhost_choice,
        'host_response_time':host_response_time_choice}

        num_data, cat_data = st.columns(2)
        user_input_num = pd.DataFrame(user_input_num, index = ['Input Data'])
        user_input_num = user_input_num.T
        num_data.write(user_input_num)
        
        user_input_cat = pd.DataFrame(user_input_cat, index = ['Input Data'])
        user_input_cat = user_input_cat.T
        cat_data.write(user_input_cat)  

        predict_array = [host_response_rate,host_acceptance_rate, bedroom,bathroom, bed,accomodates,sg_neighborhood,room_type,instant_booking,has_availability,superhost,host_response_time]

        value_to_predict = pd.DataFrame([predict_array], columns=X.columns)

        xgb_tuned = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                gamma=0, gpu_id=-1, importance_type=None,
                interaction_constraints='', learning_rate=1, max_delta_step=0,
                max_depth=2, min_child_weight=3,monotone_constraints='()', n_estimators=900, n_jobs=8,
                num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
                reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
                validate_parameters=1, verbosity=None)

        xgb_tuned.fit(X_train.values, Y_train.values)     
        xgb_tuned_pred = xgb_tuned.predict(X_test.values)
        xgb_score = xgb_tuned.score(X_train.values, Y_train.values)
        xgb_MSE = mean_squared_error(Y_test, xgb_tuned_pred)
        xgb_MAE = mean_absolute_error(Y_test, xgb_tuned_pred)

        if st.button('Predict'):
            st.write('R2', xgb_score)
            st.write('MSE', xgb_MSE)
            st.write('MAE', xgb_MAE)
            predicted_value = np.expm1(xgb_tuned.predict(value_to_predict)[0])
            st.success(f'The predicted value is $ {round(predicted_value, 2)}')

            #Plot feature importance
            ft_weights_xgb_tuned = pd.DataFrame(xgb_tuned.feature_importances_, columns=['weight'], index=X.columns)
            ft_weights_xgb_tuned.sort_values('weight', inplace=True)
            fig = px.bar(ft_weights_xgb_tuned, x = 'weight', y = ft_weights_xgb_tuned.index, title="Feature Importances")
            st.write(fig)



        




        
