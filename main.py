#main file
import streamlit as st
import time
import requests
import pandas as pd
from requests.exceptions import SSLError

def get_request(url,parameters=None, steamspy=False):
    """Return json-formatted response of a get request using optional parameters.
    
    Parameters
    ----------
    url : string
    parameters : {'parameter': 'value'}
        parameters to pass as part of get request
    
    Returns
    -------
    json_data
        json-formatted response (dict-like)
    """
    try:
        response = requests.get(url=url, params=parameters)
    except SSLError as s:
        print('SSL Error:', s)
        
        for i in range(5, 0, -1):
            print('\rWaiting... ({})'.format(i), end='')
            time.sleep(1)
        print('\rRetrying.' + ' '*10)
        
        # recursively try again
        return get_request(url, parameters, steamspy)
    
    if response:
        return response.json()
    else:
        # We do not know how many pages steamspy has... and it seems to work well, so we will use no response to stop.
        if steamspy:
            return "stop"
        else :
            # response is none usually means too many requests. Wait and try again 
            print('No response, waiting 10 seconds...')
            time.sleep(10)
            print('Retrying.')
            return get_request(url, parameters, steamspy)
        
@st.cache_data
def get_steam_df():
    """Return a list of all steam games.
    
    Returns
    -------
    
        list of all steam games
    """
    return get_request("https://api.steampowered.com/ISteamApps/GetAppList/v2/?")["applist"]["apps"]

st.write("Search for a Steam Game")
search_input = st.text_input("Search Steam Game", key="search_input")
df = pd.DataFrame(get_steam_df())
df = df[df["name"].str.contains(search_input, case=False, na=False)]
st.dataframe(df)