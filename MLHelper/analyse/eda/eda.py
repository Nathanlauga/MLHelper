import pandas as pd
import numpy as np
import base64
from facets_overview.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator

from ...utils import *


def generate_facets_overview_html(proto_list: list, _id='facets'):
    """
    Generate a html string that contains facets overview graphics given 
    a formated proto_list
    
    Parameters
    ----------
    proto_list: list
        list with data split by target column or not formated as {'name':'value', 'table':DataFrame}
    _id: str
        id for html div
        
    Returns
    -------
    str:
        string that contains facets overview html
    """
    gfsg = GenericFeatureStatisticsGenerator()
    proto = gfsg.ProtoFromDataFrames(proto_list)
    protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")

    HTML_TEMPLATE = f"""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
    <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html" >
    <facets-overview id="{_id}"></facets-overview>
    <script>
        document.querySelector("#{_id}").protoInput = "{protostr}";
    </script>"""
    html = HTML_TEMPLATE.format(protostr=protostr)
    return html


def facets_overview_html(df: pd.DataFrame, target=None, dtype=None):
    """
    Generate a html string that contains facets overview graphics given a 
    dataframe, you can possibly split by the target column or extract 
    for a specific dtype
    
    Parameters
    ----------
    df: pandas.DataFrame
        dataframe to analyse
    target: str
        target column
    dtype:
        specific data type to analyse

    Returns
    -------
    str:
        string that contains facets overview html
    """
    if dtype != None:
        df = df.select_dtypes(dtype)
    else:
        dtype = 'facets'

    proto_list = list()

    if target == None:
        proto_list.append({'name': 'data', 'table': df})
    else:
        for unique_val in data[target].unique():
            filter_data = data[data[target] == unique_val]
            proto_list.append({'name': unique_val, 'table': filter_data})
        del filter_data

    html = generate_facets_overview_html(proto_list, _id=dtype)
    return html


def facets_overview_to_file(df: pd.DataFrame, fname: str, target=None, dtype=None):
    """
    Generate a html file that contains facets overview graphics given a 
    dataframe, you can possibly split by the target column or extract 
    for a specific dtype

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe to analyse
    fname: str:
        file name (with the html extension e.g. file.html)
    target: str
        target column
    dtype:
        specific data type to analyse
    """
    html = facets_overview_html(df, target=target, dtype=dtype)
    str_to_file(html, fname=fname)
