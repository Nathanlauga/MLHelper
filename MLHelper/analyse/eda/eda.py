import pandas as pd
import numpy as np
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from facets_overview.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator

from ...utils import *


def print_missing_values(df):
    """
    Show a bar plot that display percentage of missing values on columns that have some.
    If no missing value then it use `display` & `Markdown` functions to indicate it.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    """
    df_null = pd.DataFrame(len(df) - df.notnull().sum(), columns=['Count'])
    df_null = df_null[df_null['Count'] > 0].sort_values(
        by='Count', ascending=False)
    df_null = df_null/len(df)*100

    if len(df_null) == 0:
        display(Markdown('No missing value.'))
        return

    x = df_null.index.values
    height = [e[0] for e in df_null.values]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(x, height, width=0.8)
    plt.xticks(x, x, rotation=60)
    plt.xlabel('Columns')
    plt.ylabel('Percentage')
    plt.title('Percentage of missing values in columns')
    plt.show()


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


def stack_plot(ax, tab, labels):
    """
    Add a stack plot into the current ax plot
    """
    colors = sns.color_palette("colorblind", len(labels))
    x = tab.index.values
    y = [list(tab[str(l)].values) for l in labels]

    ax.stackplot(x, y, labels=labels, colors=colors)
    ax.legend(loc=0, frameon=True)


def stack_bar_plot(ax, tab, labels, legend_labels):
    """
    Add a stack bar plot into the current ax plot
    """
    colors = sns.color_palette("colorblind", len(legend_labels))

    for i, row in tab.iterrows():
        tab.loc[i] = ((row / row.sum()).round(2)*100).astype(int)

    for i, l in enumerate(legend_labels):
        bottom = tab[legend_labels[i-1]] if i > 0 else None

        rects = ax.bar(labels, tab[l], label=l, bottom=bottom,
                       align='center', color=colors[i])

        for r in rects:
            h, w, x, y = r.get_height(), r.get_width(), r.get_x(), r.get_y()
            if h == 0:
                continue

            ax.annotate(str(h)+'%', xy=(x+w/2, y+h/2), xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='center',
                        color='white', weight='bold', clip_on=True)

    ax.legend(loc=0, frameon=True)


def show_numerical_var(df, var, target=None):
    """
    Show variable information in graphics for numerical variables.
    At least the displot & boxplot. 

    If target is set 2 more plots : stack plot and stack plot with percentage

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    var: str
        Column name that contains numerical values
    target: str (optional)
        Target column for classifier
    """
    fig, ax = plt.subplots(figsize=(16, 5))

    ax = plt.subplot(121)
    sns.distplot(df[var])

    ax = plt.subplot(122)
    x = df[target] if target != None else None

    sns.boxplot(x=x, y=df[var])

    if target != None:
        fig, ax = plt.subplots(figsize=(16, 5))
        labels = sorted(df[target].unique())
        tab = pd.crosstab(df[var], df[target])

        ax = plt.subplot(121)
        stack_plot(ax=ax, tab=tab, labels=labels)

        for i, row in tab.iterrows():
            tab.loc[i] = (row / row.sum()).round(2)
        ax = plt.subplot(122)
        stack_plot(ax=ax, tab=tab, labels=labels)

    plt.show()


def show_categorical_var(df, var, target=None):
    """
    Show variable information in graphics for categorical variables.
    For 10 most frequents values : bar plot and a pie chart

    If target is set : plot stack bar for 10 most frequents values

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    var: str
        Column name that contains categorical values
    target: str (optional)
        Target column for classifier
    """
    val_cnt = df[var].value_counts()
    if len(val_cnt) > 10:
        val_cnt = val_cnt.head(10)

    labels = val_cnt.index
    sizes = val_cnt.values
    colors = sns.color_palette("Blues", len(labels))

    fig, ax = plt.subplots(figsize=(16, 5))

    ax = plt.subplot(121)
    ax.bar(labels, sizes, width=0.8)
    plt.xticks(labels, labels, rotation=60)

    ax = plt.subplot(122)
    ax.pie(sizes, labels=labels, colors=colors,
           autopct='%1.0f%%', shadow=True, startangle=130)
    ax.axis('equal')
    ax.legend(loc=0, frameon=True)

    if target != None:
        fig, ax = plt.subplots(figsize=(16, 5))

        legend_labels = sorted(df[target].unique())
        tab = pd.crosstab(df[var], df[target])
        tab = tab.loc[labels]
        stack_bar_plot(ax=ax, tab=tab, labels=labels,
                       legend_labels=legend_labels)

    plt.show()


def show_meta_var(df, var):
    """
    Display some meta informations about a specific variable of a given dataframe
    Meta informations : # of null values, # of uniques values and 2 most frequent values

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    var: str
        Column name that is in df        
    """
    nb_null = df[var].isnull().sum()
    nb_uniq = df[var].nunique()
    most_freq = df[var].value_counts().head(2).to_dict()
    display(Markdown(
        f'**{var} :** {nb_null} nulls, {nb_uniq} unique vals, most common: {most_freq}'))


def show_df_vars(df, target=None):
    """
    Show all variables with graphics to understand each variable.
    If target is set, complement visuals will be added to take a look on the
    influence that a variable can have on target

    Data type handle : categorical, numerical

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    target: str (optional)
        Target column for classifier 
    """
    cat_vars = df.select_dtypes('object')
    num_vars = df.select_dtypes('number')

    display(Markdown('### Numerical variables'))
    for var in num_vars:
        display(Markdown('*****'))
        show_meta_var(df, var)
        show_numerical_var(df, var, target)

    display(Markdown('### Categorical variables'))
    for var in cat_vars:
        display(Markdown('*****'))
        show_meta_var(df, var)
        show_categorical_var(df, var, target)
