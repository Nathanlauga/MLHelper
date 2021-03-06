import pandas as pd
import numpy as np
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder
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

    # tab.div(tab.sum(axis=1), axis=0).round(2).astype(int)
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
    if target == None:
        sns.distplot(df[var])
    else:
        labels = sorted(df[target].unique())
        for l in labels:
            df_target = df[df[target] == l]
            if df_target[var].nunique() <= 1:
                sns.distplot(df_target[var], kde=False)
            else:
                sns.distplot(df_target[var])
            del df_target

    ax = plt.subplot(122)
    x = df[target] if target != None else None

    sns.boxplot(x=x, y=df[var])
    
    if target != None:
        fig, ax = plt.subplots(figsize=(16, 5))
        tab = pd.crosstab(df[var], df[target])
        
        ax = plt.subplot(121)
        stack_plot(ax=ax, tab=tab, labels=labels)

        tab.div(tab.sum(axis=1), axis=0)

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


def show_datetime_var(df, var, target=None):
    """
    Show variable information in graphics for datetime variables.
    Display only the time series line if no target is set else, it shows
    2 graphics one with differents lines by value of target and one stack line plot

    If difference between maximum date and minimum date is above 1000 then plot by year.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    var: str
        Column name that contains datetime values
    target: str (optional)
        Target column for classifier
    """
    df = df.copy()
    fig, ax = plt.subplots(figsize=(16, 5))

    date_min = df[var].min()
    date_max = df[var].max()
    if (date_max - date_min).days > 1000:
        df[var] = df[var].dt.year

    if target == None:
        val_cnt = df[var].value_counts()
        sns.lineplot(data=val_cnt)
    else:
        ax = plt.subplot(121)

        legend_labels = sorted(df[target].unique())
        tab = pd.crosstab(df[var], df[target])
        sns.lineplot(data=tab)

        tab.div(tab.sum(axis=1), axis=0)

        ax = plt.subplot(122)
        stack_plot(ax=ax, tab=tab, labels=legend_labels)

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

    Data type handle : categorical, numerical, datetime

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    target: str (optional)
        Target column for classifier 
    """
    cat_vars = df.select_dtypes('object')
    num_vars = df.select_dtypes('number')
    dat_vars = df.select_dtypes('datetime')

    display(Markdown('### Numerical variables'))
    for var in num_vars:
        display(Markdown('*****'))
        show_meta_var(df, var)
        if len(df[var].unique()) <= 1:
            display('Only one value.')
            continue
        show_numerical_var(df, var, target)

    display(Markdown('### Categorical variables'))
    for var in cat_vars:
        display(Markdown('*****'))
        show_meta_var(df, var)
        if len(df[var].unique()) <= 1:
            display('Only one value.')
            continue
        show_categorical_var(df, var, target)

    display(Markdown('### Datetime variables'))
    for var in dat_vars:
        display(Markdown('*****'))
        show_meta_var(df, var)
        if len(df[var].unique()) <= 1:
            display('Only one value.')
            continue
        show_datetime_var(df, var, target)


def show_numerical_jointplot(df, var1, var2, target=None):
    """
    Show two numerical variables relations with jointplot.
    
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    var1: str
        Column name that contains first numerical values
    var2: str
        Column name that contains second numerical values
    target: str (optional)
        Target column for classifier
    """
    if target == None:
        g = sns.jointplot(var1, var2, data=df, kind="hex", space=0, height=8)
    else:
        legend_labels = sorted(df[target].unique())
        grid = sns.JointGrid(x=var1, y=var2, data=df, height=7)

        g = grid.plot_joint(sns.scatterplot, hue=target, data=df, alpha=0.3)
        for l in legend_labels:
            sns.distplot(df.loc[df[target]==l, var1], ax=g.ax_marg_x)
            sns.distplot(df.loc[df[target]==l, var2], ax=g.ax_marg_y, vertical=True)
    plt.show()

    
def show_df_numerical_relations(df, target=None):
    """
    Show all numerical variables 2 by 2 with graphics understand their relation.
    If target is set, separate dataset for each target value.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    target: str (optional)
        Target column for classifier 
    """
    num_vars = df.select_dtypes('number')
    num_vars = remove_var_with_one_value(num_vars)

    cols = num_vars.columns.values
    var_combi = [tuple(sorted([v1, v2])) for v1 in cols for v2 in cols if v1 != v2]
    var_combi = list(set(var_combi))

    for var1, var2 in var_combi:
        display(Markdown('*****'))
        display(Markdown(f'Joint plot for **{var1}** & **{var2}**'))
        show_numerical_jointplot(df=df, var1=var1, var2=var2, target=target)        


def show_barplot_cat_num_var(df, cat_var, num_var, target=None):
    """
    Show boxplots for a specific pair of categorical and numerical variables
    If target is set, separate dataset for each target value.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    cat_var: str
        Column name that contains categorical values
    num_var: str
        Column name that contains numerical values
    target: str (optional)
        Target column for classifier
    """    
    val_cnt = df[cat_var].value_counts()
    if len(val_cnt) > 10:
        val_cnt = val_cnt.head(10)
        
    df_plot = df[df[cat_var].apply(lambda x: x in val_cnt.index.values)]
    
    fig, ax = plt.subplots(figsize=(16, 5))
    palette = "Blues" if target == None else "colorblind"
    sns.boxplot(x=cat_var, y=num_var, hue=target, data=df_plot, palette=palette)
    plt.xticks(rotation=60)
    plt.show()
    

def show_df_num_cat_relations(df, target=None):
    """
    Show boxplots for each pair of categorical and numerical variables
    If target is set, separate dataset for each target value.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    target: str (optional)
        Target column for classifier 
    """
    df = df.copy()
    df = remove_var_with_one_value(df)
    
    num_vars = df.select_dtypes('number')
    cat_vars = df.select_dtypes('object')
    
    var_combi = [(v1, v2) for v1 in num_vars.columns for v2 in cat_vars.columns if (v1 != v2) & (v2 != target)]
    
    for num_var, cat_var in var_combi:
        display(Markdown('*****'))
        display(Markdown(f'Box plot for **{cat_var}** & **{num_var}**'))
        show_barplot_cat_num_var(df=df, cat_var=cat_var, num_var=num_var, target=target)

    
def plot_correlation_matrix(corr_df):
    """
    Plot a seaborn heatmap based on a correlation dataframe.
    
    Parameters
    ----------
    corr_df: pd.DataFrame
        Correlation dataframe
    """
    fig, ax = plt.subplots(figsize=(10,9))
    ax = sns.heatmap(
        corr_df, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(220, 20, n=200),
        square=True,
        annot=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );    
    plt.show()
    
    

def cramers_v(x, y):
    """
    Function that return the Cramer V value for two categorical variables using
    chi square. This correlation metric is between 0 and 1.
    
    Code source found on this article : 
    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    
    Parameters
    ----------
    x:
        first categorical variable
    y:
        second categorical variable
        
    Returns
    -------
    float:
        Cramer V value
    """
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def encode_categorical_vars(df):
    """
    Encode categorical variables from a dataframe to be numerical (discrete)
    It uses LabelEncoder class from scikit-learn
    
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to update
    
    Returns
    -------
    pd.DataFrame
        Encoded dataframe
    """
    cat_vars = df.select_dtypes('object').columns
    data_encoded = df.copy()
    
    # Use Label Encoder for categorical columns (including target column)
    for feature in cat_vars:
        le = LabelEncoder()
        le.fit(data_encoded[feature])

        data_encoded[feature] = le.transform(data_encoded[feature])
    
    return data_encoded


def init_corr_matrix(columns, index, fill_diag=1.):
    """
    Return a matrix n by m fill of 0 (except on the diagonal if squared matrix)
    Recommended for correlation matrix
    
    Parameters
    ----------
    columns: 
        list of columns names
    index:
        list of index names
    fill_diag: float
        if squared matrix then set diagonal with this value
        
    Returns
    -------
    pd.DataFrame
        Initialized matrix
    """
    zeros = np.zeros((len(index),len(columns)), float)
    if len(columns) == len(index):
        rng = np.arange(len(zeros))
        zeros[rng, rng] = fill_diag
    return pd.DataFrame(zeros, columns=columns, index=index)


def show_df_correlations(df):
    """
    Show differents correlations matrix for 3 cases :
    - numerical to numerical (using Pearson coeff)
    - categorical to categorical (using Cramers V & Chi square)
    - numerical to categorical (discrete) (using Point Biserial)

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    """
    df = df.copy()
    df = remove_var_with_one_value(df)
    
    num_df = df.select_dtypes('number')
    cat_df = df.select_dtypes('object')
    num_vars = num_df.columns
    cat_vars = cat_df.columns
    
    pearson_corr = num_df.corr()
    display(Markdown('#### Pearson correlation matrix for numerical variables'))
    plot_correlation_matrix(pearson_corr)
        
    var_combi = [tuple(sorted([v1, v2])) for v1 in cat_vars for v2 in cat_vars if v1 != v2]
    var_combi = list(set(var_combi))
    
    cramers_v_corr = init_corr_matrix(columns=cat_vars, index=cat_vars)
   
    for var1, var2 in var_combi:
        corr = cramers_v(cat_df[var1],cat_df[var2])
        cramers_v_corr.loc[var1, var2] = corr
        cramers_v_corr.loc[var2, var1] = corr
    
    display(Markdown('#### Cramers V correlation matrix for categorical variables'))
    plot_correlation_matrix(cramers_v_corr)
        
    data_encoded = encode_categorical_vars(df)
#     pearson_corr = data_encoded.corr()
#     display(Markdown('#### Pearson correlation matrix for categorical variables'))
#     plot_correlation_matrix(pearson_corr)

    var_combi = [(v1, v2) for v1 in cat_vars for v2 in num_vars if v1 != v2]
    
    pbs_corr = init_corr_matrix(columns=num_vars, index=cat_vars, fill_diag=0.)
    
    for cat_var, num_var in var_combi:
        corr, p_value = ss.pointbiserialr(data_encoded[cat_var], data_encoded[num_var])
        pbs_corr.loc[cat_var, num_var] = corr
    
    display(Markdown('#### Point Biserial correlation matrix for numerical & categorical variables'))
    plot_correlation_matrix(pbs_corr)