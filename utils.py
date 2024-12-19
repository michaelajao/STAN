import io
import logging
import os
import pytz
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from epiweeks import Week
from haversine import haversine

requests.packages.urllib3.disable_warnings()

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

DATE_FORMAT = "%Y-%m-%d"
date_today = datetime.now(tz=pytz.timezone('US/Eastern')).strftime(DATE_FORMAT)
DATA_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def get_data_location(file_name, folder=None):
    """
    Generates the path for a file to be read.

    Args:
        file_name (str): Name of the file.
        folder (str, optional): Folder name. Defaults to None.

    Returns:
        str: Path to the file.
    """
    return os.path.join(DATA_LOCATION, file_name) if folder is None else os.path.join(DATA_LOCATION, folder, file_name)

def check_url(url):
    """
    Checks the existence of a URL.

    Args:
        url (str): URL to check.

    Returns:
        bool: True if the URL exists, False otherwise.
    """
    try:
        request = requests.get(url, verify=False)
        if request.status_code < 400:
            return True
        else:
            logging.info(f"URL for {url.split('/')[-1]} does not exist!")
            return False
    except requests.exceptions.RequestException as e:
        logging.info(f"Error checking URL {url}: {e}")
        return False

def download_data(url):
    """
    Downloads CSV files from GitHub.

    Args:
        url (str): URL of the CSV file.

    Returns:
        pandas.DataFrame: Content of the CSV file.
    """
    if check_url(url):
        x = requests.get(url=url, verify=False).content
        df = pd.read_csv(io.StringIO(x.decode('utf8')))
        return df
    else:
        logging.error(f"Failed to download data from {url}")
        return None

def calculate_ccc(y_true, y_pred):
    """
    Calculates the concordance correlation coefficient (CCC) between two vectors.

    Args:
        y_true (numpy.ndarray): Real data.
        y_pred (numpy.ndarray): Estimated data.

    Returns:
        float: CCC value.
    """
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator

def gravity_law_commute_dist(lat1, lng1, pop1, lat2, lng2, pop2, r=1):
    """
    Calculates the edge via the gravity law.

    Args:
        lat1 (float): Latitude of location 1.
        lng1 (float): Longitude of location 1.
        pop1 (float or int): Population of location 1.
        lat2 (float): Latitude of location 2.
        lng2 (float): Longitude of location 2.
        pop2 (float or int): Population of location 2.
        r (float, optional): Diameter. Defaults to 1.

    Returns:
        float: Edge value.
    """
    d = haversine((lat1, lng1), (lat2, lng2), unit='km')
    alpha = 0.1
    beta = 0.1
    r = 1e4
    w = (np.exp(-d / r)) / (abs((pop1 ** alpha) - (pop2 ** beta)) + 1e-5)
    return w

def envelope(x):
    """
    Calculates the envelope of a signal.

    Args:
        x (numpy.array or list): Input data.

    Returns:
        numpy.array or list: Envelope of the signal.
    """
    x = x.copy()
    for i in range(len(x) - 1):
        a = x[i]
        b = x[i + 1]
        if b < a:
            x[i + 1] = a
    return x

def map_to_week(df, date_column='date_today', groupby_target=None):
    """
    Maps 'date_today' to 'week_id'.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.
        date_column (str, optional): Column name related to 'date_today'. Defaults to 'date_today'.
        groupby_target (str or list, optional): Columns to group by. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame with 'week_id'.
    """
    df[date_column] = df[date_column].apply(lambda x: Week.fromdate(x).enddate() if pd.notna(x) else x)
    df[date_column] = pd.to_datetime(df[date_column])
    if groupby_target is not None:
        df = df.groupby('date_today', as_index=False)[groupby_target].sum()
    return df
