"""
SpiderSensei-Performance-Tracker (ALPHA version)
------------------------------------------------
IMPORTANT: This script is currently under active development.
The code is experimental, not fully optimized, and may lack proper cleaning or readability.
Please use with caution as features may change, and bugs or inefficiencies may exist.

NOTE: Contributions or suggestions for improvement are welcome!
"""

from telethon import TelegramClient
import csv
import datetime
import re
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
import requests
from io import BytesIO
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.graphics.shapes import Drawing, String
import time
import mplfinance as mpf
import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Telegram API credentials
api_id = '24770422'  # obtained from my.telegram.org
api_hash = 'ca5c7a4029912c4516b0a8a86d6770bd'  # obtained from my.telegram.org

# Create the client and connect
client = TelegramClient('session_name', api_id, api_hash)

# Function to parse relative dates
def parse_relative_date(text):
    """
    Converts relative date text (e.g., '2 days ago') into the difference in hours.
    
    Parameters:
        text (str): Relative time description like '2 days', '1 week', etc.
    
    Returns:
        int: Difference in hours from the current time.
    """
    now = datetime.datetime.now()
    # Simplified mapping of units
    units = {'year': 'years', 'month': 'months', 'week': 'weeks', 'day': 'days',
             'hour': 'hours', 'minute': 'minutes', 'second': 'seconds'}
    try:
        parts = text.split()
        number = 1 if parts[0] in ['a', 'an'] else int(parts[0])
        unit = units.get(parts[1].rstrip(',s'), None)  # Handle singular/plural and trailing commas
        if not unit:
            raise ValueError(f"Unrecognized time unit: {parts[1]}")
        past_date = now - relativedelta(**{unit: number})
        return int((now - past_date).total_seconds() / 3600)
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid date text format: {text}") from e

# Function to extract data from messages
def extract_info(filename):
    """
    Extracts structured data from a text file and handles image downloads.
    
    Parameters:
        filename (str): Path to the text file containing messages.
    
    Returns:
        list: A list of extracted data entries.
    """
    image_folder = 'saved_pics'
    os.makedirs(image_folder, exist_ok=True)
    
    field_mapping = {
        '**Name**': ('name', 10),
        '**FDV**': ('fdv', 8),
        '**Followers**': ('followers_now', 15),
        '**First discovered followers**': ('followers_discovered', 32),
        '**Following**': ('following', 15),
        '**Created**': ('created_in_hours', 13),
        '**Tweets**': ('tweets', 12),
        '**5K+ Followers**': ('5k_followers', 19),
        '**VC Followers**': ('vc_followers', 18),
    }
    
    entries = []
    current_data = {
        "timestamp": '', "name": 'n/a', "fdv": 'n/a',
        "followers_now": 'n/a', "followers_discovered": 'n/a',
        "following": 'n/a', "created_in_hours": 'n/a',
        "tweets": 'n/a', "5k_followers": 'n/a', "vc_followers": 'n/a',
        "chad_followers_list": [], "chad_followers_count": 'n/a',
        "time_to_curated": 'n/a', "contract_created": 'n/a',
        "contract": 'n/a', "website": 0,
        "x_link": 'n/a',
        "pic_filename": 'n/a',
        "top_holders_1": 'n/a', "top_holders_2": 'n/a',
        "top_holders_3": 'n/a', "top_holders_4": 'n/a',
        "top_holders_5": 'n/a', "total_top_holders": 'n/a'
    }
    
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    for line in lines:
        line = line.strip()

        # Identify start of a new entry (either with an image or without)
        if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2}:', line):
            if current_data["timestamp"] and current_data["name"] != 'n/a':
                entries.append(list(current_data.values()))
            
            current_data = {k: 'n/a' for k in current_data}  # Reset
            current_data["timestamp"] = line[:25].strip()
            
            # Check if picture is in the first line
            pic_link = re.search(r'\((https?://pbs\.twimg\.com/[^)]+)\)', line)
            if pic_link:
                extracted_pic_link = pic_link.group(1)
                file_name = os.path.join(image_folder, extracted_pic_link.split('/')[-1])
                current_data["pic_filename"] = extracted_pic_link.split('/')[-1]
                try:
                    response = requests.get(extracted_pic_link)
                    if response.status_code == 200:
                        with open(file_name, 'wb') as img_file:
                            img_file.write(response.content)
                except Exception as e:
                    print(f"Error downloading image: {e}")
                continue  # Skip to next line after processing image
            
        # Handle secondary image location in **Search** line if not found before
        elif '**Search**' in line and current_data["pic_filename"] == 'n/a':
            pic_link = re.search(r'\((https?://pbs\.twimg\.com/[^)]+)\)', line)
            if pic_link:
                extracted_pic_link = pic_link.group(1)
                file_name = os.path.join(image_folder, extracted_pic_link.split('/')[-1])
                current_data["pic_filename"] = extracted_pic_link.split('/')[-1]
                try:
                    response = requests.get(extracted_pic_link)
                    if response.status_code == 200:
                        with open(file_name, 'wb') as img_file:
                            img_file.write(response.content)
                except Exception as e:
                    print(f"Error downloading image: {e}")

        # Handle social media link
        elif '**[**' in line:
            x_link = re.search(r'\((https?://(?:x\.com|twitter\.com)/[^\)]+)\)', line)
            if x_link:
                current_data["x_link"] = x_link.group(1)

        # Parse **TH** (Top Holders)
        elif '**TH**' in line:
            values = re.findall(r"\[(.*?)\]", line)
            for i in range(5):
                current_data[f"top_holders_{i+1}"] = values[i] if i < len(values) else 'n/a'
            current_data["total_top_holders"] = values[-1] if values else 'n/a'
        
        # Use field mapping for standard fields
        elif any(field in line for field in field_mapping):
            for field, (key, offset) in field_mapping.items():
                if field in line:
                    current_data[key] = line[offset:].strip()
        
        # Handle additional cases
        elif '**Chad Followers**' in line or '**Notable Followers**' in line:
            matches = re.findall(r'\[([^\]]+)\]\(https?://twitter\.com/[^)]+\)', line)
            current_data["chad_followers_list"] = [match.strip() for match in matches if match.strip()]
        elif '**Time to Curated**' in line:
            time_curated = re.findall(r'\d+', line[21:].strip())
            current_data["time_to_curated"] = int(time_curated[0]) if time_curated else 0
        elif '**Contract Created' in line:
            date_str = line.split(': ')[1].split('**')[0].strip()
            try:
                date_obj = datetime.datetime.strptime(date_str, '%d/%m/%Y')
                current_data["contract_created"] = date_obj.strftime('%Y-%m-%d')
            except ValueError as e:
                print(f"Error parsing contract creation date: {e}")
        elif '**Contract**' in line:
            current_data["contract"] = line[15:].strip()
        elif '**Website**' in line and 'pump.fun' not in line:
            current_data["website"] = line[13:].strip()
        
        # Update Chad followers count
        current_data["chad_followers_count"] = len(current_data["chad_followers_list"])
    
    if current_data["timestamp"] and current_data["name"] != 'n/a':
        entries.append(list(current_data.values()))
    
    return entries

def save_to_csv(filename, entries):
    """
    Saves extracted data into a CSV file.

    Parameters:
        filename (str): Path to the output CSV file.
        entries (list): List of extracted data entries, where each entry is a list of values.

    Returns:
        bool: True if the CSV is successfully saved, False otherwise.
    """
    try:
        # Validate that there are entries to write
        if not entries:
            print("No data to save to CSV.")
            return False

        # Dynamically determine headers from the first entry
        headers = [
            'Timestamp', 'Name', 'FDV', 'Followers_now', 'Followers_discovered',
            'Following', 'Created_x_hours_ago', 'Tweets', '5k+_Followers',
            'VC_follower', 'Chad_followers', 'Chad_followers_count',
            'Time_to_Curated', 'Contract_Created', 'Contract', 'Website', 'x_link', 'pic_filename', 'top_holders_1', 'top_holders_2',
            'top_holders_3', 'top_holders_4', 'top_holders_5', 'total_top_holders'
        ]

        # Validate the length of each entry matches the header length
        for entry in entries:
            if len(entry) != len(headers):
                print(f"Invalid entry found: {entry}. Skipping.")
                entries.remove(entry)

        # Write data to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(entries)

        print(f"Data successfully saved to {filename}")
        return True
    except Exception as e:
        print(f"Failed to save data to CSV: {e}")
        return False

def make_request(url, request_tracker):
    """
    Makes an API request, respecting rate limits.

    Parameters:
        url (str): The API URL to fetch data from.
        request_tracker (dict): Tracker to monitor API request counts.

    Returns:
        Response: The API response object, or None if the request fails.
    """
    try:
        # Check API limit and pause if necessary
        if request_tracker['count'] >= 20:
            print("API request limit reached. Pausing for 1 minute...")
            time.sleep(60)
            request_tracker['count'] = 0

        # Make the request
        response = requests.get(url)
        request_tracker['count'] += 1  # Increment request count

        # Return the response
        if response.status_code == 200:
            return response
        else:
            print(f"API request failed: {response.status_code}, URL: {url}")
            return None
    except Exception as e:
        print(f"Error in make_request: {e}")
        return None


def fetch_liquidity_pool(token, chain, request_tracker):
    """
    Fetches the liquidity pool with the highest volume for the given token.

    Parameters:
        token (str): Token address.
        chain (str): Blockchain network.
        request_tracker (dict): Tracker to monitor API request counts.

    Returns:
        str: Pool ID or None if the request fails.
    """
    try:
        url = f"https://api.geckoterminal.com/api/v2/networks/{chain}/tokens/{token}/pools?page=1"
        response = make_request(url, request_tracker)
        if response:
            pool_data = response.json()
            first_id = pool_data['data'][0]['id']
            return first_id.split('_', 1)[1] if '_' in first_id else first_id
        else:
            return None
    except Exception as e:
        print(f"Error fetching liquidity pool: {e}")
        return None

def read_dex(token, chain, timeframe, timestamp_marker, limit, request_tracker):
    """
    Fetches OHLCV price data from Gecko Terminal for a given token, respecting API limits.

    Parameters:
        token (str): Token address.
        chain (str): Blockchain network.
        timeframe (str): Time aggregation (e.g., '/minute?aggregate=1').
        timestamp_marker (str): Alert timestamp in '%Y-%m-%d %H:%M:%S%z' format.
        limit (int): Number of records to fetch per request (up to 1000).
        request_tracker (dict): Tracker to monitor API request counts.

    Returns:
        tuple: (alert_price, alert_price_timestamp, max_price_before, max_price_before_timestamp,
                price_never_higher_before, max_price_after, max_price_after_timestamp, price_never_higher_after, max_x)
    """
    try:
        # Convert timestamp_marker to seconds since epoch
        timestamp_marker_epoch = int(datetime.datetime.strptime(timestamp_marker, '%Y-%m-%d %H:%M:%S%z').timestamp())
        print(f"Timestamp of alert (epoch): {timestamp_marker_epoch}")

        # Fetch the liquidity pool
        pool = fetch_liquidity_pool(token, chain, request_tracker)
        if not pool:
            print(f"Failed to fetch liquidity pool for token: {token}")
            return None, None, None, None, None, None, None, None, None

        # Fetch data before the alert
        pre_alert_data = fetch_dex_data(pool, timeframe, limit, timestamp_marker_epoch, "before", request_tracker, chain)

        # Fetch the first chunk of post-alert data
        post_alert_limit_timestamp = timestamp_marker_epoch + (limit * 60)
        post_alert_data_1 = fetch_dex_data(pool, timeframe, limit, post_alert_limit_timestamp, "before", request_tracker, chain)

        # Determine the timestamp for the second post-alert fetch
        last_timestamp = post_alert_data_1[0][0] if post_alert_data_1 else None
        if last_timestamp:
            post_alert_data_2 = fetch_dex_data(pool, timeframe, limit, last_timestamp, "before", request_tracker, chain)
        else:
            post_alert_data_2 = []

        # Combine all data for deduplication and sorting
        combined_data = pre_alert_data + post_alert_data_1 + post_alert_data_2
        combined_data = deduplicate_and_sort_data(combined_data)

        # Find the closest timestamp to the alert timestamp
        alert_price_data = min(
            combined_data,
            key=lambda row: abs(row[0] - timestamp_marker_epoch),
            default=None
        )
        alert_price = alert_price_data[4] if alert_price_data else None
        alert_price_timestamp = datetime.datetime.fromtimestamp(alert_price_data[0], tz=datetime.timezone.utc).isoformat() if alert_price_data else None

        # Split combined data into pre- and post-alert segments
        pre_alert_prices = [(row[0], row[4]) for row in combined_data if row[0] < timestamp_marker_epoch]
        post_alert_prices = [(row[0], row[4]) for row in combined_data if row[0] >= timestamp_marker_epoch]

        # Calculate maximum prices and their timestamps
        if pre_alert_prices:
            max_price_before_timestamp, max_price_before = max(pre_alert_prices, key=lambda x: x[1])
            max_price_before_timestamp = datetime.datetime.fromtimestamp(max_price_before_timestamp, tz=datetime.timezone.utc).isoformat()
        else:
            max_price_before = alert_price
            max_price_before_timestamp = alert_price_timestamp

        # Initialize variables for max_price_after
        max_price_after = alert_price
        max_price_after_timestamp = alert_price_timestamp
        price_never_higher_after = 1
        max_x = 1

        if post_alert_prices:
            max_price_after_timestamp, max_price_after = max(post_alert_prices, key=lambda x: x[1])
            max_price_after_timestamp_dt = datetime.datetime.fromtimestamp(max_price_after_timestamp, tz=datetime.timezone.utc)
            max_price_after_timestamp = max_price_after_timestamp_dt.isoformat()

            # Apply filters for percentage increase and time
            if max_price_after >= alert_price * 1.1 and (max_price_after_timestamp_dt - datetime.datetime.fromtimestamp(timestamp_marker_epoch, tz=datetime.timezone.utc)).total_seconds() > 120:
                price_never_higher_after = 0
                max_x = max_price_after / alert_price
            else:
                max_price_after = alert_price
                max_price_after_timestamp = alert_price_timestamp
                price_never_higher_after = 1
                max_x = 1

        # Determine flags
        price_never_higher_before = int(max_price_before == alert_price)

        print(f"Alert price: {alert_price}, Max before: {max_price_before}, Max after: {max_price_after}, Max x's: {max_x}")
        return (alert_price, alert_price_timestamp, max_price_before, max_price_before_timestamp,
                price_never_higher_before, max_price_after, max_price_after_timestamp, price_never_higher_after, max_x)
    except Exception as e:
        print(f"Error in read_dex: {e}")
        return None, None, None, None, None, None, None, None, None




def fetch_dex_data(pool, timeframe, limit, timestamp, fetch_type, request_tracker, chain):
    """
    Fetches OHLCV data for the specified timeframe and pool.

    Parameters:
        pool (str): Pool ID.
        timeframe (str): Time aggregation (e.g., '/minute?aggregate=1').
        limit (int): Number of records to fetch.
        timestamp (int): Epoch timestamp to fetch data relative to.
        fetch_type (str): Either "before" for all requests.
        request_tracker (dict): Tracker to monitor API request counts.

    Returns:
        list: List of OHLCV data entries.
    """
    try:
        # Build the query using only before_timestamp
        if fetch_type == "before":
            url = (f"https://api.geckoterminal.com/api/v2/networks/{chain}/pools/{pool}/ohlcv{timeframe}"
                   f"&limit={limit}&before_timestamp={timestamp}")
        else:
            raise ValueError("fetch_type must be 'before'.")

        # Make the API request
        response = make_request(url, request_tracker)
        if response:
            return response.json().get('data', {}).get('attributes', {}).get('ohlcv_list', [])
        else:
            return []
    except Exception as e:
        print(f"Error fetching {fetch_type} data: {e}")
        return []


def deduplicate_and_sort_data(data):
    """
    Deduplicates and sorts OHLCV data by timestamp.

    Parameters:
        data (list): Combined list of OHLCV entries.

    Returns:
        list: Deduplicated and sorted data.
    """
    seen_timestamps = set()
    unique_data = []
    for entry in data:
        if entry[0] not in seen_timestamps:
            unique_data.append(entry)
            seen_timestamps.add(entry[0])
    return sorted(unique_data, key=lambda x: x[0])


def save_combined_data(data, token, time_label):
    """
    Saves combined OHLCV data to a file.

    Parameters:
        data (list): Combined and sorted OHLCV data.
        token (str): Token address.
        time_label (str): Time aggregation label (e.g., '1m').

    Returns:
        str: File path of the saved data.
    """
    try:
        folder = "saved_data"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f'dex_data_{token}_{time_label}.txt')
        with open(filename, 'w') as file:
            for entry in data:
                timestamp = datetime.datetime.fromtimestamp(entry[0], tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                open_price, high_price, low_price, close_price, volume = entry[1:6]
                file.write(f"{timestamp} {open_price} {high_price} {low_price} {close_price} {volume}\n")
        return filename
    except Exception as e:
        print(f"Error saving data to file: {e}")
        return None

def load_ohlcv_data(token, timeframe):
    """
    Loads OHLCV data for a given token and timeframe.

    Parameters:
        token (str): Token address.
        timeframe (str): Timeframe label (e.g., '1m', '5m').

    Returns:
        pd.DataFrame: DataFrame containing the OHLCV data, or None if loading fails.
    """
    try:
        file_path = os.path.join('saved_data', f'dex_data_{token}_{timeframe}.txt')
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        data = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 7:
                    timestamp, price = parts[0] + " " + parts[1], float(parts[2 if timeframe == '1m' else 5])
                    data.append([timestamp, price])

        df = pd.DataFrame(data, columns=["timestamp", "price"])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
        df.set_index('timestamp', inplace=True)
        return df.sort_index()
    except Exception as e:
        print(f"Error loading OHLCV data for {token} ({timeframe}): {e}")
        return None


# Main function to run Telegram scraping and extraction
async def main():
    """
    Main execution loop for scraping Telegram messages, analyzing alerts, and generating the PDF report.
    """
    try:
        # Initialize request tracker
        request_tracker = {'count': 0}

        # Configuration
        group_id = -1002131165158  # Replace with the actual Telegram group ID
        message_limit = 1000000
        chain = 'solana'

        # Fetch and save messages
        messages_file = 'tweet-messages.txt'
        messages = await fetch_telegram_messages(group_id, message_limit)
        save_messages_to_file(messages, messages_file)
        
        # Extract and save structured data
        extracted_file = 'extracted_info.csv'
        entries = extract_info('tweet-messages.txt')
        save_to_csv(extracted_file, entries)
        cleaned_file = 'extracted_info_cleaned.csv'
        # Load tokens and alert times
        tokens, alert_times = load_tokens_and_alert_times(extracted_file, cleaned_file, request_tracker)

        print("Execution completed successfully.")
    except Exception as e:
        print(f"Error in main execution loop: {e}")



async def fetch_telegram_messages(group_id, limit):
    """
    Fetches the last 'limit' messages from a specified Telegram group.

    Parameters:
        group_id (int): Telegram group ID.
        limit (int): Number of messages to fetch.

    Returns:
        list: List of formatted messages.
    """
    try:
        # Fetch the last 'limit' messages using get_messages
        messages = await client.get_messages(group_id, limit=limit)
        
        # Format the messages for further processing
        formatted_messages = [
            f"{message.date}: {message.sender_id}: {message.text or ''}" for message in messages
        ]
        return formatted_messages
    except Exception as e:
        print(f"Error fetching messages with get_messages: {e}")
        return []


def save_messages_to_file(messages, file_path):
    """
    Saves fetched Telegram messages to a text file.

    Parameters:
        messages (list): List of messages.
        file_path (str): Path to save the messages.
    """
    try:
        with open(file_path, 'w', encoding='utf-8', errors='replace') as file:
            file.write("\n".join(messages))
        print(f"Messages saved to {file_path}")
    except Exception as e:
        print(f"Error saving messages to file: {e}")

def load_tokens_and_alert_times(file_path, output_file_path, request_tracker):
    """
    Loads tokens and their respective alert times from a CSV file,
    fetches price data, and writes enriched rows with additional analysis to a new CSV file.

    Parameters:
        file_path (str): Path to the input CSV file.
        output_file_path (str): Path to the output CSV file.
        request_tracker (dict): Tracker to monitor API request counts.

    Returns:
        tuple: A list of tokens and a dictionary of alert times.
    """
    try:
        tokens = []
        alert_times = {}

        with open(file_path, 'r', encoding='utf-8', errors='replace') as csv_file, \
             open(output_file_path, 'w', encoding='utf-8', newline='') as output_file:

            csv_reader = csv.DictReader(csv_file)
            fieldnames = csv_reader.fieldnames  # Original headers

            # Add new headers for analysis results and timestamps
            additional_headers = [
                "alert_price", "alert_price_timestamp",
                "max_price_before", "max_price_before_timestamp", "price_never_higher_before",
                "max_price_after", "max_price_after_timestamp", "price_never_higher_after", "max_x"
            ]
            fieldnames = fieldnames + additional_headers

            csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            csv_writer.writeheader()  # Write headers to the output file

            for row in csv_reader:
                token = row.get('Contract')
                time_of_ping = row.get('Timestamp')
                chain = 'solana'
                timeframe = '/minute?aggregate=1'
                FDV = row.get('FDV')
                limit = 1000

                if token and token != 'n/a' and FDV !='n/a':  # Only process rows with a valid contract
                    tokens.append(token)
                    alert_times[token] = time_of_ping

                    # Call read_dex and retrieve analysis data
                    result = read_dex(token, chain, timeframe, alert_times[token], limit, request_tracker)

                    # Skip rows where liquidity pool fetch fails
                    if result == (None, None, None, None, None, None, None, None, None):
                        print(f"Skipping row for token: {token}, no liquidity pool found.")
                        continue

                    # Unpack the result including timestamps
                    (alert_price, alert_price_timestamp,
                     max_price_before, max_price_before_timestamp,
                     price_never_higher_before,
                     max_price_after, max_price_after_timestamp,
                     price_never_higher_after, max_x) = result

                    # Add new data to the row
                    row["alert_price"] = alert_price
                    row["alert_price_timestamp"] = alert_price_timestamp
                    row["max_price_before"] = max_price_before
                    row["max_price_before_timestamp"] = max_price_before_timestamp
                    row["price_never_higher_before"] = price_never_higher_before
                    row["max_price_after"] = max_price_after
                    row["max_price_after_timestamp"] = max_price_after_timestamp
                    row["price_never_higher_after"] = price_never_higher_after
                    row["max_x"] = max_x

                    csv_writer.writerow(row)  # Write the enriched row to the new CSV file

        return tokens, alert_times
    except Exception as e:
        print(f"Error loading tokens and alert times: {e}")
        return [], {}


with client:
    client.loop.run_until_complete(main())
