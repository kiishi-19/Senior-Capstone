import re
import subprocess
import pandas as pd
from pathlib import Path
import logging
import shutil

# Configure logging
logging.basicConfig(filename='preprocessing.log', level=logging.INFO)

# Global sets to maintain known syscalls and arguments
KNOWN_SYSCALLS = set()
KNOWN_ARGUMENTS = set()

def update_known_syscalls(syscalls):
    """Update the set of known syscalls."""
    global KNOWN_SYSCALLS
    KNOWN_SYSCALLS.update(syscalls)

def update_known_arguments(args):
    """Update the set of known normalized arguments."""
    global KNOWN_ARGUMENTS
    KNOWN_ARGUMENTS.update(args)

def tokenize_arguments(arg_str):
    """Convert argument string into list of key-value pairs or tokens."""
    if not arg_str or arg_str == "<NO_ARGS>":
        return []
        
    tokens = []
    for arg in arg_str.split():
        if '=' in arg:
            key, value = arg.split('=', 1)
            tokens.append(f"{key}={value}")
        else:
            tokens.append(arg)
    return tokens

def normalize_arguments(args_str):
    """Normalize syscall arguments by removing variable data and update known arguments."""
    if not args_str or args_str.isspace():
        return []
        
    # Replace IP addresses
    args_str = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '<IP>', args_str)
    
    # Replace specific paths with <PATH>
    args_str = re.sub(r'/[\w\./-]+', '<PATH>', args_str)
    
    # Replace timestamps in logs
    args_str = re.sub(r'\[\d{2}/\w+/\d{4}:\d{2}:\d{2}:\d{2}\s[+-]\d{4}\]', '<TIMESTAMP>', args_str)
    
    # Replace numbers with <NUM>
    args_str = re.sub(r'\b\d+\b', '<NUM>', args_str)
    
    # Replace hex values with <HEX>
    args_str = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', args_str)
    
    # Replace multiple spaces with single space
    args_str = ' '.join(args_str.split())
    
    # Tokenize arguments
    arg_tokens = tokenize_arguments(args_str)
    
    # Update known arguments set
    update_known_arguments(arg_tokens)
    
    return arg_tokens

def extract_syscalls(scap_file, window_duration='1s'):
    """
    Extract syscalls from a .scap file and group them into time windows.
    
    Args:
        scap_file (str or Path): Path to the .scap file
        window_duration (str): Duration of time windows (e.g., '1s', '2s')
        
    Returns:
        pd.DataFrame: DataFrame with columns [time_window, timestamp, syscall, arguments]
    """
    # Ensure sysdig is installed
    if not shutil.which("sysdig"):
        raise EnvironmentError("Sysdig is not installed or not found in PATH.")
    
    cmd = [
        'sysdig',
        '-r', str(scap_file),
        '-p', '%evt.datetime,%evt.type,%evt.args'
    ]
    
    try:
        logging.info(f"Processing file: {scap_file}")
        output = subprocess.check_output(cmd).decode('utf-8', errors='replace')
        lines = output.strip().split('\n')
        data = [line.split(',', 2) for line in lines if line.count(',') >= 2]
        df = pd.DataFrame(data, columns=['timestamp', 'syscall', 'arguments'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Normalize syscall arguments and tokenize
        df['arguments'] = df['arguments'].apply(normalize_arguments)
        
        # Update known syscalls set
        update_known_syscalls(df['syscall'].unique())
        
        # Filter invalid rows
        if df.empty:
            logging.warning(f"No data extracted from {scap_file}")
            return pd.DataFrame(columns=['time_window', 'timestamp', 'syscall', 'arguments'])

        df = df.dropna(subset=['syscall', 'timestamp'])
        
        # Create time windows
        df['time_window'] = df['timestamp'].dt.floor(window_duration)
        
        # Group by time window and aggregate
        grouped_df = df.groupby('time_window').agg({
            'timestamp': list,  # Preserve timestamps for inter-arrival times
            'syscall': list,
            'arguments': list
        }).reset_index()
        
        # Add flow-level metadata
        grouped_df['num_syscalls'] = grouped_df['syscall'].apply(len)
        grouped_df['num_unique_arguments'] = grouped_df['arguments'].apply(lambda x: len(set(x)))
        
        return grouped_df

    except subprocess.CalledProcessError as e:
        logging.error(f"Sysdig error on file {scap_file}: {e.output}")
        return pd.DataFrame(columns=['time_window', 'timestamp', 'syscall', 'arguments'])
