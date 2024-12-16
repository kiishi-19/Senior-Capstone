import logging
from src.preprocessing import extract_syscalls

# Configure logging for the test
logging.basicConfig(filename='output/test_preprocessing.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')

def test_extract_syscalls(scap_file):
    try:
        logging.info(f"Testing extract_syscalls with file: {scap_file}")
        result = extract_syscalls(scap_file)
        logging.info(f"Result: {result}")
    except Exception as e:
        logging.error(f"Error during extract_syscalls: {e}")

if __name__ == "__main__":
    # Replace with the path to a sample .scap file for testing
    test_scap_file = '/home/ubuntu/GHIDS2/Senior-Capstone/Data/CB-DS/NORMAL/112.scap'
    test_extract_syscalls(test_scap_file)