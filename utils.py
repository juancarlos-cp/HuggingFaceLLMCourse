import requests
import gzip
import shutil
from pathlib import Path

def download_and_decompress(url, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()  # raises an error on failure

    with gzip.open(response.raw, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


'''
download_and_decompress(
    "https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz",
    "./data/SQuAD_it-train.json"
)
download_and_decompress(
"https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz",
    "./data/SQuAD_it-test.json"
)
'''

