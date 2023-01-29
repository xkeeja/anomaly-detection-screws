import gdown
import zipfile
import os

def data_download():
    url = "https://drive.google.com/u/0/uc?id=1O0yQ-VyqSxDKXFVW96Gj15mDHA4dU4SA&export=download"
    output = "screws.zip"
    gdown.download(url, output)

    print('Unzipping file...')
    with zipfile.ZipFile('screws.zip', 'r') as zip_ref:
        zip_ref.extractall('data')
    print('Data successfully unzipped to data/archive.')

    print('Deleting zip file...')
    os.remove('screws.zip')
    print('Deleted zip file.')
