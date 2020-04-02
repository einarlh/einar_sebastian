import zipfile as zf
files = zf.ZipFile("assignment4 - Copy.zip", 'r')
files.extractall('directory to extract')
files.close()