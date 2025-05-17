def download_figshare_file(code, filename,private_link = '',force_download = False):
    """
    Download a file from a private Figshare link if not present locally.
    """
    
    if len(private_link)>0:
        link = f'https://figshare.com/ndownloader/files/{code}?private_link={private_link}'
    else:
        link = f'https://figshare.com/ndownloader/files/{code}
    if (not os.path.exists(filename)) or force_download:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(link, filename)
    else:
        print(f"{filename} already exists. Skipping download.")
