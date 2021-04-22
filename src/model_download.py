import os

from google_drive_downloader import GoogleDriveDownloader as gdd

print("Downloading pre-trained models...")
DEFAULT_PATH = "epiai_doc/models/model_20210315_150752/model.bin"
os.makedirs("static/models", exist_ok=True)
gdd.download_file_from_google_drive(
    file_id="1CRHtXiTK1SIbSFeRuEgdp7T1fEUpAcS8", dest_path=DEFAULT_PATH
)
