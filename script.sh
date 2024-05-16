import gdown

# Google Drive URL or file ID
url = 'https://drive.google.com/file/d/1kfjvgIwJQn5p3MoUpyHEeyBfP7Lm5ePy/view?usp=sharing'

# Output path for the downloaded zip file
output = '/root/ODGNet/data.zip'

# Download the zip file
gdown.download(url, output, quiet=False)