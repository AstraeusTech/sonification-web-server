# Sonification Web Server

This is a web server that is used by the client to create 3D models from 2D images and to sonify them.

## Installation

### Requirements

- Python 3.8 or higher

### Steps

1. Clone the repository

```bash
git clone https://https://github.com/AstraeusTech/sonification-web-server
```

2. Install the dependencies

```bash
pip install -r requirements.txt
```

3. Run the server

```bash
python main.py
```

#### Docker

```bash
docker build --platform linux/amd64 -t sonification:latest . 
docker run -p 8080:8080 sonification:latest
```

## Environment Variables

```txt
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY="/cAV6Q"
S3_BUCKET=""
PORT=
```

## Usage

### API

#### `GET /[id]`

Given an ID, retrieves the image file from S3, converts it to a 3D model, creates a sonification and returns a success message.
