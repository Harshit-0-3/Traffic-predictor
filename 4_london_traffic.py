import requests

# 🔴 PASTE YOUR KEY INSIDE THE QUOTES BELOW
API_KEY = "db09fb9ee80c4c8aa06851bd67f6086d" 

# URL to get all traffic cameras ("Jam Cams")
url = "https://api.tfl.gov.uk/Place/Type/JamCam"
params = {'app_key': API_KEY}

print("🌍 Connecting to London Traffic Network...")
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    print(f"✅ SUCCESS! Found {len(data)} active cameras.")
    
    # Let's grab the first camera that actually has a video feed
    for camera in data:
        # Check if it has an image property
        image_url = None
        for prop in camera['additionalProperties']:
            if prop['key'] == 'imageUrl':
                image_url = prop['value']
                break
        
        # If we found an image, print it and stop
        if image_url:
            print(f"📷 Camera Location: {camera['commonName']}")
            print(f"🔗 LIVE FEED URL: {image_url}")
            print("👉 Copy the URL above and paste it into your browser to see the live traffic!")
            break
else:
    print(f"❌ Connection Failed. Error Code: {response.status_code}")