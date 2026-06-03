import requests

# Set the API endpoint and API key
api_url = "https://api.example.com/v1/endpoint"  # Replace with the actual API URL
api_key = "sk-ant-oat01-UYNFAd01F6n3BH09rXfzLYRWMQ74P4Mo"  # Your API key

# Set the headers for the request, including the Authorization header
headers = {
    "Authorization": f"Bearer {api_key}",
}

# Send the GET request
response = requests.get(api_url, headers=headers)

# Check if the response is successful
if response.status_code == 200:
    print("Success!")
    print("Response:", response.json())  # If the response is JSON, print it
else:
    print(f"Error: {response.status_code}")
    print("Message:", response.text)  # Print error message if any