import requests

API_KEY = "ca0b98a1cd522a876b3cac4213af38ce"

def get_weather(city):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )
    data = requests.get(url).json()

    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]

    return f"Temperature: {temp}°C | Humidity: {humidity}%"
