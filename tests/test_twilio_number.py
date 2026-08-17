import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

client = Client(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN")
)

phone_number = os.getenv("TWILIO_PHONE_NUMBER")

try:
    numbers = client.incoming_phone_numbers.list(phone_number=phone_number)

    if numbers:
        number = numbers[0]

        print("Phone number found!")
        print("Number:", number.phone_number)
        print("Friendly name:", number.friendly_name)
        print("Voice URL:", number.voice_url)
        print("Voice method:", number.voice_method)

    else:
        print("Phone number was NOT found in this Twilio account.")

except Exception as e:
    print("Error:")
    print(e)