import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")

client = Client(account_sid, auth_token)

try:
    account = client.api.accounts(account_sid).fetch()

    print("✅ Twilio credentials are working!")
    print("Account SID:", account.sid)
    print("Account Status:", account.status)
    print("Account Name:", account.friendly_name)

except Exception as e:
    print("❌ Authentication failed")
    print(e)