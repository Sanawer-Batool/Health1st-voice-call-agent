import sqlite3


# Step 1: Create database and tables
connection = sqlite3.connect('db/clinic.db')
cursor = connection.cursor()

# Perform a JOIN query
query = '''
SELECT providers.name AS provider_name, appointment_types.name AS appointment_type_name
FROM providers
JOIN provider_appointment_types ON providers.id = provider_appointment_types.provider_id
JOIN appointment_types ON provider_appointment_types.appointment_type_id = appointment_types.id;
'''

cursor.execute(query)
results = cursor.fetchall()

print(results)

connection.close()