import runpod
import os

runpod.api_key = os.getenv("RUNPOD_API_KEY")

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Specify the file path of the .wav file
file_path = os.getcwd() + "sound/maklowicz.wav"

# Open the file in binary mode
with open(file_path, "rb") as file:
    # Send the POST request to the REST API
    files={"file": file}

try:
    run_request = endpoint.run_sync(
        {
            "input": {"file": file}
        },
        timeout=600,  # Timeout in seconds.
    )

    print(run_request)
except TimeoutError:
    print("Job timed out.")