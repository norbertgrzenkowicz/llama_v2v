import record
import runpod
import os
runpod.api_key = os.getenv("RUNPOD_API_KEY")

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

file_path = os.getcwd() + record.record()
# file_path = os.getcwd() + "sound/maklowicz.wav" #Just in case you dont want to record

with open(file_path, "rb") as file:
    files={"file": file}

try:
    run_request = endpoint.run_sync(
        {
            "input": {"file": file}
        },
        timeout=600,
    )

    print(run_request)
except TimeoutError:
    print("Job timed out.")