""" Example handler file. """

import runpod
import init_models
import glue
# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
models = init_models.load_models()



def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    file = job_input.get('file')
    
    if file is None:
        return {"error": "No file provided."}
    
    # Save the file to the disk
    file_path = "/tmp/" + file.filename
    with open(file_path, "wb") as f:
        f.write(file.read())
    
    # Run the inference
    llama_v2v_ans = glue.doStuff(file_path)

    return {"output": llama_v2v_ans}


runpod.serverless.start({"handler": handler})