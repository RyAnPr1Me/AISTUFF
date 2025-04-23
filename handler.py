import json

def handler(event, context):
    """
    Example handler function for inference.
    Expects 'text' in the event input.
    """
    # ...load model/tokenizer if needed...
    input_text = event.get('text', '')
    # ...perform inference or processing...
    result = {
        "input": input_text,
        "output": f"Echo: {input_text}"
    }
    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }
