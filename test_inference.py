from huggingface_hub import InferenceClient

def chatbot(prompt,hf_token):

    client=InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=hf_token
    )
    
    messages=[
        {"role": "system", "content": "You are a helpful and knowledgeable and intelligent assistant."},
        {"role": "user","content":prompt }
    ]
    response=client.chat_completion(messages=messages,max_tokens=200,temperature=0.2)

    return response.choices[0].message.content
