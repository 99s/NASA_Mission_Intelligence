from typing import Dict, List
from openai import OpenAI


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> str:

    system_prompt = """
You are a NASA mission intelligence assistant.

Use the provided context to answer questions accurately.
If the answer is not present in the context, clearly say so.
Keep answers concise, factual, and mission-focused.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    if context:
        messages.append({
            "role": "system",
            "content": f"Context:\n{context}"
        })

    for message in conversation_history:
        messages.append(message)

    messages.append({
        "role": "user",
        "content": user_message
    })

    client = OpenAI(api_key=openai_key)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content