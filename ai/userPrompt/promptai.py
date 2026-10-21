import os
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

api_key = os.getenv('OPENROUTER_API_KEY')

client = OpenAI( 
    base_url= 'https://openrouter.ai/api/v1/',
    api_key = api_key
    )
system_prompt = """
Ты — ИИ-парсер параметров для подбора компьютера. 
Проанализируй текст пользователя и извлеки из него требования.
Верни ответ СТРОГО в формате JSON со следующими ключами:
- "budget": число (целое число в рублях, если указано, иначе null)
- "tasks": массив строк (строго ["Games", "Montage", "Office", "3D"])
- "preferences": строка (особые пожелания, если есть тоже на английском, по примеру tasks, тип охлаждения [Water/Air], цвет корпуса, бренд, и т.д., если нет — null)
"""

user_query = input()

response = client.chat.completions.create(
    model= "openrouter/free",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ],
    response_format={
        "type": "json_object"
    }, 
)


raw_answer = response.choices[0].message.content
print("Ответ от модели:", raw_answer)

data_dict = json.loads(raw_answer)

with open("ai/userPrompt/AIRequest.json", "w", encoding="utf-8") as f:
    json.dump(data_dict, f, ensure_ascii=False, indent=4)
