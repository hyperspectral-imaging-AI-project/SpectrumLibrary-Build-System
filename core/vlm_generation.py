import os
from openai import OpenAI
from dotenv import load_dotenv
import base64

load_dotenv()
client = OpenAI()

def load_prompt(prompt_path:str):
    # public_prompt_path = os.path.join(os.getcwd(),'./public/prompts/',f'{prompt_path}.md')
    public_prompt_path = prompt_path
    
    if not os.path.exists(public_prompt_path):
        raise FileNotFoundError(f"🚨 {public_prompt_path} 파일을 찾을 수 없음!")

    with open(public_prompt_path, "r", encoding="utf-8") as file:
        return file.read().strip()

def _build_messages(prompt: str, image_url: str | None):
    if image_url:
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }]
    else:
        return [{"role": "user", "content": prompt}]

def openai_inference(prompt, image_path=None, streaming=False):
    """
    streaming=True  → return generator[str]
    streaming=False → return str
    """
    messages = _build_messages(prompt, image_path)

    if streaming:
        # 바깥 함수에는 yield가 없도록! 내부 제너레이터로 분리
        def _gen():
            stream = client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                text = None
                try:
                    text = chunk.choices[0].delta.content
                except Exception:
                    text = None
                if text:
                    yield text
        return _gen()  # 제너레이터 '객체' 반환

    # ← 여기서는 일반 함수처럼 최종 문자열 반환
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=messages,
        stream=False
    )
    return (resp.choices[0].message.content or "")


# prompt = '안녕'
# result = openai_inference(prompt)
# print(result)