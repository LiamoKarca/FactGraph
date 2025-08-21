import os
import glob
from pathlib import Path
import openai
from dotenv import load_dotenv

# 初始化環境變數
load_dotenv()
OPENAI_API_KEY = os.getenv("GPT_API")
OPENAI_MODEL = os.getenv("GPT_MODEL", "gpt-4o")  # 預設 gpt-4o，必要時自訂

openai.api_key = OPENAI_API_KEY

PROMPT_PATH = Path("src/qa/fast_mode/prompts/keywords_extraction.txt")
USER_INPUT_DIR = Path("data/interim/fast_mode/user-input")
OUTPUT_DIR = Path("data/interim/fast_mode/keywords")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_prompt():
    with open(PROMPT_PATH, "r", encoding="utf-8-sig") as f:
        return f.read()


def extract_keywords(text, prompt_template):
    prompt = prompt_template.replace("{input}", text)
    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "你是一個關鍵詞與三元組提取專家，根據指示只產出結構化結果，不做解釋。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] LLM 提示失敗：{e}"


def main():
    prompt_template = load_prompt()
    txt_files = glob.glob(str(USER_INPUT_DIR / "*.txt"))
    if not txt_files:
        print("未偵測到輸入檔案。")
        return

    for file_path in txt_files:
        file_name = Path(file_path).stem
        with open(file_path, "r", encoding="utf-8-sig") as f:
            text = f.read()

        result = extract_keywords(text, prompt_template)

        output_file = OUTPUT_DIR / f"{file_name}_keywords.json"
        with open(output_file, "w", encoding="utf-8-sig") as f:
            f.write(result)
        print(f"[完成] {file_name} -> {output_file}")


if __name__ == "__main__":
    main()
