# import module
import openai

# set up OpenAI API Key

openai.api_key = "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"

# Set up ChatGPT Function

def chating_bot(question, lang_code):

    # Set up the GPT-3 model
    model_engine = "text-davinci-002"
    model_prompt_cn = (
        "Q: 你是誰？\n"
        "A: 我叫作「陸奧娜貓」.\n"
        "Q: 你是做什麼的？\n"
        "A: 我是一位專業護士，可以回答你的問題並和你聊天。\n"
        "Q: 誰創造了你?\n"
        "A: 我是由Asa Robotics Limited開發的。\n"
        "Q: 請你用粵語口語嚟答我問題\n"
        "A: 冇問題,我可以用粵語口語回答你嘅問題。\n"
    )

    model_prompt_en = (
        "Q: Who are you?\n"
        "A: My name is Luna Cat.\n"
        "Q: What are you?\n"
        "A: I am a professional nurse, that can answer your question and chat with you.\n"
        "Q: Who created you\n"
        "A: I am developed by Asa Robotics Limited.\n"
    ) 

    if lang_code == "en-us": 

        model_prompt = model_prompt_en

    if lang_code == "yue-hant-hk":

        model_prompt = model_prompt_cn

    # Define function to ask OpenAI API a question

    def ask_openai(prompt, temperature=0.2, max_tokens=1024):
        response = openai.Completion.create(
            engine=model_engine,
            prompt=f"{model_prompt}\n\nQ: {prompt}\nA:",
            temperature=temperature,
            max_tokens=max_tokens,
            stop=None,
        )

        #print(response)

        answer = response.choices[0].text.strip()
        return answer
    
    answer = ask_openai(question)

    return answer, True

if __name__ == "__main__":

    text = "Who won the 2018 Fifa World Cup."

    answer, counter = chating_bot(text)

    print(answer)