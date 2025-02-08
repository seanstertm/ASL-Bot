import os
from transformers import pipeline

def main():
    model_name = "google/flan-t5-large"

    print(f"Loading model: {model_name} (this may take a moment)...")
    generator = pipeline(
        "text2text-generation",
        model=model_name,
        device=-1,
    )

    input_file = "recognized_letters.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Make sure your ASL script writes to it.")
        return
    
    with open(input_file, "r", encoding="utf-8") as f:
        spelled_text = f.read().strip()
    
    if not spelled_text:
        print("No text found in recognized_letters.txt. Exiting.")
        return
    
    print(f"Spelled text: '{spelled_text}'")

    prompt = (
        f"""
        Below are examples of how to respond in ALL CAPS. 
        Respond conversationally.
        If they ask you a question, answer it.
        If they say hello, say hello back.
        If they talk about a topic, give your thoughts on the topic.

        Examples:
        The user said: HELLO
        response: HI HOW ARE YOU

        The user said: IAMJOE
        response: HI JOE

        The user said: WHATISTHEWEATHER
        response: IT IS SUNNY

        The user said: DOYOULIKEPIZZA
        response: YES I LIKE PEPPERONI PIZZA

        Do the same for the user's input. Do not repeat the user's input as your response.
        The user said: {spelled_text}
        """
    )

    response = generator(
        prompt,
        max_new_tokens=50,
        num_beams=4,
        do_sample=True,
        temperature=1.5,
        repetition_penalty=2.0
    )

    llm_output = response[0]["generated_text"]

    print("\nLLM Response:")
    print(llm_output)
    print("\nDone.")

if __name__ == "__main__":
    main()