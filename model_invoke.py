from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "TheBloke/Yi-34B-AWQ"

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    revision="f1b2cd1b7459ceecfdc1fac5bb8725f13707c589",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cuda:0",
    trust_remote_code=True
)

def invoke(input_text):
    prompt = input_text

    tokens = tokenizer(
        prompt,
        return_tensors='pt'
    ).input_ids.cuda()

    generation_params = {
        # "do_sample": True,
        # "temperature": 0.7,
        # "top_p": 0.95,
        # "top_k": 40,
        "max_new_tokens": 512,
        "repetition_penalty": 1.1
    }

    generation_output = model.generate(
        tokens,
        **generation_params
    )

    text_output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    return text_output