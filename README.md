---

# Uploaded  model

- **Developed by:** Ellight
- **License:** apache-2.0
- **Finetuned from model :** unsloth/llama-3.2-1b

This llama model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)


# For Inference
The Llama model is trained using Bitext-customer-support-llm-chatbot-training-dataset for generating more accurate responses to customer queries. The model can be further finetuned based on specific companies data.

```
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Ellight/Llama-3.2-1B-Customer-support-Chatbot", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}
    
    ### Response:
    {}"""
    
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "I want assistance to cancel order", # instruction
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")
    
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 512)

    Model Response:
    <|begin_of_text|>Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    I want assistance to cancel order
    
    ### Response:
    I understand that you need assistance in canceling your order. I'm here to guide you through the process. To cancel your order, you can follow these steps:
    
    1. Log in to your account on our website.
    2. Navigate to the "Orders" or "My Account" section.
    3. Locate your order and click on it.
    4. On the order details page, locate the "Cancel Order" or "Cancel Order" option.
    5. Click on it to initiate the cancellation process.
    
    If you encounter any difficulties or have any questions during the cancellation process, please don't hesitate to reach out to us. We're here to assist you every step of the way.<|end_of_text|>
```
