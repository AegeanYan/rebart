"""
Adapted from https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
"""
import re
import json
import tqdm
import torch
import logging
import argparse

logger = logging.getLogger("__name__")

handler1 = logging.StreamHandler()
handler2 = logging.FileHandler(filename="abs.log")

logger.setLevel(logging.DEBUG)
handler1.setLevel(logging.INFO)
handler2.setLevel(logging.WARNING)

formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
handler1.setFormatter(formatter)
handler2.setFormatter(formatter)

logger.addHandler(handler1)
logger.addHandler(handler2)



from common import init_model, load_data


def main() -> None:
    """
    Generate outputs
    """
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument(
        "--in_file",
        default=None,
        type=str,
        required=True,
        help="The input json file",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        required=True,
        help="out jsonl file",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2",
        type=str,
        help="LM checkpoint for initialization.",
    )

    # Optional
    parser.add_argument(
        "--max_length", default=40, type=int, required=False, help="Maximum text length"
    )
    parser.add_argument(
        "--k", default=0, type=int, required=False, help="k for top k sampling"
    )
    parser.add_argument(
        "--p", default=0, type=float, required=False, help="p for nucleus sampling"
    )
    parser.add_argument(
        "--beams", default=0, type=int, required=False, help="beams for beam search"
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        required=False,
        help="temperature for sampling",
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--task",
        default="",
        type=str,
        help="what is the task?"
    )
    args = parser.parse_args()
    logger.debug(args)

    if (
        (args.k == args.p == args.beams == 0)
        or (args.k != 0 and args.p != 0)
        or (args.beams != 0 and args.p != 0)
        or (args.beams != 0 and args.k != 0)
    ):
        raise ValueError(
            "Exactly one of p, k, and beams should be set to a non-zero value."
        )

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    logger.debug(f"Initializing {args.device}")

    tokenizer, model = init_model(args.model_name_or_path, device)

    examples = load_data(args.in_file, args.task)

    # logger.info(examples[:5])

    special_tokens = ["[shuffled]", "[orig]", "<eos>"]
    extra_specials = [f"<S{i}>" for i in range(args.max_length)]
    special_tokens += extra_specials


    with open(args.out_file, "w") as f_out:
        example_len = len(examples)
        batch_size_ = 128
        for i in tqdm.tqdm(range(int((example_len - 1)/batch_size_) + 1)):
            try:
                if i == int((example_len - 1)/batch_size_):
                    preds = generate_conditional(
                        tokenizer,
                        model,
                        args,
                        examples[i*batch_size_:],
                        device,
                    )
                else:
                    preds = generate_conditional(
                        tokenizer,
                        model,
                        args,
                        examples[i*batch_size_:(i+1)*batch_size_],
                        device,
                    )
                
                # Remove any word that has "]" or "[" in it
                preds = [re.sub(r"(\w*\])", "", pred) for pred in preds]
                preds = [re.sub(r"(\[\w*)", "", pred) for pred in preds] 
                preds = [re.sub(" +", " ", pred).strip() for pred in preds] 
            except Exception as exp:
                logger.info(exp)
                preds = []
            
            if i == int((example_len-1)/batch_size_):
                
                for j in range(i*batch_size_, example_len):
                    input,output = examples[j]
                    f_out.write(
                        json.dumps({"input": input, "gold": output, "predictions": [preds[j-i*batch_size_]]})
                        + "\n"
                    )
            else:
                for j in range(i*batch_size_, (i+1)*batch_size_):
                    input,output = examples[j]
                    f_out.write(
                        json.dumps({"input": input, "gold": output, "predictions": [preds[j-i*batch_size_]]})
                        + "\n"
                    )

    # preds = generate_conditional(
    # tokenizer,
    # model,
    # args,
    # [examples[i][0] for i in range(5)],
    # device,
    # )

def generate_conditional(tokenizer, model, args, input, device):
    """
    Generate a sequence with models like Bart and T5
    """
    input = [input[i][0] for i in range(len(input))]

    not_pad_encoded = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input[0]))

    encoded = tokenizer(input, return_tensors='pt', padding=True)
    input_ids = encoded['input_ids'].to(device)
    decoder_start_token_id = not_pad_encoded[-1]
    # logger.warning(decoder_start_token_id)
    
    max_length = args.max_length


    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=encoded['attention_mask'].to(device),
        do_sample=args.beams == 0,
        max_length=max_length,
        min_length=5,
        temperature=args.temperature,
        top_p=args.p if args.p > 0 else None,
        top_k=args.k if args.k > 0 else None,
        num_beams=args.beams if args.beams > 0 else None,
        early_stopping=True,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=decoder_start_token_id,
        num_return_sequences=1 #max(1, args.beams)
    )

    preds = [tokenizer.decode(
        output, skip_special_tokens=False, clean_up_tokenization_spaces=False) for output in outputs]

    return preds



if __name__ == "__main__":
    main()