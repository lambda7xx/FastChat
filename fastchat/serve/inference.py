"""Inference for FastChat models."""
import abc
import gc
import json
import math
import os
import sys
import time
from typing import Iterable, Optional, Dict
import warnings

import psutil
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from fastchat.conversation import get_conv_template, SeparatorStyle
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig
from fastchat.utils import is_partial_stop, is_sentence_complete, get_context_length


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
):
    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)
    print(f"0***generate_stream,temperature:{temperature} and repetition_penalty:{repetition_penalty} and top_p:{top_p} top_k:{top_k}***")
    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids #输入端prompt对应的token id
    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:  # truncate
        max_src_len = context_len - max_new_tokens - 1
    print(f"***1 generate_stream, model.config.is_encoder_decoder:{model.config.is_encoder_decoder} and context_len:{context_len} and max_new_token:{max_new_tokens} and max_src_len:{max_src_len} and len(input_ids):{len(input_ids)}***")
    input_ids = input_ids[-max_src_len:]
    print(f"***2 generate_stream new input_ids.shape:{len(input_ids)}***")
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)
    print(f"3 generate_stream output_ids:{len(output_ids)} and input_echo_len:{input_echo_len}***")
    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )
        print(f"4 generate_stream encoder_output.shape:{encoder_output.shape} and start_ids.shape:{start_ids.shape}")

    past_key_values = out = None
    sent_interrupt = False
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)#forward計算
                logits = out.logits
            past_key_values = out.past_key_values
            print(f"5 generate_stream type(out):{type(out)} and logits.shape:{logits.shape}  ") #logits.shape:torch.Size([1, 40, 32000]) for vicnua 7b. 40 is the prompt length
        else:  # decoding
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False
                logits = out.logits
            past_key_values = out.past_key_values
            #print(f"6 out.shape:{out.shape} and logits.shape:{logits.shape} and past_key_values.shape:{past_key_values.shape} ")
        print(f"7 generate_stream logits_processor:{logits_processor} and repetition_penalty:{repetition_penalty}")
        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]
        print(f"8 generate_stream last_token_logits.shape:{last_token_logits.shape} and last_token_logits.shape:{last_token_logits.shape} and last_token_logits.dtype:{last_token_logits.dtype}")
            ###logits[0, -1, :] 通常用于提取batch中第一个样本的最后一个时间步的所有特征/类别的logits。
            #如，假设你正在处理一个NLP任务，并且你的logits tensor有形状(batch_size, sequence_length, num_classes)，
            # 那么logits[0, -1, :]将返回第一个样本的最后一个词的所有类别的logits。
            ####

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2) ## 从多项分布中抽取5个样本
            print(f"9 generate_stream last_token_logits.shape:{last_token_logits.shape} and probs.shape:{probs.shape} and indices.shape:{indices.shape}")
            tokens = [int(token) for token in indices.tolist()]
            print(f"10 generate_stream len(tokens):{len(tokens)} and token:{tokens}")
        token = tokens[0]#TODO, 为什么是第一个元素？
        output_ids.append(token)
        
        print(f"10.5 generate_stream, token:{token} and stop_token_ids:{stop_token_ids}")
        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False
        #20230928 注: 当token是stop_token_ids中的元素时，stopped为True，否则为False，即stopped表示是否停止生成，在不断的生成的时候，stopped总有机会设置为true

        # Yield the output tokens
        print(f"11 generate_stream i:{i} and stream_interval:{stream_interval} and max_new_tokens:{max_new_tokens} and stopped:{stopped} and echo:{echo}")
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            #这里的判断条件是，每隔stream_interval个token，或者是最后一个token，或者是stopped为True时，就会执行下面的代码
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            ) #将output id转换为文本
            # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
            print(f"12 generate_stream i:{i} and stop_str:{stop_str} and  judge_sent_end:{judge_sent_end} and stopped:{stopped} and is_sentence_complete(output):{is_sentence_complete(output)}")
            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1] #TODO, 为什么是第二个元素？
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    print(f"13 generate_stream i:{i} and stop_str:{stop_str} and rfind_start:{rfind_start} and output:{output}")
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    print(f"14 generate_stream i:{i} and stop_str:{stop_str} and rfind_start:{rfind_start} and output:{output}")
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            #使用yield关键字的代码，用于从生成器函数中返回一个值，但不终止函数的执行，允许下次从上次停止的地方继续执行。
            if not partially_stopped: 
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""

    @abc.abstractmethod
    def print_output(self, text: str):
        """Print output."""


def chat_loop(
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: str,
    load_8bit: bool,
    cpu_offloading: bool,
    conv_template: Optional[str],
    conv_system_msg: Optional[str],
    temperature: float,
    repetition_penalty: float,
    max_new_tokens: int,
    chatio: ChatIO,
    gptq_config: Optional[GptqConfig] = None,
    awq_config: Optional[AWQConfig] = None,
    revision: str = "main",
    judge_sent_end: bool = True,
    debug: bool = True,
    history: bool = True,
):
    # Model
    model, tokenizer = load_model(
        model_path,
        device=device,
        num_gpus=num_gpus,
        max_gpu_memory=max_gpu_memory,
        load_8bit=load_8bit,
        cpu_offloading=cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        revision=revision,
        debug=debug,
    )
    generate_stream_func = get_generate_stream_function(model, model_path)#得到generate_stream函数

    model_type = str(type(model)).lower()
    print("Model type:", model_type)
    is_t5 = "t5" in model_type
    is_codet5p = "codet5p" in model_type
    print(f"***1 chat_loop, is_t5:{is_t5}, is_codet5p:{is_codet5p} and repetition_penalty:{repetition_penalty}***")
    # Hardcode T5's default repetition penalty to be 1.2
    if is_t5 and repetition_penalty == 1.0:
        repetition_penalty = 1.2

    # Set context length
    #print(f"***chat_loop, model.config:{model.config}***")
    context_len = get_context_length(model.config)
    print(f"***2 chat_loop, context_len:{context_len} and conv_template:{conv_template} and conv_system_msg:{conv_system_msg}***")
    # Chat
    def new_chat():
        if conv_template:
            conv = get_conv_template(conv_template)
        else:
            conv = get_conversation_template(model_path)
        if conv_system_msg is not None:
            conv.set_system_message(conv_system_msg)
        print(f"new_chat, conv:{conv} and conv_template:{conv_template} and conv_system_msg:{conv_system_msg}***")
        return conv

    def reload_conv(conv):
        """
        Reprints the conversation from the start.
        """
        for message in conv.messages[conv.offset :]:
            chatio.prompt_for_output(message[0])
            chatio.print_output(message[1])

    conv = None

    while True:
        if not history or not conv:
            conv = new_chat()

        try:
            inp = chatio.prompt_for_input(conv.roles[0])#用户输入
        except EOFError:
            inp = ""
        print(f"***3 chat_loop, inp:{inp}***")
        if inp == "!!exit" or not inp:
            print("exit...")
            break
        elif inp == "!!reset":
            print("resetting...")
            conv = new_chat()
            continue
        elif inp == "!!remove":
            print("removing last message...")
            if len(conv.messages) > conv.offset:
                # Assistant
                if conv.messages[-1][0] == conv.roles[1]:
                    conv.messages.pop()
                # User
                if conv.messages[-1][0] == conv.roles[0]:
                    conv.messages.pop()
                reload_conv(conv)
            else:
                print("No messages to remove.")
            continue
        elif inp == "!!regen":
            print("regenerating last message...")
            if len(conv.messages) > conv.offset:
                # Assistant
                if conv.messages[-1][0] == conv.roles[1]:
                    conv.messages.pop()
                # User
                if conv.messages[-1][0] == conv.roles[0]:
                    reload_conv(conv)
                    # Set inp to previous message
                    inp = conv.messages.pop()[1]
                else:
                    # Shouldn't happen in normal circumstances
                    print("No user message to regenerate from.")
                    continue
            else:
                print("No messages to regenerate.")
                continue
        elif inp.startswith("!!save"):
            args = inp.split(" ", 1)

            if len(args) != 2:
                print("usage: !!save <filename>")
                continue
            else:
                filename = args[1]

            # Add .json if extension not present
            if not "." in filename:
                filename += ".json"

            print("saving...", filename)
            with open(filename, "w") as outfile:
                json.dump(conv.dict(), outfile)
            continue
        elif inp.startswith("!!load"):
            args = inp.split(" ", 1)

            if len(args) != 2:
                print("usage: !!load <filename>")
                continue
            else:
                filename = args[1]

            # Check if file exists and add .json if needed
            if not os.path.exists(filename):
                if (not filename.endswith(".json")) and os.path.exists(
                    filename + ".json"
                ):
                    filename += ".json"
                else:
                    print("file not found:", filename)
                    continue

            print("loading...", filename)
            with open(filename, "r") as infile:
                new_conv = json.load(infile)

            conv = get_conv_template(new_conv["template_name"])
            conv.set_system_message(new_conv["system_message"])
            conv.messages = new_conv["messages"]
            reload_conv(conv)
            continue

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if is_codet5p:  # codet5p is a code completion model.
            prompt = inp
        print(f"***4 chat_loop, prompt:{prompt} and temperature:{temperature} and  repetition_penalty:{repetition_penalty}***")
        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        try:
            print(f"***5 chat_loop, conv.roles[1]:{conv.roles[1]}***")
            chatio.prompt_for_output(conv.roles[1])#机器人输出
            output_stream = generate_stream_func(
                model,
                tokenizer,
                gen_params,
                device,
                context_len=context_len,
                judge_sent_end=judge_sent_end,
            )#调用generate_stream函数
            t = time.time()
            outputs = chatio.stream_output(output_stream)
            duration = time.time() - t
            print(f"***6 chat_loop, type(outputs):{type(outputs)} and outputs.stip():{outputs.strip()}***")
            conv.update_last_message(outputs.strip())
            debuf = True 
            if debug:
                num_tokens = len(tokenizer.encode(outputs))
                msg = {
                    "conv_template": conv.name,
                    "prompt": prompt,
                    "outputs": outputs,
                    "speed (token/s)": round(num_tokens / duration, 2),
                }
                print(f"\n{msg}\n")

        except KeyboardInterrupt:
            print("stopped generation.")
            # If generation didn't finish
            if conv.messages[-1][1] is None:
                conv.messages.pop()
                # Remove last user message, so there isn't a double up
                if conv.messages[-1][0] == conv.roles[0]:
                    conv.messages.pop()

                reload_conv(conv)
