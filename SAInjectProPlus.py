from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset
from struct import pack, unpack
import torch
from cmath import isinf, isnan
import random
import torch
import numpy as np

logging.set_verbosity_error()

import argparse
import sys
import json

import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from tool.fault_injector_d import SA_FaultInjector_d as SA_FaultInjector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MAX_float_16 = 65504
calculateSize = 16
headSize = 128

inject_pos = {'value': 0}
selectHead = {'value': 0}
block = 8
blockX = 9
blockY = 9
flag = {'value': 0, 'normal': 0}
pos_info = {
    'faultPosInCul_X': 0,
    'faultPosInCul_Y': 0,
    'blockPos_X': 0,
    'blockPos_Y': 0,
}
faultin = { 'value': 0 }
set_fault = {"fx": 0, "fy": 0, "bx": 0, "by": 0}



def Fp32Bitflip(data, pos):
    fs = pack('f', data)
    bval = list(unpack('BBBB', fs))
    q, r = divmod(pos, 8)
    bval[q] ^= 1 << r
    fs = pack('BBBB', *bval)
    fnew = unpack('f', fs)[0]
    if isnan(fnew) or isinf(fnew):
        fnew = 1.0 if data > 0 else 0.0
    if abs(fnew) > 1e4:
        fnew = 1.0 if data > 0 else -1.0
    if data >= 0 and fnew < 0:
        fnew = 0.0

    return fnew

def BFloat16Bitflip(data: torch.Tensor, pos: int) -> torch.Tensor:
    assert data.dtype == torch.bfloat16 and data.numel() == 1, "data must be a scalar tensor of dtype torch.bfloat16"

    int_repr = data.view(torch.uint16).item()
    int_repr ^= 1 << pos  # pos in [0, 15]

    if int_repr >> pos & 1 == 1:
        pos_info['z2o'].append(1) 
    else:
        pos_info['z2o'].append(0)

    flipped_tensor = torch.tensor(int_repr, dtype=torch.uint16).view(torch.bfloat16)

    return flipped_tensor

def get_module_by_path(module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr)
    return module

faultpos = {'value':0}

pos_mapping = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "mlp-gate": "mlp.gate_proj",
    "mlp-up": "mlp.up_proj",
    "mlp-down": "mlp.down_proj",
    "all": "all"
}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--layerList", type=int, nargs='+', default=[0], help="Input fault injection layer numbers [0-27], default is 0")
    parser.add_argument("--affect", action='store_true', help="Enable injection to subsequent layers, default is False")
    parser.add_argument("--pos", type=int, nargs='+', default=[11,12,13,14], help="Input fault injection positions, default is empty, separate positions with spaces")
    parser.add_argument("--layerType",  type=str, choices=pos_mapping.keys(),
                        help="Injection layer options: q, k, v, ffn-gate, ffn-up, ffn-down, attention-norm, input-norm")
    parser.add_argument("--outputfile", type=str, default="/workplace/home/aistation/qwen3/bf16bit/rank/", help="Input directory for storing fault injection results, default is /home/aistation/python-code/qwen3/bf16bit/normlayer/, filename will be automatically appended based on injection position, file type is txt")
    parser.add_argument("--run", type=int, default=1, help="Number of runs for selected samples, default is 1")
    parser.add_argument("--faultin", type=int, nargs=4, default=[], help="Input 4 injection position parameters, in order: fx fy bx by, default is all random values" )
    parser.add_argument("--sampleid", type=int, nargs='*', default=[], help="Specify sample IDs within the randomly selected 1000 samples, default is all 1000 samples for fault injection testing")
    parser.add_argument("--pe", type=int, nargs=2, default=[-1, -1], help="Manually specify injection PE position, format: row col, default -1 -1 means random selection")
    parser.add_argument("--injectConfig", type=str, default="input_stuck_at_1", help="Fault injection type, default is input_stuck_at_1")

    faultin['value'] = 1 if "--faultin" in sys.argv else 0
    ifsample = True if "--sampleid" in sys.argv else False
    
    args = parser.parse_args()


    if args.layerType is None:
        print("[ERROR] Fault injection mode must specify --layerType parameter!")
        sys.exit(1)
    
    layertype = args.layerType
    print(f"[INFO] Injection layer: {layertype}")    
    
    if faultin['value'] == 1:
        set_fault['fx'] = args.faultin[0] if args.faultin else 0
        set_fault['fy'] = args.faultin[1] if len(args.faultin) > 1 else 0
        set_fault['bx'] = args.faultin[2] if len(args.faultin) > 2 else 0
        set_fault['by'] = args.faultin[3] if len(args.faultin) > 3 else 0
        print(f"[INFO] Injection position parameters set: fx={set_fault['fx']}, fy={set_fault['fy']}, bx={set_fault['bx']}, by={set_fault['by']}")
    else:
        print("[INFO] Injection position parameters set to random values")
        
        
    poslist = args.pos
    layerlist = args.layerList
    print(f"[INFO] Injection position list: {poslist}")
    print(f"[INFO] Injection layer list: {layerlist}")
    if not args.outputfile.endswith("/"):
        args.outputfile += "/"
    out_path = args.outputfile
    runtimes = args.run
    sample_index = args.sampleid
    print(f"[INFO] Output file path: {out_path}")
    print(f"[INFO] Fault injection runs: {runtimes}")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct").to(device)
    model.eval()
    print(f"[INFO] Model loaded")

    ds = load_dataset("rajpurkar/squad", split="validation")
    random.seed(37)
    samples = random.sample(list(ds), 200)
    print("[INFO] Dataset loaded")

    if sample_index:
        print(f"[INFO] Specified sample IDs: {sample_index}")
        try:
            samples = [samples[i] for i in sample_index]
        except IndexError as e:
            raise ValueError(f"Specified sample ID out of range: {e}")
    else:
        print("[INFO] No sample IDs specified, using all randomly selected samples for fault injection testing")
        sample_index = [str(i) for i in range(len(samples))]
    
    Fault_Type = args.injectConfig  
    injector = SA_FaultInjector(sa_rows=128, sa_cols=128, fault_type=Fault_Type)
    print(f"[INFO] Fault injection module initialized, size 128x128, fault type: {Fault_Type}")
    # injector.enabled = False
    if args.pe != [-1, -1]:
        injector.set_fault_position(args.pe[0], args.pe[1])
    else:
        injector.init_fault_position()
    handles =[]
    for posid in poslist:
        injector.set_fault_config_injpos(posid)
        for layerid in layerlist:
            # hook = get_module_by_path(model.model.layers[layerid], pos_mapping[layertype])
            for runid in range(runtimes):
                print(f"[INFO] Injection layer: {layertype}")   
                print(f"[INFO] Current injection position: {posid} in {poslist}")
                print(f"[INFO] Current injection layer: {layerid} in {layerlist}")
                print(f"[INFO] Run count: {runid + 1}/{runtimes}")
                inject_pos['value'] = posid

                # if faultin['value'] == 1:
                    # output_path = out_path + str(layerid) + "_" + layertype + "_" + str(posid) + "_" + str(set_fault['fx']) + "_" + str(set_fault['fy']) + "_" + str(set_fault['bx']) + "_" + str(set_fault['by']) + ".jsonl"
                if injector.enabled == False:
                    output_path = out_path + f"origin_{len(samples)}.jsonl"
                else:
                    output_path = f"{out_path}{injector.fault_config['mode']}_{injector.fault_config['type']}{injector.fault_config['stuck'] if injector.fault_config['type'] == 'stuck' else ''}_{layertype}_L{layerid}{'plus' if args.affect else ''}_P{posid}_FPE{args.pe[0]},{args.pe[1]}.jsonl"
                if args.affect:
                    print(f"[INFO] Subsequent layer injection mode enabled")
                    lastLayer = 28
                else:
                    print(f"[INFO] Only inject specified layer mode")
                    lastLayer = layerid + 1
                for layernumber in range(lastLayer):
                    if layernumber < layerid:
                        continue
                    if pos_mapping[layertype] != "all":
                        hook = get_module_by_path(model.model.layers[layernumber], pos_mapping[layertype])
                        hookRegister = hook.register_forward_hook(injector.hook_fn)
                        handles.append(hookRegister)
                    else:
                        for kernal in pos_mapping.values():
                            if kernal == "all":
                                continue
                            hook = get_module_by_path(model.model.layers[layernumber], kernal)
                            hookRegister = hook.register_forward_hook(injector.hook_fn)
                            handles.append(hookRegister)
                print(f"[INFO] Fault injection hooks registered on layer {layerid} and all subsequent {layertype} modules")

                with open(output_path, "a", encoding="utf-8") as f:
                    for idx, sample in enumerate(tqdm(samples)):
                        faultpos['value'] = 0
                        flag['normal'] = 0
                        messages = [
                            {
                                'role': 'user',
                                'content': "Read the following passage and answer the question with a text span directly from the passage.\n" + sample['context'] + "\nQuestion: " + sample['question'] + "\nAnswer:"
                            }
                        ]
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                        tokenized = tokenizer(prompt, return_tensors="pt")
                        num_tokens = tokenized["input_ids"].shape[-1]

                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=200,
                                do_sample=False,
                                top_p=0.95,
                                temperature=0,
                                eos_token_id=tokenizer.eos_token_id,
                            )
                            
                            token_length = outputs[0].shape[0] - num_tokens

                        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        sep = 'assistant\n'
                        try:
                            response = decoded.split(sep, 1)[1].strip().replace("\n\n", "\n")
                        except IndexError:
                            response = decoded.strip().replace("\n\n", "\n")

                        result_data = {
                            "sample_id": sample_index[idx],
                            "token_length": token_length,
                            "reference_answer": sample['answers'],
                            "thinking_process": response.split('</think>', 1)[0].strip() if '</think>' in decoded else response,
                            "generated_answer": response.split('</think>', 1)[1].strip() if '</think>' in decoded else "",
                            "prompt": prompt,
                        }
                        json_line = json.dumps(result_data, ensure_ascii=False)
                        f.write(json_line + "\n")
                        f.flush()
                        
                print(f"Generation completed, results saved at: {output_path}")
                for handle in handles:
                    handle.remove()
