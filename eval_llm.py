import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")

    # 参数名：模型加载路径
    # 作用：指定从哪里加载推理模型；传 model 表示加载项目原生 .pth 权重，传目录路径表示加载 transformers 格式模型。
    # 范围：【model，minimind-3，其他transformers模型目录】
    # 影响：使用原生权重时会走本项目的 MiniMind 模型结构；使用 transformers 目录时会走 AutoModelForCausalLM。
    # 推荐值：【model】测试本项目训练出的 pretrain / full_sft 权重时使用。
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")

    # 参数名：模型权重目录
    # 作用：指定原生 .pth 权重所在目录；仅在 --load_from=model 时生效。
    # 范围：【out，自定义权重目录名】
    # 影响：目录写错会导致找不到 pretrain_768.pth 或 full_sft_768.pth。
    # 推荐值：【out】使用项目默认训练输出目录时最方便。
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")

    # 参数名：权重前缀名
    # 作用：指定要加载哪种训练阶段产出的权重，如预训练权重或指令微调权重。
    # 范围：【pretrain，full_sft，rlhf，reason，ppo_actor，grpo，spo】
    # 影响：选择 pretrain 时更像续写模型；选择 full_sft 时更像指令助手。
    # 推荐值：【full_sft】测试当前对话效果时优先使用；【pretrain】做对比测试时使用。
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")

    # 参数名：LoRA权重名
    # 作用：在主模型基础上额外挂载 LoRA 权重进行推理。
    # 范围：【None，lora_identity，lora_medical，自定义LoRA权重前缀】
    # 影响：为 None 时不加载 LoRA；加载后会让模型输出风格偏向对应任务领域。
    # 推荐值：【None】当前常规测试先不要叠加 LoRA。
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")

    # 参数名：隐藏层维度
    # 作用：构建原生 MiniMind 模型时使用的隐藏层大小；需要和权重真实结构一致。
    # 范围：【512，768，1024】
    # 影响：写错会导致模型结构与权重不匹配，加载失败。
    # 推荐值：【768】当前这个项目默认 64M 模型使用该值。
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")

    # 参数名：隐藏层数量
    # 作用：构建原生 MiniMind 模型时使用的 Transformer 层数；需要和权重真实结构一致。
    # 范围：【4，8，12】
    # 影响：写错会导致权重维度不匹配或加载失败。
    # 推荐值：【8】当前项目默认模型结构使用该值。
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")

    # 参数名：是否使用MoE
    # 作用：指定加载的是否为 MoE 结构模型。
    # 范围：【0，1】
    # 影响：0 表示普通 Dense 模型；1 表示 MoE 模型，权重文件名也会带 _moe 后缀。
    # 推荐值：【0】当前大多数常规训练与测试场景都用普通 Dense 模型。
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")

    # 参数名：RoPE外推
    # 作用：推理时启用位置编码外推，用于尝试支持更长上下文。
    # 范围：【False，True】
    # 影响：开启后可能改善超出训练长度时的位置编码问题，但不保证真正提升长文本理解质量。
    # 推荐值：【False】常规测试先关闭；确实需要更长上下文时再尝试开启。
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")

    # 参数名：最大生成token数
    # 作用：限制模型单次最多新生成多少 token。
    # 范围：【32，64，128，256，512，1024，2048，4096，8192】
    # 影响：越大越可能生成更长回答，但也更容易拖慢速度、放大重复输出；越小越省时，但回答可能过早截断。
    # 推荐值：【128，256】日常问答测试；【512】需要长回答时再提高。
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")

    # 参数名：生成温度
    # 作用：控制采样随机性，决定输出是更保守还是更发散。
    # 范围：【0.1，0.3，0.5，0.7，0.85，1.0】
    # 影响：越大越随机、越活跃，但更容易跑偏和重复；越小越稳定、越保守，但可能更死板。
    # 推荐值：【0.7】测试 SFT 模型稳定问答时；【0.85】想保留一定表达变化时使用。
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")

    # 参数名：Top-p采样阈值
    # 作用：只在累计概率达到阈值的候选 token 中采样，控制生成时的候选范围。
    # 范围：【0.7，0.8，0.9，0.95，0.98，1.0】
    # 影响：越大候选越多，输出更发散；越小候选越少，输出更保守稳定。
    # 推荐值：【0.9】减少跑偏和重复时使用；【0.95】当前常规测试可用。
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")

    # 参数名：重复惩罚系数
    # 作用：降低已经生成过的 token 再次被选中的概率，用于缓解复读和循环输出。
    # 范围：【1.0，1.05，1.1，1.15，1.2】
    # 影响：越大越不容易重复，但太大可能让回答变僵、用词不自然；1.0 表示不做重复惩罚。
    # 推荐值：【1.1】轻微重复时使用；【1.15】重复明显时使用。
    parser.add_argument('--repetition_penalty', default=1.0, type=float, help="重复惩罚系数（>=1，越大越不容易重复）")

    # 参数名：是否开启思考模式
    # 作用：控制聊天模板里是否启用自适应思考模式。
    # 范围：【0，1】
    # 影响：开启后可能生成带思考风格的回答；关闭后更直接。
    # 推荐值：【0】当前常规问答测试先关闭，避免输出额外思考痕迹。
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")

    # 参数名：携带历史轮数
    # 作用：控制保留多少轮历史对话，让模型做多轮上下文回答。
    # 范围：【0，2，4，6，8】
    # 影响：越大越能参考前文，但也更占上下文并更容易把旧内容带入当前回答；0 表示每轮独立测试。
    # 推荐值：【0】单轮测评时使用；【2】需要简单多轮跟进时使用。
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")

    # 参数名：是否显示生成速度
    # 作用：控制是否打印每次生成的 tokens/s 速度信息。
    # 范围：【0，1】
    # 影响：1 会输出测速信息，便于看性能；0 输出更干净。
    # 推荐值：【1】本地调试和远端测速时使用；【0】只关心内容时使用。
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")

    # 参数名：运行设备
    # 作用：指定推理运行在哪个设备上。
    # 范围：【cpu，cuda，cuda:0，cuda:1，mps】
    # 影响：GPU/MPS 速度更快；CPU 最慢但兼容性最高。
    # 推荐值：【cuda:0】远端单卡测试；【mps】本地 Mac 且 MPS 可用时使用。
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=args.repetition_penalty
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()
