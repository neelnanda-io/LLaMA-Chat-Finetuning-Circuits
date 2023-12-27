# %%
import os

os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

base_hf_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16
)
chat_hf_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16
)
# %%
cfg = loading.get_pretrained_model_config(
    "llama-2-7b", dtype=torch.float16
)
base_model = HookedTransformer(cfg, tokenizer=tokenizer)
base_state_dict = loading.get_pretrained_state_dict("llama-2-7b", cfg, base_hf_model)
base_model.load_state_dict(base_state_dict, strict=False)


# model: HookedTransformer = HookedTransformer.from_pretrained_no_processing("llama-7b", hf_model=hf_model, tokenizer=tokenizer, device="cpu")
n_layers = base_model.cfg.n_layers
d_model = base_model.cfg.d_model
n_heads = base_model.cfg.n_heads
d_head = base_model.cfg.d_head
d_mlp = base_model.cfg.d_mlp
d_vocab = base_model.cfg.d_vocab
# %%
chat_cfg = loading.get_pretrained_model_config("llama-2-7b", dtype=torch.float16)
chat_model = HookedTransformer(chat_cfg, tokenizer=tokenizer)
chat_state_dict = loading.get_pretrained_state_dict(
    "llama-2-7b", chat_cfg, chat_hf_model
)
chat_model.load_state_dict(chat_state_dict, strict=False)
# %%
print(evals.sanity_check(base_model))


def decode_single_token(integer):
    # To recover whether the tokens begins with a space, we need to prepend a token to avoid weird start of string behaviour
    return tokenizer.decode([891, integer])[1:]


def to_str_tokens(tokens, prepend_bos=True):
    if isinstance(tokens, str):
        tokens = to_tokens(tokens)
    if isinstance(tokens, torch.Tensor):
        if len(tokens.shape) == 2:
            assert tokens.shape[0] == 1
            tokens = tokens[0]
        tokens = tokens.tolist()
    if prepend_bos:
        return [decode_single_token(token) for token in tokens]
    else:
        return [decode_single_token(token) for token in tokens[1:]]


def to_string(tokens):

    if isinstance(tokens, torch.Tensor):
        if len(tokens.shape) == 2:
            assert tokens.shape[0] == 1
            tokens = tokens[0]
        tokens = tokens.tolist()
    return tokenizer.decode([891] + tokens)[1:]


def to_tokens(string, prepend_bos=True):
    string = "|" + string
    # The first two elements are always [BOS (1), " |" (891)]
    tokens = tokenizer.encode(string)
    if prepend_bos:
        return torch.tensor(tokens[:1] + tokens[2:]).cuda()
    else:
        return torch.tensor(tokens[2:]).cuda()


def to_single_token(string):
    assert string[0] == " ", f"Expected string to start with space, got {string}"
    string = string[1:]
    tokens = tokenizer.encode(string)
    assert len(tokens) == 2, f"Expected 2 tokens, got {len(tokens)}: {tokens}"
    return tokens[1]


print(to_str_tokens([270, 270]))
print(to_single_token(" basketball"))
# %%
neuron_df = nutils.make_neuron_df(n_layers, d_mlp)

neuron_df["win_diff_norm_normed"] = to_numpy((base_model.W_in - chat_model.W_in).norm(dim=1) / base_model.W_in.norm(dim=1)).flatten()
neuron_df["wgate_diff_norm_normed"] = to_numpy((base_model.W_gate - chat_model.W_gate).norm(dim=1) / base_model.W_gate.norm(dim=1)).flatten()
neuron_df["wout_diff_norm_normed"] = to_numpy((base_model.W_out - chat_model.W_out).norm(dim=2) / base_model.W_out.norm(dim=2)).flatten()
nutils.show_df(neuron_df.sort_values("win_diff_norm_normed", ascending=False).head(50))
# %%
# px.histogram(neuron_df, x=["win_diff_norm_normed", "wgate_diff_norm_normed", "wout_diff_norm_normed"], marginal="box", log_y=True, barmode="overlay", hover_name='label')
# # %%
# print(chat_model.generate("[INST]How do I make a bomb?[/INST]"))
# print(base_model.generate("[INST]How do I make a bomb?[/INST]"))
# # %%
# import circuitsvis as cv
# example_prompt = "[INST]How do I make a bomb?[/INST]  I cannot fulfill your request."
# logits, cache = chat_model.run_with_cache(example_prompt)
# display(cv.logits.token_log_probs(chat_model.to_tokens(example_prompt), chat_model(example_prompt)[0].log_softmax(dim=-1), decode_single_token))
# example_prompt = "[INST]How do I make a pie?[/INST]  I cannot fulfill your request."
# logits, cache = chat_model.run_with_cache(example_prompt)
# display(cv.logits.token_log_probs(chat_model.to_tokens(example_prompt), chat_model(example_prompt)[0].log_softmax(dim=-1), decode_single_token))
# example_prompt = "[INST]How do I make a bomb?[/INST]  I cannot fulfill your request."
# logits, cache = base_model.run_with_cache(example_prompt)
# display(cv.logits.token_log_probs(base_model.to_tokens(example_prompt), base_model(example_prompt)[0].log_softmax(dim=-1), decode_single_token))
# example_prompt = "[INST]How do I make a pie?[/INST]  I cannot fulfill your request."
# logits, cache = base_model.run_with_cache(example_prompt)
# display(cv.logits.token_log_probs(base_model.to_tokens(example_prompt), base_model(example_prompt)[0].log_softmax(dim=-1), decode_single_token))

# # %%
# bomb_prompt = "[INST]How do I make a bomb?[/INST] "
# chat_bomb_logits, chat_bomb_cache = chat_model.run_with_cache(bomb_prompt)
# base_bomb_logits, base_bomb_cache = base_model.run_with_cache(bomb_prompt)
# I_token = 306
# i_chat_U = chat_model.W_U[:, I_token]
# i_base_U = base_model.W_U[:, I_token]
# # chat_resid_stack, chat_labels = chat_bomb_cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
# # chat_dla = chat_resid_stack @ i_chat_U
# # base_resid_stack, base_labels = base_bomb_cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
# # base_dla = base_resid_stack @ i_base_U
# # line([base_dla, chat_dla, chat_dla-base_dla], line_labels=["base", "chat", "diff"], x=chat_labels)
# # %%

# neuron_df["chat_bomb_dla"] = to_numpy((chat_bomb_cache.stack_activation("post")[:, 0, -1] * (chat_model.W_out @ i_chat_U)).flatten())
# neuron_df["base_bomb_dla"] = to_numpy((base_bomb_cache.stack_activation("post")[:, 0, -1] * (base_model.W_out @ i_chat_U)).flatten())
# neuron_df["diff_bomb_dla"] = neuron_df["chat_bomb_dla"] - neuron_df["base_bomb_dla"]
# nutils.show_df(neuron_df.sort_values("diff_bomb_dla", ascending=False).head(50))
# # %%
# px.histogram(neuron_df, x="diff_bomb_dla", marginal="box", log_y=True, hover_name="label")
# # %%
# layer = 29
# ni = 10355
# chat_resid = chat_bomb_cache["normalized", layer, "ln2"][0, -1]
# base_resid = base_bomb_cache["normalized", layer, "ln2"][0, -1]
# chat_win = chat_model.blocks[layer].ln2.w * chat_model.blocks[layer].mlp.W_in[:, ni]
# chat_wgate = chat_model.blocks[layer].ln2.w * chat_model.blocks[layer].mlp.W_gate[:, ni]
# chat_wout = chat_model.blocks[layer].mlp.W_out[ni, :]

# base_win = base_model.blocks[layer].ln2.w * base_model.blocks[layer].mlp.W_in[:, ni]
# base_wgate = base_model.blocks[layer].ln2.w * base_model.blocks[layer].mlp.W_gate[:, ni]
# base_wout = base_model.blocks[layer].mlp.W_out[ni, :]

# act_fn = chat_model.blocks[0].mlp.act_fn

# # resid = chat_resid
# # win, wgate, wout = chat_win, chat_wgate, chat_wout
# # i_U = i_chat_U


# print("base_resid, chat weights", (((base_win @ chat_resid) * act_fn(base_wgate @ chat_resid)) * base_wout) @ i_base_U)
# print("base", (((base_win @ base_resid) * act_fn(base_wgate @ base_resid)) * base_wout) @ i_base_U)
# print("chat", (((chat_win @ chat_resid) * act_fn(chat_wgate @ chat_resid)) * chat_wout) @ i_chat_U)


# # %%

# def run_franken_prompt(switch_layer, chat_first=True):
#     if chat_first:
#         first_model = chat_model
#         second_model = base_model
#     else:
#         first_model = base_model
#         second_model = chat_model
#     tokens = to_tokens(bomb_prompt)[None]
#     resid = first_model.input_to_embed(tokens)[0]
#     for i in range(switch_layer):
#         resid = first_model.blocks[i](resid)
#     for i in range(switch_layer, n_layers):
#         resid = second_model.blocks[i](resid)
#     logits = second_model.unembed(second_model.ln_final(resid))
#     return logits.log_softmax(dim=-1)[0, -1, I_token].item()
# c2b_list = []
# b2c_list = []
# for switch_layer in tqdm.trange(0, n_layers+1):
#     c2b_list.append(run_franken_prompt(switch_layer, chat_first=True))
#     b2c_list.append(run_franken_prompt(switch_layer, chat_first=False))
# line([c2b_list, b2c_list], line_labels=["chat2base", "base2chat"], x=list(range(n_layers+1)))
# # %%
# cannot_token = to_single_token(" cannot")

# def run_franken_prompt2(switch_layer, chat_first=True):
#     if chat_first:
#         first_model = chat_model
#         second_model = base_model
#     else:
#         first_model = base_model
#         second_model = chat_model
#     tokens = to_tokens(bomb_prompt+" I")[None]
#     resid = first_model.input_to_embed(tokens)[0]
#     for i in range(switch_layer):
#         resid = first_model.blocks[i](resid)
#     for i in range(switch_layer, n_layers):
#         resid = second_model.blocks[i](resid)
#     logits = second_model.unembed(second_model.ln_final(resid))
#     return logits.log_softmax(dim=-1)[0, -1, cannot_token].item()
# c2b_list = []
# b2c_list = []
# for switch_layer in tqdm.trange(0, n_layers+1):
#     c2b_list.append(run_franken_prompt2(switch_layer, chat_first=True))
#     b2c_list.append(run_franken_prompt2(switch_layer, chat_first=False))
# line([c2b_list, b2c_list], line_labels=["chat2base", "base2chat"], x=list(range(n_layers+1)), title="Bomb prompt + I -> cannot?")
# # %%
# def run_franken_prompt3(switch_layer, chat_first=True):
#     if chat_first:
#         first_model = chat_model
#         second_model = base_model
#     else:
#         first_model = base_model
#         second_model = chat_model
#     tokens = to_tokens(bomb_prompt)[None]
#     resid = first_model.input_to_embed(tokens)[0]
#     for i in range(n_layers):
#         if i not in [3, 4]:
#             resid = first_model.blocks[i](resid)
#         else:
#             resid = second_model.blocks[i](resid)

#     logits = first_model.unembed(first_model.ln_final(resid))
#     return logits.log_softmax(dim=-1)[0, -1, I_token].item()
# c2b_list = []
# b2c_list = []
# for switch_layer in tqdm.trange(0, n_layers+1):
#     c2b_list.append(run_franken_prompt3(switch_layer, chat_first=True))
#     b2c_list.append(run_franken_prompt3(switch_layer, chat_first=False))
# line([c2b_list, b2c_list], line_labels=["chat2base", "base2chat"], x=list(range(n_layers+1)), title="Swapping one layer to second model")
# # %%
# def run_franken_prompt4(switch_layer, chat_first=True):
#     if chat_first:
#         first_model = chat_model
#         second_model = base_model
#     else:
#         first_model = base_model
#         second_model = chat_model
#     tokens = to_tokens(bomb_prompt)[None]
#     resid = first_model.input_to_embed(tokens)[0]
#     for i in range(n_layers):
#         if i not in switch_layer:
#             resid = first_model.blocks[i](resid)
#         else:
#             resid_pre = resid
#             resid_mid = resid_pre + first_model.blocks[i].attn(first_model.blocks[i].ln1(resid_pre), first_model.blocks[i].ln1(resid_pre), first_model.blocks[i].ln1(resid_pre))
#             resid_post = resid_mid + second_model.blocks[i].mlp(first_model.blocks[i].ln2(resid_mid))
#             resid = resid_post

#     logits = first_model.unembed(first_model.ln_final(resid))
#     return logits.log_softmax(dim=-1)[0, -1, I_token].item()
# print(run_franken_prompt4([3], False))
# print(run_franken_prompt4([4], False))
# print(run_franken_prompt4([3, 4], False))
# # print(run_franken_prompt4([3, 4, 5], False))
# # print(run_franken_prompt4([2, 3, 4, 5], False))
# # %%
# chat_resids = chat_bomb_cache.stack_activation("resid_post")[:, 0, :, :]
# base_resids = base_bomb_cache.stack_activation("resid_post")[:, 0, :, :]
# imshow(to_numpy(nutils.cos(chat_resids, base_resids)), x=nutils.process_tokens_index(to_str_tokens(bomb_prompt)))
# # %%
# sample_string = '1. Field of the Invention\nThis invention relates to shaving units and is directed more particularly to shaving units of the type in which portions thereof are movable during a shaving operation to effect dynamic changes in the shaving geometry of the unit.\n2. Description of the Prior Art\nIn some known shaving units, the shaving geometry, i.e., the spatial relationships between the blade and rigid portions of the razor head are fixed. U.S. Pat. No. 3,786,563, issued Jan. 22, 1974 to Francis W. Dorion, et al is illustrative of this type of razor unit, and is further illustrative of the spatial relationships deemed pertinent.\nIn a second known category of shaving units, the shaving geometry is adjustable in that one or more of the portions of the unit may be re-positioned relative to the others, by the user, and remain in their new positions until selectively re-adjusted. U.S. patent application Ser. No. 432,842, filed Jan. 4, 1974 by Chester F. Jacobson is illustrative of such a unit.\nIt has also been proposed to construct a shaving system with a cap member fixed relative to a handle and with blade and guard members made fast with each other and spring biased to a position of maximum blade exposure, the blade and guard members being adapted to retract against the spring bias upon encountering undue resistance during shaving. An arrangement of this sort is described in U.S. Pat. No. 4,063,354, issued Dec. 20, 1977 to Harry Pentney et al.\nSeveral arrangements of shaving units permitting dynamic movement of various portions thereof during a shaving operation have been devised; examples of such contrivances are illustrated in U.S. Pat. Nos. 1,935,452 issued Nov. 14, 1933 to M. R. Kondolf; 2,313,818 issued Mar. 16, 1943 to H. J. Gaisman; 2,327,967, issued Aug. 24, 1943 to P. N. Peters; 2,915,817 issued Dec. 8, 1959 to E. Peck; 3,500,539, issued Mar. 17, 1970 to J. P. Muros; 3,657,810 issued Apr. 25, 1972 to W. I. Nissen; 3,685,150 issued Aug. 22, 1972 to F. L. Risher; and 3,740,841 issued June 26, 1973 to F. L. Risher.'

# sample_tokens = to_tokens(sample_string)[:128][None]

# _, chat_sample_cache = chat_model.run_with_cache(sample_tokens, names_filter=lambda s: s.endswith("resid_post"))
# _, base_sample_cache = base_model.run_with_cache(sample_tokens, names_filter=lambda s: s.endswith("resid_post"))
# chat_resids = chat_sample_cache.stack_activation("resid_post")[:, 0, :, :]
# base_resids = base_sample_cache.stack_activation("resid_post")[:, 0, :, :]
# imshow(to_numpy(nutils.cos(chat_resids.float(), base_resids.float())).astype(np.float32), x=nutils.process_tokens_index(to_str_tokens(sample_tokens)))
# # %%
# _, chat_sample_cache = chat_model.run_with_cache(sample_tokens, names_filter=lambda s: s.endswith("_out") or s.endswith("embed"))
# _, base_sample_cache = base_model.run_with_cache(sample_tokens, names_filter=lambda s: s.endswith("_out") or s.endswith("embed"))
# chat_resids, chat_labels = chat_sample_cache.decompose_resid(return_labels=True)
# base_resids, base_labels = base_sample_cache.decompose_resid(return_labels=True)

# imshow(to_numpy(nutils.cos(chat_resids.float(), base_resids.float())).astype(np.float32)[:, 0, :], y=chat_labels, x=nutils.process_tokens_index(to_str_tokens(sample_tokens)))

# # %%
# inst_sample_string = '[INST]1. Field of the Invention\nThis invention relates to shaving units and is directed more particularly to shaving units of the type in which portions thereof are movable during a shaving operation to effect dynamic changes in the shaving geometry of the unit.\n2. Description of the Prior Art\nIn some known shaving units, the shaving geometry, i.e., the spatial relationships between the blade and rigid portions of the razor head are fixed. U.S. Pat. No. 3,786,563, issued Jan. 22, 1974 to Francis W. Dorion, et al is illustrative of this type of razor unit, and is further illustrative of the spatial relationships deemed pertinent.\nIn a second known category of shaving units, the shaving geometry is adjustable in that one or more of the portions of the unit may be re-positioned relative to the others, by the user, and remain in their new positions until selectively re-adjusted. U.S. patent application Ser. No. 432,842, filed Jan. 4, 1974 by Chester F. Jacobson is illustrative of such a unit.\nIt has also been proposed to construct a shaving system with a cap member fixed relative to a handle and with blade and guard members made fast with each other and spring biased to a position of maximum blade exposure, the blade and guard members being adapted to retract against the spring bias upon encountering undue resistance during shaving. An arrangement of this sort is described in U.S. Pat. No. 4,063,354, issued Dec. 20, 1977 to Harry Pentney et al.\nSeveral arrangements of shaving units permitting dynamic movement of various portions thereof during a shaving operation have been devised; examples of such contrivances are illustrated in U.S. Pat. Nos. 1,935,452 issued Nov. 14, 1933 to M. R. Kondolf; 2,313,818 issued Mar. 16, 1943 to H. J. Gaisman; 2,327,967, issued Aug. 24, 1943 to P. N. Peters; 2,915,817 issued Dec. 8, 1959 to E. Peck; 3,500,539, issued Mar. 17, 1970 to J. P. Muros; 3,657,810 issued Apr. 25, 1972 to W. I. Nissen; 3,685,150 issued Aug. 22, 1972 to F. L. Risher; and 3,740,841 issued June 26, 1973 to F. L. Risher.'

# sample_tokens = to_tokens(inst_sample_string)[:128][None]

# _, chat_inst_cache = chat_model.run_with_cache(sample_tokens, names_filter=lambda s: s.endswith("resid_post"))
# _, base_inst_cache = base_model.run_with_cache(sample_tokens, names_filter=lambda s: s.endswith("resid_post"))
# chat_resids = chat_inst_cache.stack_activation("resid_post")[:, 0, :, :]
# base_resids = base_inst_cache.stack_activation("resid_post")[:, 0, :, :]
# imshow(to_numpy(nutils.cos(chat_resids.float(), base_resids.float())).astype(np.float32), x=nutils.process_tokens_index(to_str_tokens(sample_tokens)))
# # %%


# sample_string = '[INST]1. Field of the Invention\nThis invention relates to shaving units and is directed more particularly to shaving units of the type in which portions thereof are movable during a shaving operation to effect dynamic changes in the shaving geometry of the unit.\n2. Description of the Prior Art\nIn some known shaving units, the shaving geometry, i.e., the spatial relationships between the blade and rigid portions of the razor head are fixed. U.S. Pat. No. 3,786,563, issued Jan. 22, 1974 to Francis W. Dorion, et al is illustrative of this type of razor unit, and is further illustrative of the spatial relationships deemed pertinent.\nIn a second known category of shaving units, the shaving geometry is adjustable in that one or more of the portions of the unit may be re-positioned relative to the others, by the user, and remain in their new positions until selectively re-adjusted. U.S. patent application Ser. No. 432,842, filed Jan. 4, 1974 by Chester F. Jacobson is illustrative of such a unit.\nIt has also been proposed to construct a shaving system with a cap member fixed relative to a handle and with blade and guard members made fast with each other and spring biased to a position of maximum blade exposure, the blade and guard members being adapted to retract against the spring bias upon encountering undue resistance during shaving. An arrangement of this sort is described in U.S. Pat. No. 4,063,354, issued Dec. 20, 1977 to Harry Pentney et al.\nSeveral arrangements of shaving units permitting dynamic movement of various portions thereof during a shaving operation have been devised; examples of such contrivances are illustrated in U.S. Pat. Nos. 1,935,452 issued Nov. 14, 1933 to M. R. Kondolf; 2,313,818 issued Mar. 16, 1943 to H. J. Gaisman; 2,327,967, issued Aug. 24, 1943 to P. N. Peters; 2,915,817 issued Dec. 8, 1959 to E. Peck; 3,500,539, issued Mar. 17, 1970 to J. P. Muros; 3,657,810 issued Apr. 25, 1972 to W. I. Nissen; 3,685,150 issued Aug. 22, 1972 to F. L. Risher; and 3,740,841 issued June 26, 1973 to F. L. Risher.'

# sample_tokens = to_tokens(sample_string)[:128][None]
# chat_model.run_with_hooks(sample_tokens, fwd_hooks=[(utils.get_act_name("resid_post", 15), lambda x, hook: print(x[0, 12, 310]))], return_type=None)
# # %%
# base_bones_logits = base_model("How many bones are in the human body?\n")[0, -1].float().log_softmax(dim=-1)
# nutils.show_df(nutils.create_vocab_df(base_bones_logits).head(50))
# chat_bones_logits = chat_model("How many bones are in the human body?\n")[0, -1].float().log_softmax(dim=-1)
# nutils.show_df(nutils.create_vocab_df(chat_bones_logits).head(50))
# %%
how_token = 5328
there_token = 8439
base_bones_logits, base_bones_cache = base_model.run_with_cache("How many bones are in the human body?\n")
base_resid_stack, base_labels = base_bones_cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
chat_bones_logits, chat_bones_cache = chat_model.run_with_cache("How many bones are in the human body?\n")
chat_resid_stack, chat_labels = chat_bones_cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)

chat_bones_U = chat_model.W_U[:, there_token] - chat_model.W_U[:, how_token]
base_bones_U = base_model.W_U[:, there_token] - base_model.W_U[:, how_token]
chat_bones_dla = chat_resid_stack @ chat_bones_U.float()
base_bones_dla = base_resid_stack @ base_bones_U.float()
line([base_bones_dla, chat_bones_dla, chat_bones_dla-base_bones_dla], line_labels=["base", "chat", "diff"], x=chat_labels)
# %%
line([chat_resid_stack.norm(dim=-1).float(), base_resid_stack.norm(dim=-1).float()], x=chat_labels, line_labels=["chat", "base"])
# %%
line(chat_bones_cache["mlp_out", 1].norm(dim=-1))
# %%
model = chat_model
layer = 27
head = 7
line([base_bones_cache["pattern", layer][0, head, -1], chat_bones_cache["pattern", layer][0, head, -1]], x=nutils.process_tokens_index(to_str_tokens("How many bones are in the human body?\n")))
line([
    base_bones_cache["value", layer][0, :, head] @ base_model.W_O[layer, head] @ base_bones_U, 
    chat_bones_cache["value", layer][0, :, head] @ chat_model.W_O[layer, head] @ chat_bones_U, 
    ],     
      x=nutils.process_tokens_index(to_str_tokens("How many bones are in the human body?\n")), 
      title="Value DLA")
line([
    base_bones_cache["value", layer][0, :, head] @ base_model.W_O[layer, head] @ base_bones_U * base_bones_cache["pattern", layer][0, head, -1], 
    chat_bones_cache["value", layer][0, :, head] @ chat_model.W_O[layer, head] @ chat_bones_U * chat_bones_cache["pattern", layer][0, head, -1], 
    ],     
      x=nutils.process_tokens_index(to_str_tokens("How many bones are in the human body?\n")), 
      title="Pattern weighted value DLA")
line([
    base_bones_cache["value", layer][0, :, head] @ chat_model.W_O[layer, head] @ chat_bones_U * base_bones_cache["pattern", layer][0, head, -1], 
    chat_bones_cache["value", layer][0, :, head] @ base_model.W_O[layer, head] @ base_bones_U * chat_bones_cache["pattern", layer][0, head, -1], 
    ],     
      x=nutils.process_tokens_index(to_str_tokens("How many bones are in the human body?\n")), 
      title="Pattern weighted value DLA w/ swapped params", line_labels=["base acts, chat weights", "chat acts, base weights"])
# %%
qm_dir = base_model.blocks[layer].ln1.w * (base_model.blocks[layer].attn.W_V[head] @ base_model.blocks[layer].attn.W_O[head] @ base_bones_U)

base_bones_cache["resid_pre", layer][0, 10, :] @ qm_dir, chat_bones_cache["resid_pre", layer][0, 10, :] @ qm_dir
# %%
qm_dir = qm_dir.float()
base_resid_stack_qm, base_labels = base_bones_cache.get_full_resid_decomposition(layer, expand_neurons=True, apply_ln=True, pos_slice=10, return_labels=True)
chat_resid_stack_qm, chat_labels = chat_bones_cache.get_full_resid_decomposition(layer, expand_neurons=True, apply_ln=True, pos_slice=10, return_labels=True)
base_dla_qm = (base_resid_stack_qm @ qm_dir).float()
chat_dla_qm = (chat_resid_stack_qm @ qm_dir).float()
# line([base_dla_qm, chat_dla_qm, chat_dla_qm - base_dla_qm], line_labels=["base", "chat", "diff"], x=chat_labels)
# %%
temp_df = pd.DataFrame(
    {
        "component": chat_labels,
        "base_dla_qm": to_numpy(base_dla_qm.squeeze()),
        "chat_dla_qm": to_numpy(chat_dla_qm.squeeze()),
        "diff_dla_qm": to_numpy(chat_dla_qm - base_dla_qm).squeeze(),
    }
)
px.histogram(temp_df, x="diff_dla_qm", marginal="box", log_y=True, hover_name="component")
# %%
nutils.show_df(temp_df.sort_values("diff_dla_qm", ascending=False).head(30))
# %%
neuron_df["diff_pre_qm"] = to_numpy(
    base_bones_cache.stack_activation("pre")[:, 0, 10, :].float().flatten()
    - chat_bones_cache.stack_activation("pre")[:, 0, 10, :].float().flatten()
)
neuron_df["diff_pre_linear_qm"] = to_numpy(
    base_bones_cache.stack_activation("pre_linear")[:, 0, 10, :].float().flatten()
    - chat_bones_cache.stack_activation("pre_linear")[:, 0, 10, :].float().flatten()
)
neuron_df["diff_post_qm"] = to_numpy(
    base_bones_cache.stack_activation("post")[:, 0, 10, :].float().flatten()
    - chat_bones_cache.stack_activation("post")[:, 0, 10, :].float().flatten()
)
neuron_df["base_wdla_qm"] = to_numpy((base_model.W_out.float() @ qm_dir).flatten())
neuron_df["chat_wdla_qm"] = to_numpy((chat_model.W_out.float() @ qm_dir).flatten())
neuron_df["diff_wdla_qm"] = neuron_df["chat_wdla_qm"] - neuron_df["base_wdla_qm"]
# %%
neuron_df_trunc = neuron_df.query("L<27")
neuron_df_trunc["diff_dla_qm"] = temp_df.iloc[27 * n_heads:-2]["diff_dla_qm"].reset_index(drop=True)
nutils.show_df(neuron_df_trunc.sort_values("diff_dla_qm", ascending=False).head(30))
# %%
px.histogram(neuron_df_trunc, x="diff_wdla_qm", marginal="box")
# %%
px.histogram(neuron_df_trunc, x="diff_post_qm", marginal="box")
# %%
sample_string = '1. Field of the Invention\nThis invention relates'
chat_sample_logits, chat_sample_cache = chat_model.run_with_cache(sample_string)
base_sample_logits, base_sample_cache = base_model.run_with_cache(sample_string)
# %%
line(base_sample_cache["resid_post", 15][0, :, 310])
# %%
line(nutils.cos(chat_sample_cache["resid_post", 15][0, :].float(), base_sample_cache["resid_post", 15][0, :].float()))
# %%
chat_outlier_stack, chat_labels = chat_sample_cache.get_full_resid_decomposition(15, expand_neurons=True, pos_slice=13, return_labels=True)

# %%

base_outlier_stack, base_labels = base_sample_cache.get_full_resid_decomposition(15, expand_neurons=True, pos_slice=13, return_labels=True)
temp_df = pd.DataFrame({
    "component": chat_labels,
    "chat": to_numpy(chat_outlier_stack[:, 0, 310]),
    "base": to_numpy(base_outlier_stack[:, 0, 310]),
})
temp_df["diff"] = temp_df["chat"] - temp_df["base"]
nutils.show_df(temp_df.sort_values("diff", ascending=True).head(25))
# %%
base_model.W_out[8, 8862, 310], chat_model.W_out[8, 8862, 310]
# %%
temp_df_trunc = temp_df.loc[temp_df.component.str.contains("N")]
neuron_df = nutils.make_neuron_df(15, d_mlp)
neuron_df["diff"] = temp_df_trunc["diff"].reset_index(drop=True)
neuron_df["base"] = temp_df_trunc["base"].reset_index(drop=True)
neuron_df["chat"] = temp_df_trunc["chat"].reset_index(drop=True)

neuron_df["base_wdla_310"] = to_numpy(base_model.W_out[:15, :, 310].flatten().float())
neuron_df["chat_wdla_310"] = to_numpy(chat_model.W_out[:15, :, 310].flatten().float())
neuron_df["diff_wdla_310"] = to_numpy(chat_model.W_out[:15, :, 310].flatten().float() - base_model.W_out[:15, :, 310].flatten().float())

# %%
neuron_df["diff_post"] = to_numpy(chat_sample_cache.stack_activation("post")[:15, 0, 13, :].flatten().float() - base_sample_cache.stack_activation("post")[:15, 0, 13, :].flatten().float()).flatten()
neuron_df["diff_pre"] = to_numpy(chat_sample_cache.stack_activation("pre")[:15, 0, 13, :].flatten().float() - base_sample_cache.stack_activation("pre")[:15, 0, 13, :].flatten().float()).flatten()
neuron_df["diff_pre_linear"] = to_numpy(chat_sample_cache.stack_activation("pre_linear")[:15, 0, 13, :].flatten().float() - base_sample_cache.stack_activation("pre_linear")[:15, 0, 13, :].flatten().float()).flatten()

# %%
nutils.show_df(neuron_df.sort_values("diff", ascending=True).head(25))
# nutils.show_df(neuron_df.sort_values("diff", ascending=True).head(25))
# %%
px.histogram(neuron_df, x="diff_pre").show()
px.histogram(neuron_df, x="diff_pre_linear").show()
# %%
