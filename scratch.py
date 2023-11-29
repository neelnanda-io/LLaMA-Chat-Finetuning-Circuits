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

base_hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16)
chat_hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16)
# %%
cfg = loading.get_pretrained_model_config("llama-2-7b", torch_type=torch.float16, layer_norm_eps=1e-5)
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
chat_cfg = loading.get_pretrained_model_config("llama-2-7b", torch_type=torch.float16)
chat_model = HookedTransformer(chat_cfg, tokenizer=tokenizer)
chat_state_dict = loading.get_pretrained_state_dict("llama-2-7b", chat_cfg, chat_hf_model)
chat_model.load_state_dict(chat_state_dict, strict=False)
# %%
print(evals.sanity_check(base_model))
# %%
# for layer in range(n_layers):
#     model.blocks[layer].attn.W_K[:] = model.blocks[layer].attn.W_K * model.blocks[layer].ln1.w[None, :, None]
#     model.blocks[layer].attn.W_Q[:] = model.blocks[layer].attn.W_Q * model.blocks[layer].ln1.w[None, :, None]
#     model.blocks[layer].attn.W_V[:] = model.blocks[layer].attn.W_V * model.blocks[layer].ln1.w[None, :, None]
#     model.blocks[layer].ln1.w[:] = torch.ones_like(model.blocks[layer].ln1.w)
#     model.blocks[layer].mlp.W_in[:] = model.blocks[layer].mlp.W_in * model.blocks[layer].ln2.w[:, None]
#     model.blocks[layer].mlp.W_gate[:] = model.blocks[layer].mlp.W_gate * model.blocks[layer].ln2.w[:, None]
#     model.blocks[layer].ln2.w[:] = torch.ones_like(model.blocks[layer].ln2.w)
    
#     model.blocks[layer].mlp.b_out[:] = model.blocks[layer].mlp.b_out + model.blocks[layer].mlp.b_in @ model.blocks[layer].mlp.W_out
#     model.blocks[layer].mlp.b_in[:] = 0.

#     model.blocks[layer].attn.b_O[:] = model.blocks[layer].attn.b_O[:] + (model.blocks[layer].attn.b_V[:, :, None] * model.blocks[layer].attn.W_O).sum([0, 1])
#     model.blocks[layer].attn.b_V[:] = 0.

# model.unembed.W_U[:] = model.unembed.W_U * model.ln_final.w[:, None]
# model.unembed.W_U[:] = model.unembed.W_U - model.unembed.W_U.mean(-1, keepdim=True)
# model.ln_final.w[:] = torch.ones_like(model.ln_final.w)
# print(evals.sanity_check(model))
# model.generate("The capital of Germany is", max_new_tokens=20, temperature=0)
# %%
def decode_single_token(integer):
    # To recover whether the tokens begins with a space, we need to prepend a token to avoid weird start of string behaviour
    return tokenizer.decode([891, integer])[1:]
def to_str_tokens(tokens, prepend_bos=True):
    if isinstance(tokens, str):
        tokens = to_tokens(tokens)
    if isinstance(tokens, torch.Tensor):
        if len(tokens.shape)==2:
            assert tokens.shape[0]==1
            tokens = tokens[0]
        tokens = tokens.tolist()
    if prepend_bos:
        return [decode_single_token(token) for token in tokens]
    else:
        return [decode_single_token(token) for token in tokens[1:]]

def to_string(tokens):
    if isinstance(tokens, torch.Tensor):
        if len(tokens.shape)==2:
            assert tokens.shape[0]==1
            tokens = tokens[0]
        tokens = tokens.tolist()
    return tokenizer.decode([891]+tokens)[1:]
def to_tokens(string, prepend_bos=True):
    string = "|"+string
    # The first two elements are always [BOS (1), " |" (891)]
    tokens = tokenizer.encode(string)
    if prepend_bos:
        return torch.tensor(tokens[:1] + tokens[2:]).cuda()
    else:
        return torch.tensor(tokens[2:]).cuda()

def to_single_token(string):
    assert string[0]==" ", f"Expected string to start with space, got {string}"
    string = string[1:]
    tokens = tokenizer.encode(string)
    assert len(tokens)==2, f"Expected 2 tokens, got {len(tokens)}: {tokens}"
    return tokens[1]
print(to_str_tokens([270, 270]))
print(to_single_token(" basketball"))
ass_index = to_single_token(" ass")
scatter(base_model.W_E[ass_index], chat_model.W_E[ass_index])
scatter(base_model.W_E[ass_index], base_model.W_E[ass_index])
scatter(base_model.W_E[ass_index], torch.randn_like(base_model.W_E[ass_index]))
# %%
for k, v in base_model.named_parameters():
    print(k, v.norm())
# %%
vocab = to_str_tokens(np.arange(d_vocab))
W_E_diff = chat_model.W_E - base_model.W_E
W_E_diff.norm(dim=-1)
print("Chat model median W_E norm", chat_model.W_E.norm(dim=-1).median())
print("base model median W_E norm", base_model.W_E.norm(dim=-1).median())
line(W_E_diff.norm(dim=-1), x=[f"{s}/{i}" for i, s in enumerate(vocab)], title="Norm of embed difference")

W_E_cosine_sim = (chat_model.W_E * base_model.W_E).sum(-1) / chat_model.W_E.norm(dim=-1) / base_model.W_E.norm(dim=-1)
line(W_E_cosine_sim, x=[f"{s}/{i}" for i, s in enumerate(vocab)], title="cosine sim")
# px.histogram(to_numpy(W_E_cosine_sim), marginal="rug", hover_name=vocab, hover_text=np.arange(d_vocab))
# %%
px.histogram(to_numpy(W_E_cosine_sim), marginal="rug", hover_name=[f"{s}/{i}" for i, s in enumerate(vocab)])
# %%
vocab_df = pd.DataFrame({"vocab": vocab, "sim": to_numpy(W_E_cosine_sim), "base_norm":to_numpy(base_model.W_E.norm(dim=-1)), "base_norm_U": to_numpy(base_model.W_U.norm(dim=0)), "chat_norm":to_numpy(chat_model.W_E.norm(dim=-1)), "chat_norm_U": to_numpy(chat_model.W_U.norm(dim=0))})
vocab_df.sort_values("sim").head(30)
# %%
vocab_df.sort_values("base_norm").head(50)
# %%
vocab_df
# %%
vocab_df.query("base_norm>0.2").sort_values("sim").head(50)
# px.histogram(vocab_df, x="base_norm")
# %%
W_U_cosine_sim = (chat_model.W_U.T * base_model.W_U.T).sum(-1) / chat_model.W_U.T.norm(dim=-1) / base_model.W_U.T.norm(dim=-1)
vocab_df["sim_U"] = to_numpy(W_U_cosine_sim)
px.histogram(vocab_df, x="sim_U", hover_name="vocab", marginal="box").show()
vocab_df.sort_values("sim_U").head(50)
# %%
px.scatter(vocab_df.query("base_norm>0.2"), x="sim", y="sim_U", hover_name="vocab", opacity=0.3)
# %%
records = []
for (k, v), (k2, v2) in zip(base_model.named_parameters(), chat_model.named_parameters()):
    if "b_" in k:
        continue
    if k=="unembed.W_U":
        layer = 32
        sublayer = "other"
    elif k=="embed.W_E":
        layer = -1
        sublayer = "other"
    elif k=="ln_final.w":
        layer = 32
        sublayer = "other"
    else:
        layer = int(k.split(".")[1])
        sublayer = k.split(".")[2]
    weight_name = k.split(".")[-1]
    records.append({
        "full_name": k,
        "name": weight_name,
        "sublayer": sublayer,
        "layer": layer,
        "base_norm": v.norm().item(),
        "chat_norm": v2.norm().item(),
        "diff_norm": (v-v2).norm().item(),
        "dot": (v*v2).sum().item(),
    })
param_df = pd.DataFrame(records)
param_df["cos_sim"] = param_df["dot"] / param_df["base_norm"] / param_df["chat_norm"]
param_df["diff_norm_div"] = param_df["diff_norm"] / np.sqrt(param_df["base_norm"]*param_df["chat_norm"])
display(param_df.sort_values("cos_sim").head(20))
display(param_df.sort_values("cos_sim").tail(20))
# %%
px.box(param_df, x="sublayer", y="cos_sim").show()
px.box(param_df, x="layer", y="cos_sim").show()
px.box(param_df, x="name", y="cos_sim").show()
px.box(param_df, x="sublayer", y="diff_norm").show()
px.box(param_df, x="layer", y="diff_norm").show()
px.box(param_df, x="name", y="diff_norm").show()
px.box(param_df, x="sublayer", y="diff_norm_div").show()
px.box(param_df, x="layer", y="diff_norm_div").show()
px.box(param_df, x="name", y="diff_norm_div").show()
# %%
param_df.sort_values("diff_norm_div", ascending=False).head(20)
# %%
v_base = base_model.W_O[0].flatten()
v_chat = chat_model.W_O[0].flatten()
print(nutils.cos(v_base, v_chat))
print(v_base.norm())
print(v_chat.norm())
print((v_base - v_chat).norm())
print((v_base * v_chat).sum() / v_base.norm() / v_chat.norm())
# %%
print((v_base  - v_chat).norm())
print((v_base * (v_chat.norm() / v_base.norm()) - v_chat).norm())
# %%
data = load_dataset("NeelNanda/pile-10k", split='train')
tokenized_data = utils.tokenize_and_concatenate(data, tokenizer, max_length=256)
tokenized_data = tokenized_data.shuffle(44)
# %%
def run_franken_model(tokens, layer, model1=base_model, model2=chat_model):
    residual = model1.embed(tokens)
    for l in range(layer):
        residual = model1.blocks[l](residual)

    for l in range(layer, n_layers):
        residual = model2.blocks[l](residual)
    franken_logits = model2.unembed(model2.ln_final(residual))
    return franken_logits
# %%
batch_size = 16
layer = 28
tokens = tokenized_data[:batch_size]["tokens"]

model1 = base_model
model2 = chat_model

franken_logits = run_franken_model(tokens, layer)

base_logits = base_model(tokens)
chat_logits = chat_model(tokens)

print("Base loss", base_model.loss_fn(base_logits, tokens))
print("chat loss", chat_model.loss_fn(chat_logits, tokens))
print("franken loss", base_model.loss_fn(franken_logits, tokens))
# %%
base_clp = base_model.loss_fn(base_logits, tokens, per_token=True)
chat_clp = base_model.loss_fn(chat_logits, tokens, per_token=True)
franken_clp = base_model.loss_fn(franken_logits, tokens, per_token=True)

# %%
# old_str_tokens = base_model.to_str_tokens
# base_model.to_str_tokens = to_str_tokens
# nutils.make_token_df(tokens, model=base_model)
# %%
token_df = nutils.make_token_df(tokens, model=base_model, len_prefix=10, len_suffix=3)
token_df = token_df[token_df.pos < token_df.pos.max()]
token_df["base_clp"] = to_numpy(base_clp.flatten())
token_df["chat_clp"] = to_numpy(chat_clp.flatten())
token_df["franken_clp"] = to_numpy(franken_clp.flatten())
token_df["base_chat_diff"] = token_df["base_clp"] - token_df["chat_clp"]
token_df.sort_values("base_chat_diff", ascending=False).head(20)
# %%
px.scatter(token_df, x="base_clp", y="chat_clp", color="franken_clp", color_continuous_scale='Portland', hover_name="context")
# %%
temp_logits = chat_model("-2 Luton Town 13-2")
temp_log_probs = temp_logits.log_softmax(-1)[0, -1]
temp_df = nutils.create_vocab_df(temp_log_probs, model=base_model, make_probs=True)
temp_df.loc[to_single_token(" Draw")]
# %%
to_filter = (token_df["base_clp"]<12.) & (token_df["chat_clp"]<12.).values
new_loss_b2c = []
new_loss_c2b = []
for layer in tqdm.trange(0, n_layers+1):
    new_franken_logits_b2c = run_franken_model(tokens, layer)
    new_franken_logits_c2b = run_franken_model(tokens, layer, chat_model, base_model)
    new_loss_b2c.append(to_numpy(base_model.loss_fn(new_franken_logits_b2c, tokens, per_token=True).flatten())[to_filter].mean().item())
    new_loss_c2b.append(to_numpy(base_model.loss_fn(new_franken_logits_c2b, tokens, per_token=True).flatten())[to_filter].mean().item())
line([new_loss_b2c, new_loss_c2b])
# %%

fig = line([new_loss_b2c, new_loss_c2b], line_labels=["Base then Chat", "Chat then Base"], return_fig=True, xaxis="Layer", yaxis="Loss", title="Loss of Franken Model by Layer at Which We Switch")
fig = fig.add_hline(line_dash="dash", y=token_df["base_clp"][to_filter].mean())
fig = fig.add_hline(line_dash="dash", y=token_df["chat_clp"][to_filter].mean())
fig.show()
# %%
new_tokens = torch.tensor([1, 
#  518,
#  25580,
#  29962,
 5328,
 437,
 306,
 1207,
 263,
 to_single_token(" bomb"),
 29973,
#  518,
#  29914,
#  25580,
#  29962
 ])[None, :].cuda()

base_t = base_model.generate(new_tokens)
print(to_string(base_t))
chat_t = chat_model.generate(new_tokens)
print(to_string(chat_t))
# %%
for i in range(20):
    base_t = base_model.generate(new_tokens, verbose=False)
    print(to_string(base_t))
# %%
print(chat_t)
print(to_str_tokens(chat_t))
# %%
new_tokens = to_tokens("[INST]How do I make a bomb?[/INST]  I cannot fulfill your request")[None]
# new_tokens = torch.tensor([[    1,  5328,   437,   306,  1207,   263, 13585, 29973,    13,    13,
#          29902,  2609,  3867, 11994,   373,   920,   304,  1207]]).cuda()
base_logits, base_cache = base_model.run_with_cache(new_tokens)
chat_logits, chat_cache = chat_model.run_with_cache(new_tokens)
base_log_probs = base_logits.log_softmax(-1)
chat_log_probs = chat_logits.log_softmax(-1)

records = []
for i in range(len(new_tokens[0])-1):
    record = {"curr": to_string([new_tokens[0, i].item()]),
              "next": to_string([new_tokens[0, i+1].item()]),
              "base_lp": base_log_probs[0, i, new_tokens[0, i+1]].item(),
              "base_prob": base_log_probs[0, i, new_tokens[0, i+1]].exp().item(),
              "base_rank": (base_log_probs[0, i]>base_log_probs[0, i, new_tokens[0, i+1]]).sum().item(),
              "chat_lp": chat_log_probs[0, i, new_tokens[0, i+1]].item(),
              "chat_prob": chat_log_probs[0, i, new_tokens[0, i+1]].exp().item(),
              "chat_rank": (chat_log_probs[0, i]>chat_log_probs[0, i, new_tokens[0, i+1]]).sum().item(),
              }
    records.append(record)
    # print("Curr:", to_string([new_tokens[0, i].item()]), "Next:", to_string([new_tokens[0, i+1].item()]))
    # print("base",
    #     base_log_probs[0, i, new_tokens[0, i+1]], 
    #       base_log_probs[0, i, new_tokens[0, i+1]].exp(), 
    #       (base_log_probs[0, i]>base_log_probs[0, i, new_tokens[0, i+1]]).sum())
    # print("chat",
    #       chat_log_probs[0, i, new_tokens[0, i+1]], 
    #       chat_log_probs[0, i, new_tokens[0, i+1]].exp(), 
    #       (chat_log_probs[0, i]>chat_log_probs[0, i, new_tokens[0, i+1]]).sum())
df = pd.DataFrame(records)
nutils.show_df(df)

# %%
for i in range(10, 14):
    print(to_string([new_tokens[0, i]]))
    nutils.show_df(nutils.create_vocab_df(chat_logits[0, i], True, model=chat_model).head(20))
# %%
new_tokens_2 = torch.tensor([1, 
 518,
 25580,
 29962,
 5328,
 437,
 306,
 1207,
 263,
 to_single_token(" bomb"),
 29973,
 518,
 29914,
 25580,
 29962
 ])[None, :].cuda()
for i in range(30):
    print("--------------------------")
    print(to_string(chat_model.generate(new_tokens_2, verbose=False)))
# %%
new_tokens = to_tokens("[INST]How do I make a bomb?[/INST]  I")[None]
CANNOT = 2609
# new_tokens = torch.tensor([[    1,  5328,   437,   306,  1207,   263, 13585, 29973,    13,    13,
#          29902,  2609,  3867, 11994,   373,   920,   304,  1207]]).cuda()
base_logits, base_cache = base_model.run_with_cache(new_tokens)
chat_logits, chat_cache = chat_model.run_with_cache(new_tokens)
base_log_probs = base_logits.log_softmax(-1)
chat_log_probs = chat_logits.log_softmax(-1)
BASE_BASELINE = base_log_probs[0, -1, CANNOT]
CHAT_BASELINE = chat_log_probs[0, -1, CANNOT]
def patching_metric(logits):
    if len(logits.shape)==3:
        logits = logits[0, -1]
    return (logits.log_softmax(-1)[CANNOT] - BASE_BASELINE) / (CHAT_BASELINE - BASE_BASELINE)
print(patching_metric(base_logits), patching_metric(chat_logits))
# %%
for i in range(len(new_tokens[0])):
    print(i, to_string([new_tokens[0, i].item()]))
# %%
patching_metric_franken_b2c = []
patching_metric_franken_c2b = []
for i in tqdm.trange(33):
    franken_logits = run_franken_model(new_tokens, i)
    patching_metric_franken_b2c.append(patching_metric(franken_logits).item())
    franken_logits = run_franken_model(new_tokens, i, chat_model, base_model)
    patching_metric_franken_c2b.append(patching_metric(franken_logits).item())
line([patching_metric_franken_b2c, patching_metric_franken_c2b], line_labels=["Base then Chat", "Chat then Base"])

# %%
def patch_resid_post(resid_post, hook, pos, new_resid_post):
    resid_post[:, pos, :] = new_resid_post
    return resid_post
records = []
num_tokens = len(new_tokens[0])
for i in tqdm.trange(n_layers):
    for pos in range(9, num_tokens):
        logits = base_model.run_with_hooks(new_tokens, fwd_hooks=[(utils.get_act_name("resid_post", i), partial(patch_resid_post, new_resid_post=chat_cache["resid_post", i][:, pos, :], pos=pos))])
        record = {"layer": i, "pos": pos, "metric": patching_metric(logits).item()}
        records.append(record)
resid_post_df = pd.DataFrame(records)
imshow(resid_post_df.pivot(columns="layer", index="pos", values="metric").values)
# px.imshow(resid_post_df, x="pos", y="layer", "metric")

# %%
x = resid_post_df.pivot(columns="layer", index="pos", values="metric").values
line([patching_metric_franken_c2b[:-1], x[-1]])
# %%
imshow(resid_post_df.pivot(columns="layer", index="pos", values="metric").values, y=to_str_tokens(new_tokens[0, 9:]))
# %%
# def temp_hook(resid_post, hook):
#     resid_post[:, :, :] = chat_cache["resid_post", 5]
#     return resid_post
# print(patching_metric(base_model.run_with_hooks(new_tokens, fwd_hooks=[(utils.get_act_name("resid_post", 5), temp_hook)])))
for i in range(17):
    def temp_hook(resid_post, hook):
        resid_post[:, :i, :] = chat_cache["resid_post", 5][:, :i, :]
        resid_post[:, i+1:, :] = chat_cache["resid_post", 5][:, i+1:, :]
        return resid_post
    print(i, to_string([new_tokens[0, i].item()]), patching_metric(base_model.run_with_hooks(new_tokens, fwd_hooks=[(utils.get_act_name("resid_post", 5), temp_hook)])))
# %%
def temp_hook(resid_post, hook):
    for i in [3, 14, 16]:
        resid_post[:, i, :] = chat_cache["resid_post", 5][:, i, :]
    # resid_post[:, :, :] = chat_cache["resid_post", 5][:, :, :]
    # resid_post[:, 14, :] = chat_cache["resid_post", 5][:, 14, :]
    return resid_post
print(patching_metric(base_model.run_with_hooks(new_tokens, fwd_hooks=[(utils.get_act_name("resid_post", 5), temp_hook)])))
# %%
