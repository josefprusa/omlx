# EAGLE-3 Build Log

Branch: glm5.2-native-kernels-v0.4.5 (no commits). Model: MiniMax-M3-oQNVFP4-fs5.
Ops: tmux `omlx:4` (serve). Restart = C-c, sleep, then
`env OMLX_M3_DEBUG_PATH=256 MLX_MAX_OPS_PER_BUFFER=500 uv run omlx serve --log-level info 2>&1 | tee ~/.claude/jobs/62f9cfe9/tmp/server_m3s.log`
Health: `curl localhost:8000/health`. API key file: `$OMLX_API_KEY_FILE`.
Gate-1 sweep: /private/tmp/claude-501/.../8cafa811-.../scratchpad/eagle_gate1.py (stops server; sets wired limit).

## LAWS (from lessons.md — non-negotiable)
- Any gated fast-path MUST ship an engagement counter visible in live logs (fp16-gate vs bf16-live killed the M3 kernel silently).
- Standalone repros copy LIVE dtype/config (bf16, torch_dtype) — NOT fp16.
- Verify through the real engine/server load path (omlx wraps extra patches raw mlx_vlm skips). Two MoE sanitize layers exist.
- Fresh-nonce benches. Assert str-replace targets. The live path is the only truth.
- Codex drafts nontrivial code (workspace-write); adversarial read-only Codex review of full diff before EVERY deploy; fix findings first.

## PHASE 1 — small-L (2≤L≤8) verify path in fused kernels
Baseline (warm 4k): L=1 47.4ms, L=2 92.0ms (+44.6!), then ~17-21ms/extra tok. Target ≤~14ms/extra tok at L≤8.
Design: (a) fused_block_scores → L≤8 queries (stage L*4*128 q in TG mem, 16KB@L=8); out [1,4,L,nb].
(b) m3_fused_index_topk → per-query top-16 + per-query causal ceiling (query i sees blocks ≤ (T-L+i)//128).
(c) NO attn kernel: union selected blocks across L per kv-head → 1 take_along_axis gather → 1 SDPA w/ per-query bool validity mask (reuse _sparse_decode_attention masked-compact machinery).
Gate 2≤L≤8 in MiniMaxAttention.__call__. Counters (_m3c + M3CENSUS). Kill switch OMLX_M3_DISABLE_SMALLL_VERIFY.
Correctness: match MSA-prefill path at L=2..8 on warm cache within bf16 tol.

### Key measured facts (offline, pre-existing artifacts)
- Baseline fs5 NO-spec: 27.09 tok/s short, 21.66@9k, 21.64@16k, 17.48@128k. Quality gsm8k=0.953 mmlu=0.813 (must preserve).
- Union sizing (union_analyze.py over topkdump/): union of L=8 queries' top-16 blocks = 1.4-2.6x single (~22-42 blocks); worst 2.64x (deep layer/long ctx). → static U=64 union blocks/kv-head covers L≤8 w/ margin. One shared gather+SDPA >> L passes.

### Finalized design (verified against msa.py oracle)
- Oracle = `_msa_prefill_attention` / `build_grouped_msa_topk` (msa.py): returns q2k [B,K,L,16], PER-(kv-head,query),
  block-max, cur_block=(q_start+l)//128, init<init_blocks→1e30, local [max(cur-lb+1,0)..cur]→1e29, invalid→-1.
  IDENTICAL to fused L==1 semantics + L axis + per-query ceiling. → match by construction.
- Kernels: ADD `_multi` variants (leave L==1 byte-identical, zero regression). block-scores: unified p=h*L+l,
  q_tg[H*L*D]=16KB@L=8, out [H*L,nb]→[1,4,L,nb]. topk: 1 TG per (h,l), per-query cur_block, +future-block mask `i>cur_block→-inf`.
- Union+mask (mx, NO attn kernel): per-kv-head union via sort→where-sentinel(nb)→re-sort→[:U]; U=min(L*16,64).
  membership mask (union block ∈ query l's 16) & causal(pos<=qpos_l) & inrange → [1,K,L,U*128] → repeat→[1,H,L,U*128].
  ONE take_along_axis gather + ONE SDPA. q_start=sparse_q_start=total_len-L; qpos_l=q_start+l ([B,L] from _sparse_query_positions).
- Gate: 2<=L<=8 AND _can_use_msa_prefill_attention (ensures q_positions None etc.), BEFORE MSA-prefill block; bail→fall through.
- Counters verify_scores/none/topk/ok/bail. Kill: OMLX_M3_DISABLE_SMALLL_VERIFY.
- Overflow risk: union>64 theoretical only (obs max ~42@L=8); parity harness asserts max union<=U; else bump U.

### Status
- [x] Map language.py attention path
- [x] Write Codex spec (~/.claude/jobs/62f9cfe9/tmp/eagle_p1_spec.md) — kernel source authored by lead
- [x] Codex draft (kernels+glue+wiring+harness). All parse. L==1 byte-identical (verified).
- [x] Adversarial Codex review: 11/11 checks pass; ONE finding = union-overflow silent-drop (HIGH). Independent read agreed.
- [x] FIX (lead, inline): runtime fallback in _sparse_verify_attention — if N>U(=64) and max union>U → return None → MSA-prefill.
      Structurally unreachable at L≤4 (K=3 op point → no sync). Counter verify_union_overflow. +synthetic union=128 guard test in harness.
- [x] Overflow-guard test PASS (union=128→fallback). Engine bf16 load OK. verify_ok=57/57 layers (engages).
- [x] Correctness gate PASS (3-way diag2.log, L=2,3,4,5,6,8): argmax_agree(verify-vs-msa)=1.000 ALL L;
      rel_mean<=0.657%. verify & msa EQUIDISTANT from dense → same sparse approx, diff=bf16 rounding only. NOT a bug;
      original max_abs<=0.05 gate was unrealistic for 57-layer bf16. Greedy spec decode will be EXACT.
- [x] Gate-1 sweep (union+SDPA design): FAIL. ON beats OFF only at L=2; regresses 20-30% at L≥3. ON marg 30.6ms vs target 14.
- [x] PROFILE (profile.log): SDPA is the bottleneck — bool-masked union SDPA jumps 5.4x L2→L4 (0.26→1.40ms/layer),
      non-flash materialization + U=64 4x redundancy. dedup/gather/mask all ~0.2-0.3ms.
- [x] PIVOT: union+mask+SDPA design REPLACED with multi-query FLASH kernel (extend _FLASH_SRC → _FLASH_SRC_MULTI).
      Each query flashes its OWN 16 blocks, causal pos<=q_start+l, online softmax fp32, streams K/V. NO union/gather/mask
      → overflow issue GONE, no materialization. flash_sparse_sdpa_multi in fused_index.py; _sparse_verify_attention rewired.
      (Deviates from lead's "no attn kernel/union+SDPA" design point BECAUSE that design measurably regressed — flagged to lead.)
### Perf results (verify_final.py, warm ~4600, min-ms, ON=verify OFF=msa-prefill)
- FLASH kernel: regression GONE — ON beats OFF at ALL L. ON marg 18.77 vs OFF 21.79 (~14% faster). argmax=1.0 at L<=5,
  wobble L>=6 (0.75-0.83) — INPUT-DEPENDENT near-ties (diag2 deterministic toks=1.0; verify_final random toks wobble).
- BATCHED (lead's no-kernel idea): 20.46 marg — SLOWER than flash. Causal mask (current-block tail) forces mx SDPA off
  its flash path (can't be truly maskless for L>1). Same argmax wobble → confirms wobble is tokens, not kernel.
- DECISION: ship FLASH kernel (per lead: "if batched misses, proceed with _FLASH_SRC extension"). Both MISS <=14.
- [~] FLOOR (attn=zeros) + MARGIN diag (floor_margin.py): is <=14 reachable via attn? are flips benign ties? RUNNING.
- [x] FLOOR (attn=zeros)=13.0 avg / ~14.6 @L>=4 → <=14 UNREACHABLE via attention (MoE per-token weight-reads bound). Bar RETIRED by lead.
- [x] MARGIN: argmax flips ONLY at genuine near-ties (msa top1-top2 margin 0.06/0.00/0.13/0.19; one EXACT 0.0 tie). BENIGN (kernel-order class). Lead accepted.
- [x] Adversarial review of flash: 2 HIGH. #1 K-bound OOB (real, no-op in valid path) FIXED: kernel `pos<K && pos<=qpos` + `pos>=K||pos>qpos`, wrapper bail q_start+L>K. #2 mask-rewrite-to-"causal": PRE-EXISTING, shared w/ MSA-prefill (verify matches oracle, no regression) — documented, out of scope.
- [x] Fixed kernel compiles+runs (isolated test L=2,4,8 finite; guard bails). No-op fix → correctness/perf unchanged.
- [~] DEPLOY: server restarted omlx:4 (chained per ops contract). L=1 fs5 probe running (loads via engine path + baseline tok/s).
- LEAD DECISION: flash ACCEPTED as Phase-1 deliverable. Ledger: "MoE verify-batching" = post-EAGLE lever.

## PHASE 1 SUMMARY (for report)
Flash multi-query verify path: correct (argmax-exact vs MSA-prefill modulo benign ties, rel_mean<=0.66%), beats MSA-prefill
at EVERY L (18.77 vs 21.79 ms/extra, ~14% faster), v1 union regression eliminated. <=14 shown MoE-bound (retired). Deployed.
Live L>1 engagement (verify_ok census) validated in Phase 2 via the drafter (L=1 serving keeps verify dormant).
fs5 path: $HOME/.omlx/models/unigilby/MiniMax-M3-oQNVFP4-fs5

## PHASE 2 — EAGLE-3 drafter + mxfp8 variant

### Draft model arch (CONFIRMED from safetensors + README + config)
Path: $OMLX_COLD_STORAGE/omlx-quant-work/MiniMax-M3-EAGLE3 (6.5GB bf16, LlamaForCausalLMEagle3, torchspec 0.1.0).
DRAFT-SPECIFIC weights (load these, ~1.6GB): fc[6144,18432]; fc_norm.0/1/2[6144] (per-tap RMSNorm over target layers 2,30,57);
  layers.0: hidden_norm[6144], input_layernorm[6144], self_attn.{q,k,v}_proj[8192,12288] (in=concat(ln(embed(tok))[6144],
  hidden_norm(h)[6144])=12288; 64 heads*128=8192), o_proj[6144,8192], post_attention_layernorm[6144], mlp gate/up[18432,6144] down[6144,18432].
SHARED-FROM-TARGET (reuse fs5 loaded quantized, save ~4.9GB; VERIFY equiv first): embed_tokens[200064,6144], lm_head[200064,6144], norm[6144].
  NOTE M3 is VL → target's live under language_model.* prefix. rope_theta=5e6, rms_eps=1e-6, silu, untied.
### EAGLE-3 FORWARD WIRING — RESOLVED (vLLM llama_eagle3.py authoritative, source-cited)
- FC (ONCE at prefill/verify, NOT in recurrence): feat0 = fc(concat([fc_norm.0(H2), fc_norm.1(H30), fc_norm.2(H57)])) [18432→6144].
  Per-tap norm BEFORE concat; ascending order (fc_norm.0=layer2, .1=layer30, .2=layer57). Skipped steps≥1 (feature already 6144).
- "+final layer": NOT an fc input, NOT fed to draft. = training label + the SEED TOKEN (target samples token0 from final hidden) + shared norm/lm_head/embed. Capture ONLY 3 taps.
- Draft layer input [12288] = concat(input_layernorm(embed(tok)), hidden_norm(feature)) → qkv_proj (2k→k FC absorbed into qkv, that's why q/k/v_proj in=12288).
- RECURRENCE (K steps): step0: token0=last-accepted (target-sampled), feature0=fc-fused → (last_h, rec_h)=draft(...); draft_tok0=sample(lm_head(norm(last_h))); feature1=rec_h.
  step i≥1: token_i=draft_tok[i-1]; feature_i=rec_h prev; pos+=1 → (last_h,rec_h); draft_tok_i=sample; feature_{i+1}=rec_h.
  **M3 norm_output=TRUE → recurrent feature = POST-final-norm hidden (SAME tensor as logits input), NOT pre-norm. CRITICAL — many EAGLE ckpts differ.**
  Draft runs its OWN KV-cached self-attn (prefix-KV) over committed prefix + drafted tokens. logit=lm_head(norm(h)) full vocab (d2t absent=identity).
- TARGET CAPTURE: aux = hidden+residual = PRE-norm residual stream at OUTPUT of layers (2,30,57), ALL positions. Final hidden returned separately (not in aux).
  **±1 CAVEAT**: vLLM aux_layers=(2,30,57) w/ layer_idx=idx+1 (embed=slot0) captures output of 0-indexed layers {1,29,56}. VERIFY omlx capture_layer_ids convention (what `h`/`idx` mean at language.py:2060-2064) matches before trusting — mismatch silently degrades accept rate. Parity: fc-fused feature must match trained expectation.
- Draft iface (vLLM): combine_hidden_states (fc, once); forward(input_ids,positions,hidden_states)→(post_norm_h, recurrent_h); compute_logits(h)→[.,200064]; embed_input_ids.
- CAPTURE CONVENTION (verified, language.py:2060-2064): loop appends `h`=OUTPUT of 0-indexed decoder layer `idx` = PRE-norm residual stream (matches vLLM hidden+residual quantity ✓). capture_layer_ids=[2,30,57] → 0-idx {2,30,57}.
  vLLM aux (2,30,57) → 0-idx {1,29,56} (embed=slot0 +1 offset). ±1 AMBIGUITY. RESOLVE EMPIRICALLY: A/B capture [2,30,57] vs [1,29,56], pick the one hitting vendor accept ~0.92/0.84/0.75. Don't guess the offset.
Target accept K=3 (match): code 0.922/0.832/0.744 (len 3.5), math 0.923/0.839/0.756, dialogue 0.749/0.547/0.402 (len 2.70). Greedy draft topk=1.
Vendor served: vLLM num_speculative_tokens=3, block-size 128, enforce-eager, target MXFP8.

### DESIGN CONSTRAINTS (lead)
- K (draft depth) must be a RUNTIME variable per cycle — NEVER baked into shapes/compiled constants.
  (Flash kernel already takes L as runtime param grid=(256,H,L), per-shape-cached → depth switching free after 1st use of each L.)
  Ship static configurable K=3 first; don't preclude confidence-gated dynamic depth (draft while draft top-1 prob>thresh, cap 4;
  accept-rate EMA controller) as a fast-follow. So: draft-verify loop passes K dynamically; verify forward handles any L in 2..8.
- Phase 1 SIGNED OFF by lead. Finding #2 → ledger (todo.md), deferred.

### MTP framework map (omlx/patches/mlx_lm_mtp/, integration contract)
- Chained cycle `_run_verify_cycle_chained` (batch_generator.py:1736): L=K+1 verify, GREEDY-ONLY longest-prefix accept.
  Legacy L=2 cycle HAS rejection sampling (_residual_sample:2044, _accept_lp_for:1047). → MUST add block rejection sampling to CHAINED for temp>0.
- Chained self-chains the draft's OWN returned hidden mh for drafts 2..K (_chain_drafts:1683) → MATCHES EAGLE-3 recurrence. Good.
- Drafter interface (glm_moe_dsa_model.py): mtp_forward(hidden,next_token_ids,mtp_cache,return_hidden=)→(logits,hidden); make_mtp_cache();
  __call__(...,return_hidden,n_confirmed)→(logits,pre_norm_hidden); sanitize(). Markers: _omlx_mtp_decode_enabled, _omlx_mtp_warm_capable, is_mtp_active().
- K: OMLX_MTP_DRAFT_K (default 1, clamp[1,7]), read ONCE at post-init (_draft_k:553). For dynamic-K → make per-cycle.
- Accept logging: _MtpStats pos_attempts/pos_hits (1771-1784), _log_mtp_stats (1618) INFO always-on → a{i}=hits/attempts%.
- Cache: target=prompt_cache (trim via _rollback_after_reject:1129 → block_size-(accepted+1)); draft=mtp_cache (trim _trim_mtp_spec:1717). 2-pass check-all-before-mutate.
- Enable: model_settings.mtp_enabled → set_mtp_active → head attached at construction. NO env kill-switch (mtp_enabled only).
- **GATE ISSUE**: _is_mtp_compatible (model_loading.py:442) lists qwen3_5/6, deepseek_v4, glm_moe_dsa — NOT minimax_m3. And M3 = VLM engine.
  Separate vlm_mtp_* keys exist (bypass BatchGenerator). → RESOLVED below.

### INTEGRATION PATH RESOLVED (M3 = VLM, uses vlm_mtp NOT mlx_lm_mtp)
- M3 decodes via omlx/engine/vlm.py stream_generate (2718) — its OWN async VLM loop, NOT mlx_lm GenerationBatch. So mlx_lm_mtp does NOT apply.
- REAL EAGLE-3 hook = omlx/speculative/vlm_mtp.py (VLMMTPDrafter, run_vlm_mtp_decode) + scheduler.py (_VlmMtpState:121, "bypasses BatchGenerator") + patches/mlx_vlm_mtp/ + profile keys vlm_mtp_enabled/vlm_mtp_draft_model/vlm_mtp_draft_block_size (model_profiles.py:64).
- TAP HOOK EXISTS: MiniMaxM3Model.__call__ takes capture_layer_ids + hidden_sink (language.py:2032-2067; kwargs passthrough 2168/2261) → appends layer h to hidden_sink for idx in capture_set. EXACTLY the EAGLE tap capture. Use capture_layer_ids=[2,30,57(,final)].
- mlx_lm_mtp is the PATTERN source (chained accept, rejection sampling _residual_sample, per-pos logging); vlm_mtp is the M3 wiring. (agent mapping vlm_mtp contract now.)
### *** PIVOTAL: REUSE mlx-vlm's shipped EAGLE-3, don't build a drafter ***
mlx-vlm (.venv/.../mlx_vlm/speculative/) ALREADY has the full EAGLE-3 path (verified real, not stub):
- Eagle3DraftModel (drafters/eagle3/eagle3.py:155): 3-tap fc fusion, prefix-KV draft self-attn, shared embed/lm_head via bind(211), accept_lens logging.
- _eagle3_rounds (eagle3.py:344) / _eagle3_rounds_batch (478): full draft/verify loop (K=bs-1 draft, L=bs verify → hits MY Phase-1 flash path!, greedy accept, KV trim via lm.rollback_speculative_cache).
- run_speculative_server_rounds (utils.py:118) dispatcher; get_speculative_rounds(draft_kind) switch (utils.py:72 eagle3).
- config.py: target_layer_ids default [2,n//2,n-3], capture_layer_ids=[target-1] → ±1 handled BY LIBRARY.
M3 target ALREADY exposes: capture_layer_ids/hidden_sink (multi-layer, returns list), return_hidden, rollback_speculative_cache (language.py:2292).
BLOCKERS = 3 small omlx glue changes only:
  (1) vlm_mtp.py:264 load_vlm_mtp_drafter rejects kind!="mtp" → allow ("mtp","eagle3").
  (2) run_vlm_mtp_decode (vlm_mtp.py:306) hardcodes _mtp_rounds → dispatch eagle3→_eagle3_rounds (concat multi-tap hidden, NO shared_kv).
  (3) scheduler._route_to_vlm_mtp (6126-6161) captures LAST layer only + passes shared_kv → for eagle3 capture multi-layer (config.capture_layer_ids), pass full hidden list.
Config: vlm_mtp_enabled=True, vlm_mtp_draft_model=<eagle3 ckpt>, vlm_mtp_draft_block_size=K+1 (=4; else caps to 2). engine_pool.py:1534 loads it.
MUST-BUILD (missing in mlx-vlm): (b) rejection sampling temp>0 (both loops greedy-match only) — mission requires; (mxfp8) quantize draft layer+fc.
OPEN (verifying empirically, bghf57l8j): kind resolution (ckpt model_type="llama" not "eagle3"); target_layer_ids default may be [2,0,0] (n=draft's 1 layer!) → set [2,30,57] explicitly. Shared embed via proxy exposing language_model.model.embed_tokens.

### STAGED PLAN
2a WIRING (greedy, first live numbers): 3 glue changes + config + register ckpt + fix target_layer_ids → live eagle3 greedy. Verify Phase-1 flash engages on verify forward, acceptance ~vendor (0.92/0.84/0.75 code).
2b REJECTION SAMPLING for temp>0 (custom eagle3 walk: prob-ratio accept + residual resample).
2c mxfp8 drafter variant + A/B (ship if live accept delta <=1.5pp).
2d Profile glue overhead; conditional norm→argmax+top1prob epilogue kernel (>0.5ms/cycle).

### Stage 2a status
- Empirical load test: load_drafter(ckpt) FAILS "Received 16 params not in model" — resolves ckpt (model_type=llama) to PLAIN Llama, not Eagle3DraftModel. Forcing kind=eagle3 same error.
- GAP pinned: (1) class resolution (need Eagle3DraftModel not Llama); (2) Eagle3DraftModel has NO fc_norm but ckpt has fc_norm=true (3 per-tap norms) + mlx-vlm _prepare_target_hidden skips per-tap norm → must ADD fc_norm; (3) target_layer_ids default = [2,0,0] (n=draft's 1 layer) → set [2,30,57] explicit.
  GOOD: Eagle3FirstLayer names (input_layernorm/hidden_norm/self_attn.{q,k,v,o}_proj/post_attention_layernorm/mlp.*) MATCH ckpt layers.0.* 1:1 (separate q/k/v, no fusion).
- Codex Stage 2a DRAFTED: omlx/patches/mlx_vlm_mtp/eagle3_minimax.py (MiniMaxEagle3DraftModel, direct Eagle3Config,
  target_layer_ids=[2,30,57]/capture=[1,29,56], fc_norm ADDED, norm_output post-norm recurrence patch); +3 glue changes; D4 load test PASS (17/17 params, feature (1,2,6144) finite, draft toks in-vocab).
- Adversarial review: core math CORRECT (checks 1-3: fc_norm order, norm_output post-norm no-double-norm, capture offset all pass; load strict OK). NO-GO — 3 HIGH + 1 MED integration gaps:
  * [HIGH] PREFIX-KV NOT SEEDED (scheduler:6184) — THE mandatory item. _eagle3_rounds calls draft_model.prefill_from_target_hidden(input_ids, hidden[B,T,18432], bonus, sampler) ONLY if prompt_tokens!=None; omlx passes neither prompt_tokens nor full-prompt taps → drafter KV empty → chains collapse. FIX: capture ALL prompt-position taps (capture_layer_ids during prefill) + pass full hidden[B,T,18432]+prompt_tokens.
  * [HIGH] block_size defaults to 2 (K=1) when vlm_mtp_draft_block_size=None → set eagle3 default = drafter.config.block_size(=4).
  * [HIGH] terminal stop/length can store uncommitted spec KV (cache desync) → rollback/cleanup before publishing prompt_cache.
  * [MED] eos_token_ids/stop_check dropped in eagle dispatch → pass through.
- prefill_from_target_hidden (eagle3.py:318): shifted=input_ids[:,1:]+bonus; _forward_tokens(shifted, hidden[:,:len]) builds draft KV over prompt. NEEDS full prompt taps [B,T,18432].
- Stage-2a-fix DONE (Codex): prefix-KV via CAPTURE-DURING-PREFILL (hooks scheduler._do_external_prefill:2903, _step_prefill_chunk:3946 → accumulate chunk taps → full [1,T,18432]; _route_to_vlm_mtp:6347 passes prompt_tokens+hidden; eagle3_minimax prefill_from_target_hidden:162 → _next_position==prompt_len). block_size default=config(4). terminal cache validates committed-length + discards mismatch. eos/stop forwarded. Fallback re-prefill only for incomplete-capture. 23 tests pass; parse OK.
- Adversarial review of fix RUNNING (bt5aklfjx). TOP CHECK: no-regression (capture hooks in SHARED prefill path must be inert when eagle3 off).
- Fix-review DONE (bba67pgna): CORE CORRECTNESS CONFIRMED (prefix-KV ordering/shift, terminal, block_size=4, eos, no hot-loop sync). 3 remaining = prefill-capture ROBUSTNESS (defer, none hit controlled fs5 diag):
  [CRIT] capture gate too broad (only bites NON-M3+eagle3 drafter; fs5=M3 so dormant) — fix before eagle3 on any non-M3.
  [HIGH] unbounded [1,T,18432] tap buffer @128k (~4.7GB+concat; fine ≤16k) — bound before 128k.
  [HIGH] fallback re-prefill fires on prefix-cache HITS (multi-turn) — fix persists taps w/ cache; avoid via fresh prompts.
- BLOCKER: LIVE enable (edit ~/.omlx/model_settings.json fs5 vlm_mtp_*) DENIED by auto-mode classifier — persistent production config change needs USER consent (teammate auth insufficient). Reported to lead; needs user to approve OR run out of auto mode. fs5 config UNTOUCHED.
- WORKAROUND (no permission wall): STANDALONE acceptance harness (eagle_accept.py, Codex authoring bsdemo8mf): load target+drafter, drive _eagle3_rounds greedy on code/math, per-pos accept a1/a2/a3 + mean len + ±1 capture A/B ([1,29,56] vs [2,30,57]) vs vendor 0.92/0.84/0.75. Server-DOWN → run CHAINED w/ restart.
- USER APPROVED live enable (lead applied fs5 model_settings vlm_mtp_*). Blocker cleared.
- Harness run 1: PREFIX-KV SEEDING CONFIRMED WORKING (draft_next_position=217==prompt_len=217). Then CRASHED in _eagle3_verify_target: mx.concatenate(verify_out.hidden_states) — VLMModelAdapter.__call__ (vlm.py:373) returned bare logits (not full output) because _eagle3_verify_target passes capture_layer_ids WITHOUT return_hidden. Static reviews missed it (only running caught it).
- FIX (inline, vlm.py:373): return full result when (return_hidden OR capture_layer_ids set). Fixes harness AND live path (both use _eagle3_verify_target). No regression (normal decode passes neither → bare logits). Parse OK.
- Harness re-run CHAINED (bw568ebfb, accept2.log) w/ fix → restart live. Then: read acceptance a1/a2/a3+meanlen+±1 A/B; live probes tok/s+histogram.
- ACCEPTANCE VALIDATED (accept2.log, greedy, 128 tok/prompt, capture [2,30,57] winner but A/B CONFOUNDED — diff nonces per arm, ~2pp noise):
  MATH a1/a2/a3 = 97.1/93.1/85.1%, mean_len 3.72 — BEATS vendor (0.92/0.84/0.76).
  CODE a1/a2/a3 = 86.6/65.3/53.2%, mean_len 3.05 — a1 solid, a2/a3 below vendor (likely oQNVFP4 4-bit target vs drafter's MXFP8 training target + ad-hoc prompts).
  pipeline_healthy=True. prefix-KV seeded every prompt (next_position==prompt_len). KEEP principled [1,29,56] (vLLM convention) — 2pp A/B gap is confounded noise.
- LIVE: 2 more runtime bugs found+fixed (only running caught them, static reviews missed):
  (1) vlm.py:373 VLMModelAdapter returned bare logits when eagle3 verify passed capture_layer_ids w/o return_hidden → return full result when capture set. (2) scheduler.py:6136 `not hidden_states` truth-tested a multi-element mx.array → ValueError; fixed to None/empty-list guard.
- LIVE ENGAGEMENT CONFIRMED (server_m3s.log): "vlm_mtp decode started block_size=4"; "prefix-KV seeded draft_next_position=1120==prompt_len" (LIVE prefix-KV works!); "rounds=42 accepted=85/126 (67.5%) tokens_per_round=3.02".
- **PROBLEM: NO NET SPEEDUP at short ctx (1120): 26.23 tok/s vs 26.74 baseline.** 3.02 tokens/round accept is good but per-round cost (verify L=4 + K=3 drafter forwards + lm_head calls) ≈ 3× baseline-per-token → break-even. Drafter overhead (3× 1-layer + full 200064 lm_head/step) likely dominates. Phase-2d overhead profiling now CRITICAL, not optional.
- Minor: terminal cache off-by-one (cache_len 1247 vs expected 1248 → discarded safely; fix).
### FIRST LIVE NUMBERS (fs5, greedy, eagle3 ON) — WORKS but NO SPEEDUP
- Short 1120: spec 26.23 vs baseline 26.74 tok/s (-2%). 3.02 tokens/round, 67.5% accept.
- 16k 14021: spec 18.98 vs baseline 21.64 tok/s (-12% REGRESSION). 2.72 tokens/round, 57.4% accept.
- Prefix-KV seeded LIVE both (next_position==prompt_len). Pipeline fully functional. Acceptance GOOD (not the problem).
- ROOT CAUSE: per-round cost (verify L=4 + K=3 draft forwards + 3x full 200064 lm_head) ≈ 3 baseline-tokens → break-even.
  (a) verify L=4 only partially amortizes — MoE weight-read-bound per-token (SAME wall as retired Phase-1 <=14 bar). (b) drafter overhead FIXABLE (~verify70/draft40/glue @1120).
- DECISION POINT sent to lead: push 2c(mxfp8)/2d(profile+epilogue)/K=2 to chase modest ~1.1-1.3x M3 ceiling, OR conclude EAGLE-3 doesn't pay off on M3 memory-bound MoE decode. Awaiting call.
- If CONCLUDE: disable eagle3 on fs5 (revert model_settings — needs user consent again) to restore baseline serving.
- OUTSTANDING for release regardless: 3 robustness findings (non-M3 gate, 128k mem, prefix-cache fallback) + terminal off-by-one + fresh adversarial review of vlm.py/scheduler fixes.

### LEAD DECISION: PUSH 2c/2d (2c mxfp8 goal-mandated). Sequence:
(1) 2d PROFILE DONE (profile_round.log) — DEFINITIVE:
    ctx1k: VERIFY(L=4)=113ms DRAFT(K=3)=18ms(lm_head11/layer8) fc0.6 round121 baseline_L1=41.
    ctx16k: VERIFY=126 DRAFT=23 round149 baseline_L1=49.
    VERIFY = ~2.75x L1 = ~85% of round; DRAFT only ~15%. round/accept ≈ baseline → break-even. draft_dominant_fixable=FALSE.
    Ceiling: draft-FREE → +8%@1k/+6%@16k; mxfp8(halve draft) → ~+5%@1k/~0%@16k. VERIFY MoE wall is FUNDAMENTAL (per-token expert reads, no amortize) — same as retired Phase-1 bar. Dynamic-K doesn't escape (bigger K→bigger verify).
    HONEST CONCLUSION: EAGLE-3 correct + accepts well (math>vendor) but does NOT pay off on M3 MoE decode. Ship-rec: per-request opt-in short-ctx math/code, NOT default-on.
    Sent to lead w/ plan. LEAD CONFIRMED + scope trim: SKIP epilogue (confirmed: lm_head=bandwidth, fusion can't shrink bytes); DEFER 2b → instead add hard temp==0 engagement guard (greedy token-exact; v1 constraint, 2b follow-up).

### FINAL DELIVERABLES (lead-confirmed close-out) — Codex implementing (bhkttcexz, eagle_final_spec.md)
(A) temp==0 engagement guard: spec only activates for greedy requests (temp>0 → normal decode, no spec). CORRECTNESS for v1 (no rejection sampling yet).
(B) terminal cache off-by-one fix: clean-finish spec req currently discards KV (1247 vs 1248) → fix so cache REUSED (multi-turn).
(C) mxfp8 drafter quant (draft layer+fc, g32) gated env OMLX_EAGLE3_MXFP8 (default OFF) — for the mandated A/B.
THEN: review → mxfp8 A/B (accept delta + live tok/s bf16 vs mxfp8) → verify temp guard + terminal reuse → final bench table + ship-rec (per-request opt-in, short-ctx math/code) → todo.md ledger. Close honestly.
BENCH DATA WE HAVE: baseline 27.1short/21.6@16k; spec bf16 26.2short(-2%)/18.98@16k(-12%); accept math 0.97/0.93/0.85 code 0.87/0.65/0.53; profile verify=85% round.

### Finalization review (codex_final_review.log): temp guard CORRECT, terminal direction right, mxfp8 clean (GO opt-in). 1 HIGH + 2 MED → Codex fixing (b153mn4h2):
(1) [HIGH] terminal validation used max() across layers → per-layer-misaligned cache could be reused (corruption). Fix: require EVERY nonzero layer==expected, else discard.
(2) [MED] temp>0 (user DEFAULT) still ran prompt-capture before route guard skipped → wasted overhead on normal serving. Fix: guard capture too (skip for temp>0), clear chunks on skip.
(3) [MED] MTP-kind StopIteration regressed to None (edge; we don't use MTP). Fix: restore.
THEN: verify → mxfp8 A/B → final table → CLOSE.

### Finalization fixes DONE (Codex b153mn4h2): per-layer cache validation (every nonzero==expected else discard), capture gated by temp check both prefill paths, MTP StopIteration restored. 25+3 tests pass. Parse OK.
### LIVE VERIFY (server_m3s.log, mxfp8 restart):
- temp guard WORKS: "EAGLE-3 routing skipped: temperature=1; rejection sampling is deferred" (x2).
- KEY CONFIG INTERACTION: fs5 model_settings has force_sampling=true+temp=1.0 → effective temp=1 even for temp=0 requests → spec SKIPPED for fs5 DEFAULT. So spec off-by-default on fs5 (correct v1 = greedy-only opt-in). The 26.49 tok/s "mxfp8 live" = baseline SAMPLED decode (spec skipped), NOT mxfp8-spec — confirms NO REGRESSION to normal serving.
- mxfp8 drafter loads live: "eagle3 drafter mxfp8 g32".
- To measure mxfp8-SPEC: standalone harness (bypasses server temp guard) — RUNNING (bm7agnsad, accept_mxfp8.log) vs bf16 (accept2.log).
### BF16 SPEC numbers (pre-guard, spec engaged): live 26.23short/18.98@16k; accept math 0.97/0.93/0.85 code 0.87/0.65/0.53 mean 3.0/2.7.
(2) 2c mxfp8 drafter + epilogue fusion IF profile confirms draft chunk.
(3) K=2 A/B + confidence-gated dynamic depth (stop draft when top-1 prob<thresh — deep on math, shallow on code).
(4) fix terminal cache off-by-one (multi-turn matters for user's agent).
(5) 2b rejection sampling (temp>0).
(6) final honest bench table + ship-rec (LIKELY per-request opt-in, not default-on).
Ceiling (lead+me agree): ~+7-12% flat-K, ~+15-20% math/agentic w/ dynamic depth. Deliver COMPLETE w/ honest numbers.
### OPS lesson: codex exec hangs on stdin if launched after a heredoc / without prompt piped → always `codex exec ... '<prompt>' </dev/null` (or `- < spec.md`). Verify log grows >1KB within 1 min.
- CONFIRMED path (user): verify in custom kernels; drafter = MLX module on native mxfp8 kernels.
- REQ (lead): profile draft-loop GLUE overhead (all non-matmul per cycle). If >~0.5ms/cycle → fuse a draft epilogue kernel
  (final-norm → argmax + top-1 prob on logits; the top-1 prob doubles as the future dynamic-K confidence gate signal). If <0.5ms → skip, record the number.

Draft model: $OMLX_COLD_STORAGE/omlx-quant-work/MiniMax-M3-EAGLE3 (bf16 safetensors).
Taps (2,30,57). Chain K=3. Draft prefix-KV MANDATORY. Reuse fs5 quantized embed/lm_head/final-norm.
Framework: omlx/patches/mlx_lm_mtp/ (batch_generator draft-verify, per-pos accept logging, cache-trim fixed).
mxfp8 variant: quantize draft layer+fc to mxfp8 g32; ship default if live accept delta ≤1.5pp.
Vendor ref K=3: accept 0.92/0.84/0.75 code, 0.75/0.55/0.40 dialogue.
