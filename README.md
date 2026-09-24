# SOLOTASK02 端到端加密通信后端

要实现端到端加密通信后端的设备注册与预密钥发布，用 Python 加 cryptography，同时提供 HTTP 服务和命令行入口。注册走 POST /v1/devices，请求体是 JSON，含 user_id、device_id、identity_key 三个字符串字段，以及数组 signed_prekeys，每个元素含 key_id 与 public_key；成功返回 201，响应含 device_id 与 registered_at。同一 user_id 下 device_id 已存在时返回 409。任一必填字段缺失或 signed_prekeys 元素结构不符时返回 400，并指明是哪个字段。查询走 GET /v1/devices/{device_id}，返回 identity_key、prekey_ids 数组与 registered_at；设备不存在返回 404；prekey_ids 只列出未被撤销的预密钥，同一请求重复返回的顺序完全一致。命令行提供 register 与 show 两个子命令，与上述两个接口一一对应，打印单行 JSON 且字段名与 HTTP 响应一致。服务端只保存公开密钥与标识，不保存明文消息或私钥。同一 user_id 下多台设备互不影响，一台设备的状态变化不影响另一台。

在此基础上新增会话协商与快照查询。POST /v1/sessions 接收 JSON：initiator_device_id、recipient_device_id、prekey_id、ephemeral_key 四个字段均须为非空字符串，ephemeral_key 使用与公钥相同的格式；缺失、类型错误或编码非法返回 400，错误体 field 指明对应字段。两台设备相同返回 400/field=recipient_device_id。服务端校验设备与预密钥归属：未知的发起方/接收方/预密钥分别返回 404/field=initiator_device_id、recipient_device_id、prekey_id；发起方设备、接收方设备或所用预密钥已撤销时分别返回 409/field=initiator_device_id、recipient_device_id、prekey_id，且失败时不写入任何会话。成功返回 201，响应含八个字段：四个入参原样回显，identity_key 为接收方身份公钥，public_key 为所用预密钥公钥，session_id 全局唯一，created_at 为 UTC ISO-8601（带 +00:00）。重复 POST 总是新建会话（新 session_id），不去重。GET /v1/sessions/{session_id} 返回同样的八个字段；未知会话返回 404/field=session_id。会话快照在创建时冻结，此后设备或预密钥撤销不改变快照。创建与撤销在同一把锁下原子线性化：撤销先行则创建 409 且不写，创建先行则 201 且会话保留。命令行的 create-session 与 show-session 与两个接口一一对应。服务端不保存私钥、共享秘密或明文消息。

在既有消息契约之上新增可靠投递：重试投递、显式确认与投递状态查询，全部向后兼容（原有 POST /v1/messages 与拉取接口语义不变）。POST /v1/messages/{session_id}/retry/{message_id} 的请求体含非空字符串 device_id、attempt_id；device_id 必须是该会话当前活跃的接收方设备——未知 session/message 返回 404/field 分别为 session_id/message_id，设备不是接收方、设备未知或已撤销返回 409/field=device_id，字段缺失或类型错误 400/对应 field。该会话该消息的首次尝试返回 201 且 attempts 计为 1；重复 attempt_id 幂等返回 200、不重复计数，新的 attempt_id 返回 200 且 attempts+1；消息被 ack 后再重试仍返回 200，status 保持 acked。响应含 session_id、message_id、status（pending|acked）、attempts、sequence。POST /v1/messages/{session_id}/acks 的请求体含 device_id、message_id 与整数 sequence；权限/未知错误同上，sequence 与该消息的实际序号不符返回 409/field=sequence；首次确认返回 201 并置 acked，重复确认幂等返回 200，响应字段与重试接口相同（未发生过重试时 attempts 为 0）。GET /v1/messages/{session_id}/status/{message_id}?device_id=…：device_id 缺失、为空或重复给出返回 400/field=device_id；未知会话/消息 404 对应 field；越权设备或接收方已撤销 409/field=device_id；200 返回五字段投递状态（无任何投递记录时 status=pending、attempts=0）。所有检查与计数都在存储的同一把锁下原子完成，失败不改变状态。命令行新增 retry-message、ack-message、message-status，与三个接口一一对应，成功打印单行 JSON 到 stdout，失败打印单行 JSON 到 stderr 并以非零码退出。

serve 支持持久化：--data-file 指定状态文件路径，缺省时取环境变量 E2EE_DATA_FILE，二者皆无则保持进程内存储。状态文件为 version=1 的单个 JSON 文档：文件缺失时自动创建；文件损坏、非 JSON 对象或版本不符时拒绝启动（不丢弃既有状态）。每次变更先写同目录临时文件、fsync 文件、替换前以同目录硬链接（.state-*.bak）钉住旧 inode、os.replace 原子替换并 fsync 父目录，使重命名本身也抗崩溃，崩溃不会留下半写文件或未完成的重命名。重启后完整恢复：设备与预密钥（含撤销标记）、会话快照、消息与序号游标、会话级已用 nonce 集合（旧文件缺该字段时由历史消息重建），以及可靠投递的尝试去重集合、attempts、acked 状态与撤销状态。

启用 --data-file 后补齐离线故障恢复与持久化事务语义。重启后默认同步（不带 after）必须从该设备保存的 cursor 继续（cursor 与 updated_at、消息一并恢复）；显式 after 始终只是查询，不读取也不改写保存游标。分页按 sequence 严格升序、无跳号无重复，空页不推进游标，next_cursor 与 has_more 在连续请求间保持一致；各设备游标相互独立。同步默认推进、检查点前进与设备撤销与持久化处于同一事务边界并在存储同一把锁下线性化：变更先在内存生效并在锁内尝试原子落盘，落盘成功才成为新的已提交状态；临时文件写入、fsync 或 os.replace 任一失败时，HTTP 一律返回 503/field=data_file，内存回滚到上一个已持久化状态、旧状态文件原样保留（inode 不变）、临时文件被清理，失败的变更既不推进内存也不推进文件。与撤销并发的检查点只能观察到两种线性化结局：撤销先行则 409/field=device_id 且不留游标记录，检查点先行则 201 且游标落盘。若进程在落盘各步骤之间崩溃，同目录可能遗留 .state-*.tmp（暂存的新文档）或 .state-*.bak（替换前钉住的旧 inode）；下次启动先处理这些遗留文件：正式文件有效时它保持权威、绝不被临时文件覆盖，并清理全部遗留文件；正式文件缺失时只考虑可解析为 version=1、显式含 group_sync_cursors/message_sync_cursors/key_events 段（每段为列表，是一次完整落盘事务的标志）且通过全部语义校验（群组会话引用、消息 sequence 连续、nonce 集合、每设备 cursor 范围、updated_at 非空、key_events 审计链）的遗留快照：带 commit_seq 者先按代次取最高（旧 mtime 的高代快照不会输给新 mtime 的低代快照），最高代若有两个或以上可验证候选则拒绝提升、不动任何文件并拒启，绝不按文件名或 mtime 挑选；仅当所有候选都缺 commit_seq（旧文件）时才沿用按修改时间从新到旧挑选的旧规则；中选快照以 os.replace 原子恢复并 fsync 父目录，其余遗留删除；没有任何有效快照时删除遗留并创建空状态；正式文件存在但损坏时仍拒绝启动、绝不静默覆盖。旧 version=1 文件缺少 group_sync_cursors 时按空游标加载；该字段存在但任一记录畸形（非对象、字段缺失、类型错误、cursor 为负/布尔、updated_at 为空）、cursor 超过所属会话最大 sequence（含空会话仅允许 0）、(session_id,device_id) 重复、或关联到不存在的群组会话/设备/非冻结成员设备时，拒绝启动且绝不覆盖原文件。未启用持久化（无 --data-file 且无 E2EE_DATA_FILE）时保持既有进程内行为，不产生 503。CLI 契约不变：成功 stdout 单行 JSON，失败 stderr 单行 JSON（含 503/field=data_file）并以非零码退出。

启动恢复在 version=1 结构校验之上再做跨实体一致性校验，结构合法但语义矛盾的状态同样拒绝加载，且绝不覆盖原文件（原字节与 inode 保持不变），CLI `serve --data-file` 以 stderr 单行 JSON、field=data_file、退出码 1 拒启。sessions：initiator_device_id、recipient_device_id 必须指向 devices 中已注册设备（设备已撤销不影响历史快照恢复），prekey_id 必须是 recipient 设备自身的预钥 id（属于他人或不存在即拒绝，预钥已撤销不影响恢复），session_id 不得与其他 1:1 会话或任一 group_session 重复，session_id/initiator_device_id/recipient_device_id/prekey_id/ephemeral_key/identity_key/public_key/created_at 均必须为非空字符串（布尔、数字、null、空串一律拒绝）。groups：group_id、creator_device_id、created_at 为非空字符串；creator 必须已注册且必须等于 members 首位；members 为非空、元素为非空字符串且去重的列表；revision 必须为正整数（0、负数、小数、布尔、字符串拒绝）；group_id 不重复。group_sessions：session_id、group_id、initiator_device_id、ephemeral_key、created_at 为非空字符串；group_id 必须关联已存在群组；session_id 不重复且不与 1:1 会话冲突；members 为非空、元素为非空字符串且去重的冻结快照；initiator 必须是已注册设备且属于 members；revision 为正整数且不得超过该群组当前 revision（等于或更旧的冻结快照允许）。devices：user_id/device_id/identity_key/registered_at 为非空字符串，rotated_at 缺省沿用旧规则（等同 registered_at），存在时必须为非空字符串；prekeys 为列表，每个预钥 key_id/public_key 为非空字符串、revoked 为布尔，同设备内 key_id 去重；设备标识不重复。messages、delivery、used_nonces、group_sync_cursors 的既有字段、类型、序号与悬空引用校验继续生效。claim_session_bindings：每条为 claim_id/session_id/recipient_device_id/prekey_id/identity_key/public_key/created_at 非空字符串；claim_id 与 session_id 各自唯一，claim_id 必须指向一条 prekey_claims 记录，session_id 必须指向一条 sessions 记录，且后四字段必须同时与该领取记录和该会话快照的冻结值一致（引用缺失、重复绑定或冻结值不一致即拒绝）。prekey_batch_claims：每条为 claim_id/user_id/claimed_at 非空字符串加非空 devices 列表，每项为 device_id/identity_key/key_id/public_key 非空字符串；批次 claim_id 唯一且不与 prekey_claims 中的 claim_id 冲突，同一批次内 device_id 不重复，每个 device 必须已注册且属于该批次 user_id，key_id 必须为该设备自有预钥且冻结 public_key 与存储一致，key 必须带 consumed 标记，每个 (device,key) 在单领与批量两类领取记录中恰好出现一次。batch_claim_session_bindings：每条为 claim_id/initiator_device_id/created_at 非空字符串加非空 entries 列表，每项为 recipient_device_id/prekey_id/identity_key/public_key/session_id 非空字符串；claim_id 唯一且必须指向一条 prekey_batch_claims，initiator 必须已注册且不在领取设备中，entries 必须与该批量领取快照的设备集合及注册相对顺序逐条一致，session_id 唯一、必须存在于 sessions、且不得被其他单领或批量绑定复用；四项冻结值（recipient_device_id/prekey_id/identity_key/public_key）必须同时与批量领取记录和对应会话快照一致——冻结身份公钥只与领取记录及会话快照比对，绝不与轮换后的设备当前 identity_key 比对。任一悬空引用、重复标识或成员、类型错误、修订越界或字段缺失都拒绝启动；仅缺少既有兼容可选字段（rotated_at、used_nonces、group_sync_cursors、claim_session_bindings、prekey_batch_claims、batch_claim_session_bindings、group_session_rotations 等）的旧 version=1 文件仍按原规则加载（缺 batch_claim_session_bindings 时视为没有任何批量领取被占用）。恢复成功后 HTTP、CLI、同步游标、撤销、重放防护、投递与 AES-GCM 行为保持兼容。

多进程并发使用同一状态文件时由进程级排他锁协调，仅在启用 --data-file 或 E2EE_DATA_FILE 时生效，内存模式行为不变。`serve` 启动在触碰状态文件之前先于状态文件同目录创建（若缺失）并以非阻塞进程级排他锁锁定 `<状态文件>.lock`（POSIX 用 `flock(LOCK_EX|LOCK_NB)`，Windows 用标准库等价的 `msvcrt.locking(LK_NBLCK)` 字节区间锁；该文件只打开不截断、不复用崩溃遗留前缀，故不参与 .state-* 恢复扫描）；锁已被另一存活进程持有时拒绝启动：stderr 单行 JSON、field=data_file、退出码 1，不创建、覆盖或截断正式状态文件，不运行恢复，也不 traceback。不同状态文件各有独立锁文件，互不阻塞；未启用持久化时不创建锁文件、不加任何锁。锁由该进程打开的文件描述格持有，正常退出、收到信号或被 SIGKILL/TerminateProcess 异常终止后均由操作系统立即释放，无陈旧锁需清理；后续进程可立即启动，并沿用既有 version=1 结构校验、跨实体校验、崩溃遗留处理与旧字段兼容规则完整恢复设备、群组、消息、同步游标、投递、撤销与 key_events。持锁进程的设备、群组、消息、投递、同步游标、检查点与撤销写入仍沿用既有同目录临时文件、文件 fsync、替换前硬链接钉住旧 inode、`os.replace` 原子替换及父目录 fsync 的事务语义；平台或文件系统不支持目录 fsync 时安全跳过该步骤而不误报失败，其他 I/O 错误仍使事务失败并返回 503/field=data_file，内存回滚、旧文件字节与 inode 保持不变并清理临时遗留。HTTP 路由、状态码、响应字段、CLI 子命令及成功 stdout/失败 stderr 契约均不改变。

在 version=1 状态文件、commit_seq、进程锁与崩溃遗留恢复之上补齐**不可判定落盘失败后的同进程安全自愈**（仅启用 --data-file 或 E2EE_DATA_FILE 时生效，内存模式保持原行为）。一次提交在 `os.replace` 已落地但随后目录 fsync 失败、且回滚重命名（.bak 改回正式路径）或回滚后的第二次目录 fsync 也失败时进入不可判定状态：正式路径必须保持缺失，旧已提交 inode 仅保留为唯一的 `.state-*.bak`，未提交的新快照被移出正式路径（改为恢复扫描永不提升的 `.state-*.quarantine`，或删除），内存回滚到上一个已提交状态，本次请求返回 503/field=data_file，commit_seq 不前进。此后**同一进程**收到的下一次可持久化写请求，先在存储锁内运行自愈，再只执行这一次“当前请求”本身：把唯一 .bak 在同目录以硬链接（不是重命名，备份名持续钉住 inode 直至新目录项 fsync 落盘）原子提升回正式路径并 fsync 父目录，清理本事务的全部遗留（多余 .bak/.tmp 与该 .quarantine）；提升前对备份执行 version=1、commit_seq 恰为上一代、跨实体引用、消息 sequence、nonce 集合、每设备 cursor 范围、updated_at、key_events 审计链等**全部**语义校验，并要求与该进程最后已提交内存状态语义一致——损坏或与已提交状态不符的悬空备份拒绝提升；正式文件已有效存在且与已提交状态一致时始终优先并仅清理遗留。自愈不消耗 commit_seq；随后当前请求按常规原子事务提交，恰好把 commit_seq 增加一个连续值。这里的“当前请求”是触发自愈的后来请求，**不是**重放此前已返回 503 的旧请求（旧请求的设备撤销等效果不得出现）。自愈失败或随后的当前写入失败，仍返回 503/field=data_file：内存回滚、正式文件字节与 inode、遗留文件与 commit_seq 均不前进，正式路径继续缺失，允许下一次写请求再次尝试自愈。并发写由存储同一把锁串行：恰好一次恢复，随后各写按序各提交一代；有效正式文件始终优先于备份。校验 `.bak/.tmp` 候选时，若存在**两个或以上可验证候选**（version=1、同一上一代、全部语义校验通过且与已提交内存一致），自愈与重启恢复都必须**拒绝提升**：保持正式路径缺失、遗留文件、内存与代次原样，绝不按文件名或 mtime 在候选间挑选，由下一次写或运维介入后重试，唯全部候选都缺 commit_seq 字段的旧文件场景仍沿用旧的 mtime 规则。当连“正式路径缺失 + 唯一 .bak”都无法保证时——回滚改名失败且未提交新快照既不能移入 `.quarantine` 也不能删除，正式路径仍被残留文件占据——进入**阻断态**（落一个同目录 `.state-*.block` 标记，重启后仍然生效）：绝不把该残留正式文件当权威，本进程与重启后的一切写一律 503/field=data_file，标记、备份与残留均不前进；待正式路径被腾出且仅剩一个可验证备份时，自愈/重启才用硬链接提升该备份、清理遗留与标记并解阻；若正式路径上的文件经校验确与唯一/任一已验证备份同代且语义相等（确为上一提交状态），则“有效正式文件优先”，予以接受、清理解阻（残留的未提交新快照盖的是下一代 commit_seq，永远不可能与备份匹配，故不会被误接受）。`os.link` 提升或父目录 fsync 失败时，移除新建正式项、保留备份并返回 503。成功自愈并提交后，设备撤销、消息与投递、同步 cursor、updated_at、key_events 与文件状态一致，重启恢复、旧文件缺可选字段兼容、HTTP/CLI 状态字段、进程锁语义及未启用持久化时的行为全部保持不变。

新增只读状态完整性探针 **GET /v1/persistence/integrity**（无查询参数、无请求体，参数与请求体一律忽略）。仅在启用持久化（--data-file 或 E2EE_DATA_FILE）时可用：未启用持久化的纯内存模式返回 **409**，错误体键序为 `message`、`field`，`message` 为字符串、`field=data_file`。启用持久化时，处理器在**存储同一把锁内**（与设备/群组/消息/投递/游标等一切变更及 key_events 审计追加共锁，故读到的必是某一完整已提交代次）读取并解析正式 version=1 文件：JSON 解析失败、文档非对象、version 缺失/非整数/不等于 1、commit_seq 存在但非非负整数（布尔、负数、小数、字符串）、文件不可读或缺失（含正式路径缺失的 degraded/blocked 态），一律 **503**，错误体键序同样为 `message`、`field`，`message` 为字符串、`field=data_file`。解析通过后在锁内做两件事并比对：① 校验文件代次——文件 commit_seq（旧文件缺该字段按 0）必须恰为本进程最后已提交代次（commit_seq 计数器减一），代次不一致 503；② 把去掉 version/commit_seq 的文件载荷恢复进一个全新内存存储（复用启动恢复的全部跨实体引用、消息 sequence 连续、nonce 集合、每设备 cursor 范围与 updated_at、key_events 审计链等语义校验，任一语义错误 503；未知顶层段也拒绝），再按固定段键序 **devices、sessions、groups、group_sessions、messages、delivery、prekey_claims、prekey_batch_claims、claim_session_bindings、batch_claim_session_bindings、group_session_rotations、group_delivery、used_nonces、group_sync_cursors、message_sync_cursors、message_submissions、key_events** 投影为规范快照（旧文件缺失的段补空，messages/used_nonces 补空对象、其余补空数组；元素格式与既有快照完全一致），与当前活动内存存储在同一把锁下取得的规范快照逐段比对，不一致 503。成功返回 **200**，响应体键序固定为 `commit_seq`、`state_hash`、`consistent`：commit_seq 为上一代非负整数；state_hash 为该规范快照（去掉 version/commit_seq、段按上述顺序、键不排序）以紧凑 JSON（`separators=(",",":")`、`ensure_ascii=False`、UTF-8 字节）求得的 **SHA-256 小写 hex**（非 ASCII 按字面 UTF-8 而非 \uXXXX 转义）；consistent 恒为 `true`。探针完全只读：任何 503 都不改变活动内存、不写不改文件字节与 inode、不清理遗留、不推进任何游标或 commit_seq；同一状态重启后探针结果（代次与哈希）保持不变；探针与并发写线性化，永远只观察到某一完整已提交代次的（commit_seq, state_hash）组合，绝不会观察到文件与内存错代或半提交状态。该接口为纯 HTTP 探针，不新增 CLI 子命令。



在设备契约之上新增身份轮换与预钥补充。POST /v1/devices/{device_id}/identity-key/rotate 的请求体含 identity_key，须为非空字符串且是合法公钥；缺失、类型错误或编码非法返回 400/field=identity_key，设备未知或已撤销分别返回 404/409/field=device_id。设备的 rotated_at 初值等于 registered_at；提交与当前相同的原始公钥返回 200 且 rotated_at 不变（幂等），提交不同公钥则更新 identity_key 并生成新的 UTC ISO-8601（带 +00:00）时间戳；响应含 device_id、identity_key、rotated_at。POST /v1/devices/{device_id}/prekeys 的请求体含 key_id、public_key，二者均须为非空字符串且 public_key 是合法公钥；字段缺失或类型错误返回 400（对应 field），公钥非法返回 400/field=public_key，设备未知或已撤销分别返回 404/409/field=device_id。全新 key_id 按顺序追加并返回 201，响应含 device_id、key_id、public_key；同 key_id、同原始公钥且未撤销时幂等返回 200、响应体相同；同 key_id 但公钥不同，或该 key_id 已撤销，返回 409/field=key_id。GET 返回最新 identity_key 与未撤销的 prekey_ids。身份轮换只影响此后新建的会话，既有会话快照保持冻结；撤销的 key 仍被禁用（无法用同 id 重新补充）。所有校验与写入都在存储的同一把锁下线性化完成，失败时不写入。命令行新增 rotate-identity-key 与 add-prekey，均带 --device-id，其余参数与请求字段一一对应；成功打印单行 JSON 到 stdout，失败打印单行 JSON 到 stderr 并以非零码退出。两项变更都写入 version=1 状态文件并在重启后完整恢复。

在设备契约之上新增群组与群组会话。POST /v1/groups 接收 JSON：group_id、creator_device_id 为非空字符串，member_device_ids 为非空字符串数组（每元素非空）；创建者必须是活跃设备（未知 404/field=creator_device_id，已撤销 409/field=creator_device_id），重复 group_id 返回 409/field=group_id，字段缺失或类型错误 400/对应 field。成员 id 只需是非空字符串、不要求已注册；创建者恒为首位成员，其余按请求顺序去重保留。成功 201 返回 group_id、revision（初值 1）、members、created_at（UTC ISO-8601，+00:00）。GET /v1/groups/{group_id} 返回同样四字段，未知群组 404/field=group_id。成员增删仅创建者可操作：POST /v1/groups/{group_id}/members（等价别名 POST .../members/add）与 POST /v1/groups/{group_id}/members/remove，请求体均含 actor_device_id、device_id（非空字符串，否则 400/对应 field）；未知群组 404/field=group_id，actor 未知 404/field=actor_device_id，actor 已撤销或非创建者 409/field=actor_device_id，目标设备未知 404/field=device_id。新增成员：目标已撤销 409/field=device_id（撤销设备不得加入），新成员 201 且 revision+1，已在群内 200 幂等且 revision 不变。移除成员：成功恒为 200，成员不存在（含已被移出）幂等不递增 revision，已撤销成员仍可移除，创建者移除自己为不改变状态的 no-op（创建者始终保留）。群组查询与成员变更在存储同一把锁下线性化：与设备撤销并发时撤销先则失败、创建/变更先则保留。POST /v1/group-sessions 接收 group_id、initiator_device_id、ephemeral_key（均非空字符串，否则 400/对应 field）；发起者须为群组当前活跃成员——未知群组 404/field=group_id，发起者未知 404/field=initiator_device_id，发起者已撤销或非成员 409/field=initiator_device_id。成功 201 返回 session_id、group_id、initiator_device_id、ephemeral_key、revision、members、created_at；members 与 revision 在创建时冻结为当时的群组成员表与修订号，此后群组增删成员不改变既有会话；重复提交总是新建会话（新 session_id，不去重）。GET /v1/group-sessions/{session_id} 返回冻结快照，未知会话 404/field=session_id。消息读写以冻结成员表为准：POST /v1/messages 的发送方与 GET /v1/messages/{session_id} 的读取方 device_id 都必须是该群组会话的冻结成员（冻结后加入或移出当前群组的设备均不对应快照；冻结成员设备被撤销仍按既有 409 拒绝），未知群组会话沿用 404/field=session_id。群组、成员表、revision 与群组会话冻结快照均写入 version=1 状态文件，重启后完整恢复。命令行新增 group-create、group-show、group-add-member、group-remove-member、create-group-session、show-group-session，与接口一一对应；成功打印单行 JSON 到 stdout（新增成员 201 与重复 200 均为成功），失败打印单行 JSON 到 stderr 并以非零码退出。

在群组会话之上新增按设备的消息增量同步与检查点。GET /v1/group-sessions/{session_id}/sync 以查询参数 device_id（单值、非空，缺失/为空/重复给出均 400/field=device_id）标识读取设备；after 省略时从该设备已保存的游标起步并在锁内把游标推进到本页末条（空页保持不变），给出时必须为非负整数、只做一次性查询且不读取也不改动已保存游标（缺失/非整数/负数/重复给出 400/field=after）；limit 默认 100，取值 1..100，越界或非法 400/field=limit。未知会话 404/field=session_id；设备未知、已撤销或不在冻结成员表（含冻结后才加入当前群组者）均 409/field=device_id；冻结后被移出当前群组的设备仍可同步。成功恒为 200，返回 messages（与 POST 响应同构的信封数组，按 POST 写入顺序即 sequence 升序）、next_cursor（本页末条序号，空页为起点）、has_more（起点之后是否还有消息）。POST /v1/group-sessions/{session_id}/sync/checkpoint 请求体含非空 device_id（缺失或非字符串 400/field=device_id）与整数 cursor（缺失/非整数/布尔/负数 400/field=cursor），且 cursor 不得超过该会话当前最大 sequence；游标前进返回 201 并刷新 updated_at，相同游标返回 200 且 updated_at 不变，倒退或超过最大序号返回 409/field=cursor。响应含 session_id、device_id、cursor、updated_at（UTC ISO-8601，+00:00）。未知会话 404/field=session_id；设备未知、已撤销或非冻结成员均 409/device_id。所有鉴权、分页与游标推进都在存储同一把锁下原子完成，失败不改变消息游标。游标持久化于状态文件的 group_sync_cursors（每会话每设备一条）；旧 version=1 文件缺该字段时按空加载（游标视为 0），该字段存在但结构畸形或版本不符仍拒绝启动。命令行新增 sync-group-messages SESSION_ID --device-id DEVICE_ID [--after N] [--limit N]（省略 --after 即不带该参数，使用并推进设备游标）与 sync-checkpoint SESSION_ID --device-id DEVICE_ID --cursor N；成功在 stdout 打印单行 JSON（检查点 201/200 均为成功），失败在 stderr 打印单行 JSON 并非零退出，连接失败沿用 field=server。

在群组会话之上扩展现有可靠投递（重试、确认、状态查询），使群组会话支持逐设备可靠投递，1:1 会话语义完全不变。三个既有接口（POST /v1/messages/{session_id}/retry/{message_id}、POST /v1/messages/{session_id}/acks、GET /v1/messages/{session_id}/status/{message_id}?device_id=…）现在同时接受群组会话的 session_id：群组消息的 device_id 必须是冻结 members 中的活跃非发送者设备——未知会话/消息仍返回 404/field=session_id 或 message_id，设备未知、已撤销、非冻结成员或为发送者本人均返回 409/field=device_id，字段缺失或类型错误仍 400/对应 field。投递状态以 (session_id, message_id, device_id) 为键：该设备对该消息的首次 retry 返回 201 且 attempts=1；相同 attempt_id 重放返回 200 且不重复计数，新 attempt_id 返回 200 且 attempts+1；消息被该设备 ack 后再重试仍返回 200 且 status 保持 acked。acks 仍校验体内 message_id 与整数 sequence，sequence 与消息实际序号不符返回 409/field=sequence；各设备首次确认返回 201、重复确认幂等返回 200，设备之间互不影响；status 对该设备无任何记录时返回 pending/attempts=0。群组在会话冻结后的增删不改变投递范围：被移出当前群组的冻结成员仍可重试/确认/查询，冻结后才加入的设备一律 409/field=device_id。全部校验与写入都和设备撤销共用存储同一把锁，失败不改变任何状态。version=1 状态文件新增 group_delivery 段保存逐设备的尝试去重集合、attempts、acked 与 ack_sequence；旧文件缺该段按空加载，该段存在时恢复须校验会话（必须指向已存群组会话）、消息、冻结成员设备、attempts 与 attempt_ids 数量一致及 acked/ack_sequence 与消息序号一致，任一矛盾拒绝启动且绝不覆盖原文件；落盘失败仍返回 503/field=data_file 并回滚内存。retry-message、ack-message、message-status 三个 CLI 子命令直接接受群组 session_id，输出格式与退出码不变。

新增幂等消息提交，解决 201 响应丢失后的安全重试；原 POST /v1/messages 保持不变。POST /v1/messages/submit 接收 request_id 及原六个信封字段（session_id、sender_device_id、message_id、sequence、nonce、ciphertext）。request_id 须为非空字符串，其余沿用原校验；非法 400/对应 field。首次请求按既有会话、发送方、message_id、sequence、nonce 顺序校验，成功 201，返回 request_id 及原七字段，失败不占用该 id。request_id 全局唯一：同 id 且六字段完全相同的重放返回 200 及首次响应，即使发送方后来撤销；字段改变或跨会话复用返回 409/field=request_id。并发同 id 仅写一条；消息、序号、nonce 与幂等记录同锁提交，落盘失败 503/field=data_file 并全部回滚。version=1 新增可选 message_submissions，缺失按空；恢复校验 id 唯一、会话和消息引用及六字段一致，矛盾拒启且不覆盖原文件。补强 group_delivery 恢复：设备须已注册、属冻结成员且不是发送者；记录产生后设备撤销仍是合法历史，恢复后写操作继续 409/device_id。新增 CLI submit-message，参数对应七字段；成功 stdout、失败 stderr 单行 JSON 并非零退出，旧接口与旧文件兼容。

新增群组会话轮换：把一个既有群组会话（前序）轮换为一个全新的冻结群组会话（后继）。POST /v1/group-sessions/{session_id}/rotate 接收 JSON：rotation_id、actor_device_id 为非空字符串，ephemeral_key 为非空且合法的公钥编码，expected_revision 为正整数；缺失、类型错误、公钥非法或非正整数返回 400/对应 field。前序不是已知群组会话返回 404/field=session_id；actor 未注册返回 404/field=actor_device_id；actor 已撤销或不是该群组的创建者返回 409/field=actor_device_id；expected_revision 不等于该群组当前 revision 返回 409/field=expected_revision。首次成功返回 201，响应为群组会话七字段（session_id/group_id/initiator_device_id/ephemeral_key/revision/members/created_at）外加 rotation_id、predecessor_session_id 两个字段；后继获得全局唯一的新 session_id，其成员表与 revision 按提交时刻的群组快照冻结（后继的 initiator_device_id 为 actor），前序快照保持不变。同一 rotation_id 重放到同一前序返回 200 及与首次完全相同的原始响应（幂等，即使此后 actor 被撤销或群组 revision 已变）；同一 rotation_id 用于其他前序返回 409/field=rotation_id；前序已被另一个 rotation_id 轮换则返回 409/field=session_id，绝不允许分叉。轮换链可继续：后继本身也可被新的 rotation_id 轮换。全部校验与写入（后继会话、轮换记录）与成员变更、设备撤销、持久化在存储同一把锁下原子共锁，任何失败都不写入；落盘失败返回 503/field=data_file 并回滚内存与文件。version=1 状态文件新增可选 group_session_rotations 段（rotation_id/predecessor_session_id/successor_session_id/group_id/actor_device_id/revision/members/created_at），旧文件缺该段按空加载；该段存在时恢复须校验：rotation_id 唯一、前序与后继均指向已存群组会话、同一前序不被多条记录引用（无分叉）、同一后继不被重复产生、group_id 与前序/后继一致、actor 为已注册的群组创建者，且冻结的 initiator/revision/members/created_at 与后继会话快照完全一致——任一矛盾拒绝启动且绝不覆盖原文件。轮换产生的后继就是普通群组会话，消息与可靠投递沿用既有冻结成员规则：group_delivery 拒绝消息发送者本人或任何未注册/已撤销/非冻结成员设备。命令行新增 rotate-group-session SESSION_ID --rotation-id … --actor-device-id … --ephemeral-key … --expected-revision N（四个参数一一对应），成功（201/200）stdout 单行 JSON，失败 stderr 单行 JSON 并非零退出，输出契约与既有命令一致。

## 当前状态

接口已实现（纯标准库 HTTP 服务 + `cryptography` 校验公钥），并带有单元/集成测试。

- `POST /v1/devices`：注册设备并发布签名预密钥；成功 `201`，冲突 `409`，校验失败 `400`（错误体带 `field` 指明字段，数组元素使用 `signed_prekeys[i].key_id` 这样的路径）。
- `GET /v1/devices/{device_id}`：返回 `identity_key`、`prekey_ids`（仅未撤销，顺序与注册时一致且重复请求完全相同）、`registered_at`；不存在返回 `404`。
- `POST /v1/devices/{device_id}/revoke`：撤销设备及其全部预密钥。已存在设备返回 `200`，响应体 `{"device_id":...,"revoked":true}`；重复调用幂等；未知设备返回 `404`（`field=device_id`）。撤销后 `GET` 的 `prekey_ids` 为空，`identity_key` 与 `registered_at` 不变。
- `POST /v1/devices/{device_id}/prekeys/{key_id}/revoke`：撤销单个预密钥。成功 `200`，响应体 `{"device_id":...,"key_id":...,"revoked":true}`；重复调用幂等；未知设备 `404/field=device_id`，设备存在但 `key_id` 未知 `404/field=key_id`。仅排除目标 key，其他 key 与同用户的其他设备不受影响。
- `POST /v1/devices/{device_id}/identity-key/rotate`：轮换身份公钥。`identity_key` 须为非空合法公钥，否则 `400/field=identity_key`；设备未知/已撤销 `404/409/field=device_id`。`rotated_at` 初值等于 `registered_at`；相同原始公钥 `200` 且时间戳不变，不同公钥更新并生成新的 UTC ISO-8601（`+00:00`）。响应含 `device_id`/`identity_key`/`rotated_at`。轮换只影响此后新建的会话，既有会话快照冻结。
- `POST /v1/devices/{device_id}/prekeys`：补充预密钥。`key_id`、`public_key` 非空且公钥合法，否则 `400`（`public_key` 非法时 `field=public_key`）；设备未知/已撤销 `404/409/field=device_id`。新 `key_id` 顺序追加返回 `201`（`device_id`/`key_id`/`public_key`）；同 id 同原始公钥且未撤销幂等 `200`；同 id 异值或该 id 已撤销 `409/field=key_id`。撤销的 key 无法用同 id 重新补充。
- `POST /v1/prekeys/claim`：领取一次性预密钥。体含非空字符串 `recipient_device_id`、`claim_id`，缺失或类型错误 `400`/对应 `field`。接收方设备未知 `404/field=recipient_device_id`，已撤销 `409/同字段`，无未撤销未消费预密钥 `409/field=prekey_id`。按注册顺序选首个未撤销、未消费 key，原子标记消费后 `201` 返回 `claim_id`/`recipient_device_id`/`identity_key`/`key_id`/`public_key`/`claimed_at`（UTC ISO-8601，`+00:00`）。重复 `claim_id` 返回 `200` 及与首次完全相同的响应且不再消费；不同 `claim_id` 领取下一个 key。已消费 key 从 `GET /v1/devices/{id}` 的 `prekey_ids` 排除；用该 `key_id` 创建会话返回 `409/field=prekey_id`；撤销设备或 key 后不可领取。领取、会话创建、撤销共享存储同一把锁线性化，并发竞态仅一个领取成功。记录落于状态文件 `prekey_claims`（含 `device_id`、`key_id` 及上述字段），旧文件缺该字段且 key 无 `consumed` 标记按未消费加载；该段存在时校验 claim_id 与 (device,key) 唯一、字段类型、对已注册设备及其自有 key 的引用、与 key 的 `consumed` 标记一致，畸形拒启。落盘失败回滚（内存与文件均不前移）并经 HTTP 返回 `503/field=data_file`，重启后恢复领取映射。
- `POST /v1/prekeys/claim-batch`：为用户多台设备批量领取预钥（单领接口契约不变）。体含非空字符串 `user_id`、`claim_id`，缺失或类型错误 `400`/对应 `field`。用户不存在（无任何设备）`404/field=user_id`，设备全部已撤销（无活跃设备）`409/field=device_id`。按注册顺序枚举该用户全部未撤销设备，每台取首个未撤销且未消费预钥；任一活跃设备无可用项则 `409/field=prekey_id` 且所有设备的所有项均不消费（先全量选取、后一次性消费）。成功在存储锁内一次性消费全部选中 key 并持久化，`201` 返回 `claim_id`/`user_id`/`claimed_at`/`devices`；`devices` 按上述注册顺序排列，每项含 `device_id`/`identity_key`/`key_id`/`public_key`，身份与预钥材料在领取时冻结，`claimed_at` 为 UTC ISO-8601（`+00:00`）。重复批量 `claim_id` 返回 `200` 及与首次完全相同的响应且不再消费（即便此后设备或 key 被撤销、身份密钥轮换）。`claim_id` 与单领共用同一全局幂等命名空间：已被另一类领取使用的 id 返回 `409/field=claim_id`。批量选取与消费和单领、撤销、补钥、会话创建共享存储同一把锁线性化，任何失败不改状态。记录落于 version=1 状态文件新增 `prekey_batch_claims` 段，旧文件缺该段按空加载；该段存在时校验批次 claim_id 唯一且不与 `prekey_claims` 冲突、每个设备属于该批次 `user_id`、每个预钥归属于该设备、冻结 `identity_key`/`public_key` 与当前存储一致、每个 (device,key) 恰被一条领取（单领或批量）记录支撑且与 key 的 `consumed` 标记一致，矛盾拒启。落盘失败返回 `503/field=data_file` 并回滚（内存与文件均不前移，本次未消费任何 key）。
- `POST /v1/groups`：创建群组。`group_id`、`creator_device_id` 非空字符串，`member_device_ids` 非空字符串数组（元素非空），否则 `400`/对应 `field`；创建者未知/已撤销 `404/409/field=creator_device_id`；重复 `group_id` `409/field=group_id`。成员 id 不要求已注册；创建者恒居首位，其余按序去重。成功 `201` 返回 `group_id`/`revision`（初值 1）/`members`/`created_at`。
- `GET /v1/groups/{group_id}`：返回群组四字段快照；未知 `404/field=group_id`。
- `POST /v1/groups/{group_id}/members`（别名 `…/members/add`）：创建者新增成员。体含非空 `actor_device_id`、`device_id`，否则 `400`；未知群组 `404/field=group_id`，actor 未知 `404/field=actor_device_id`，actor 已撤销或非创建者 `409/field=actor_device_id`，目标设备未知 `404/field=device_id`、已撤销 `409/field=device_id`（撤销设备不得加入）。新成员 `201` 且 `revision+1`；已是成员 `200` 幂等且 `revision` 不变。
- `POST /v1/groups/{group_id}/members/remove`：创建者移除成员。错误码同上；成功恒为 `200`——成员不在表中为幂等 no-op（`revision` 不变），已撤销成员仍可移除，创建者移除自己为保留创建者的 no-op。
- `POST /v1/group-sessions`：冻结群组成员创建群组会话。`group_id`/`initiator_device_id`/`ephemeral_key` 非空字符串否则 `400`；未知群组 `404/field=group_id`；发起者未知 `404/field=initiator_device_id`，已撤销或非当前成员 `409/field=initiator_device_id`。成功 `201` 返回 `session_id`/`group_id`/`initiator_device_id`/`ephemeral_key`/`revision`/`members`/`created_at`，成员表与 `revision` 创建时冻结；重复提交总是新建（新 `session_id`）。
- `GET /v1/group-sessions/{session_id}`：返回冻结快照；未知 `404/field=session_id`。此后群组增删成员不影响既有快照；新建会话才反映最新成员表与 `revision`。
- `POST /v1/group-sessions/{session_id}/rotate`：把前序群组会话轮换为一个全新冻结后继。体含非空 `rotation_id`/`actor_device_id`、合法公钥 `ephemeral_key`、正整数 `expected_revision`，否则 `400`/对应 field；前序未知 `404/session_id`；actor 未知 `404/actor_device_id`，已撤销或非群组创建者 `409/actor_device_id`；`expected_revision` 与群组当前 `revision` 不符 `409/expected_revision`。首次 `201` 返回群组会话七字段加 `rotation_id`/`predecessor_session_id`，后继 `session_id` 全局唯一、成员表与 `revision` 按提交时群组快照冻结、前序不变。同一 `rotation_id` 重放同一前序返回 `200` 原始响应（幂等），用于其他前序 `409/rotation_id`；前序已被其他 id 轮换 `409/session_id`，绝不分叉；后继可继续轮换。与成员变更、撤销、持久化共用同一把锁原子共锁，失败不写；落盘失败 `503/data_file` 并回滚。记录存于状态文件可选段 `group_session_rotations`，旧文件缺段按空加载，恢复时校验引用、唯一、无分叉及冻结值与后继快照一致，矛盾拒启且不覆盖原文件。
- 群组会话消息读写限冻结成员：向群组会话 `POST /v1/messages` 的发送方、`GET /v1/messages/{sid}` 的 `device_id` 都必须在冻结成员表中（冻结后被移出当前群组的设备仍可读该冻结会话；冻结后新加入的设备不可；设备被撤销仍按既有规则 `409`）；未知群组会话 `404/field=session_id`。序号连续、nonce 会话级去重等既有消息语义不变。
- 群组创建、成员增删、群组会话创建与设备撤销共享存储同一把锁、原子线性化，失败不写入、不推进 `revision`。
- 命令行 `group-create` / `group-show` / `group-add-member` / `group-remove-member` / `create-group-session` / `show-group-session` 与接口一一对应；成功 stdout 单行 JSON（含 200/201），失败 stderr 单行 JSON 且非零退出。群组与冻结会话均持久化到 version=1 状态文件并在重启后完整恢复。
- `GET /v1/group-sessions/{session_id}/sync`：按设备增量同步群组会话消息。查询参数 `device_id` 单值非空（缺失/为空/重复 `400/field=device_id`）；`after` 省略时从该设备已保存游标起步并在锁内推进到本页末条（空页不变），显式给出时须为非负整数、仅一次性查询且不改动已保存游标（非法 `400/field=after`）；`limit` 默认 100（1..100，越界非法 `400/field=limit`）。未知会话 `404/session_id`，设备未知、已撤销或不在冻结成员表（冻结后才加入者）均 `409/device_id`；冻结后被移出当前群组者仍可同步。恒 `200` 返回 `messages`（信封数组，按 POST 写入/sequence 升序）、`next_cursor`（末条序号，空页为起点）、`has_more`（起点之后是否还有消息）。
- `POST /v1/group-sessions/{session_id}/sync/checkpoint`：推进设备同步检查点。体含非空 `device_id`（非法 `400/device_id`）与整数 `cursor`（缺失/非整数/布尔/负数 `400/cursor`），且 `cursor` 不超过该会话最大 sequence。前进 `201` 并刷新 `updated_at`，相同 `200` 且 `updated_at` 不变，倒退或超过最大序号 `409/field=cursor`；返回 `session_id`/`device_id`/`cursor`/`updated_at`。未知会话 `404/session_id`；设备未知、已撤销或非冻结成员均 `409/device_id`。鉴权、分页与游标推进共享存储同一把锁，失败不改变消息游标。游标存于状态文件 `group_sync_cursors`，旧 version=1 文件缺该字段按空（游标 0）加载，该字段畸形或版本不符拒绝启动。
- 命令行 `sync-group-messages SESSION_ID --device-id DEVICE_ID [--after N] [--limit N]`（省略 `--after` 即用并推进设备游标）与 `sync-checkpoint SESSION_ID --device-id DEVICE_ID --cursor N` 与接口一一对应；成功 stdout 单行 JSON（检查点 201/200 均成功），失败 stderr 单行 JSON 且非零退出，连接失败仍为 `field=server`。
- `GET /v1/sessions/{session_id}/sync`：兼容 1:1 与群组会话的按设备离线增量同步。查询参数 `device_id` 单值非空（缺失/为空/重复 `400/field=device_id`）；`after` 省略时从该设备已保存游标起步并在锁内推进到本页末条（空页不推进），显式给出时须为非负整数、仅一次性查询且不改动已保存游标（非法 `400/field=after`）；`limit` 默认 100（1..100，越界非法 `400/field=limit`）。会话既非 1:1 亦非群组会话时 `404/field=session_id`；设备未知、已撤销均 `409/field=device_id`；1:1 会话仅其发起方与接收方、群组会话仅冻结成员可同步（冻结后被移出当前群组者仍可同步，冻结后才加入者不可），越权均 `409/field=device_id`。恒 `200` 返回 `messages`（信封数组，按 sequence 升序）、`next_cursor`（末条序号，空页为起点）、`has_more`（起点之后是否还有消息）。
- `POST /v1/sessions/{session_id}/sync/checkpoint`：推进设备的统一同步检查点（1:1 与群组会话）。体含非空 `device_id`（缺失/空/非字符串 `400/device_id`）与整数 `cursor`（缺失/非整数/布尔/负数 `400/cursor`），且 `cursor` 不超过该会话最大 sequence。前进 `201` 并刷新 `updated_at`，相同 `200` 且 `updated_at` 不变，倒退或超过最大序号 `409/field=cursor`；返回 `session_id`/`device_id`/`cursor`/`updated_at`（从未推进过的设备提交 0 为 `200` no-op，不落记录）。会话未知 `404/session_id`；设备未知、已撤销或非会话参与者均 `409/device_id`。鉴权、分页与游标推进共享存储同一把锁，失败不改变消息游标；游标独立存于状态文件 `message_sync_cursors` 段（与群组专用的 `group_sync_cursors` 互不影响），旧 version=1 文件缺该字段按空（游标 0）加载，记录错、键重复、悬空（会话/设备未知）、越权（非会话参与者）、cursor 越界或 `updated_at` 为空均拒绝启动且原文件不变；落盘失败 `503/field=data_file` 并回滚（内存与文件均不前移）。
- 命令行 `sync-session-messages SESSION_ID --device-id DEVICE_ID [--after N] [--limit N]`（省略 `--after` 即用并推进设备游标）与 `sync-session-checkpoint SESSION_ID --device-id DEVICE_ID --cursor N` 与上述接口一一对应；成功 stdout 单行 JSON（检查点 201/200 均成功），失败 stderr 单行 JSON 且非零退出，连接失败仍为 `field=server`。
- 命令行 `register` / `show` / `revoke-device` / `revoke-prekey` / `rotate-identity-key` / `add-prekey` / `claim-prekey` / `claim-user-prekeys` / `create-session` / `create-session-from-claim` / `create-batch-sessions` / `show-session` / `group-create` / `group-show` / `group-add-member` / `group-remove-member` / `create-group-session` / `show-group-session` / `rotate-group-session` / `sync-group-messages` / `sync-checkpoint` / `sync-session-messages` / `sync-session-checkpoint` 与接口一一对应，成功时在 stdout 打印单行 JSON，字段名与 HTTP 一致；失败时在 stderr 打印单行 JSON 错误并以非零码退出。连接失败或超时时，API 命令在 stderr 打印 `field` 为 `server` 的单行 JSON、非零退出，且不输出 traceback。
- `POST /v1/sessions`：协商会话。四个入参（`initiator_device_id`、`recipient_device_id`、`prekey_id`、`ephemeral_key`）须为非空字符串，`ephemeral_key` 须为合法公钥编码，否则 `400` 并以 `field` 指明；两台设备相同返回 `400/field=recipient_device_id`。未知设备/预密钥返回 `404`，已撤销返回 `409`，`field` 分别为 `initiator_device_id` / `recipient_device_id` / `prekey_id`；预密钥已被领取（consumed）同样 `409/field=prekey_id`；失败不写入。成功 `201` 返回八字段：四入参回显、`identity_key`（接收方身份公钥）、`public_key`（所用预密钥公钥）、唯一 `session_id`、`created_at`（UTC ISO-8601，`+00:00`）。重复 POST 总是新建会话。
- `POST /v1/sessions/from-claim`：凭已成功的预密钥领取建立会话（原 `POST /v1/sessions` 契约不变）。请求体含 `claim_id`、`initiator_device_id`、`ephemeral_key`，均须为非空字符串，`ephemeral_key` 沿用既有公钥编码；缺失、类型或编码错误返回 `400/field=对应字段`。`claim_id` 无对应领取记录返回 `404/field=claim_id`；发起设备未知返回 `404/field=initiator_device_id`，已撤销返回 `409/同字段`；领取关联的接收设备已撤销返回 `409/field=recipient_device_id`，领取所用预钥已撤销返回 `409/field=prekey_id`。成功 `201` 返回现有会话八字段；其中 `recipient_device_id`、`prekey_id`、`identity_key`、`public_key` 必须采用领取时冻结值（接收设备此后轮换身份密钥不改变这四项），`initiator_device_id`、`ephemeral_key` 取请求值，`session_id`/`created_at` 按现有会话契约生成。每个 `claim_id` 只能成功建立一次会话；首次成功后重复提交（即便换发起设备或临时公钥）一律 `409/field=claim_id` 且不新建会话。绑定检查、状态校验、会话写入与设备/预钥撤销共享存储同一把锁线性化，任一校验失败不写入会话、也不占用该领取（领取在后续条件满足时仍可建立一次会话）。绑定写入 version=1 状态文件（`claim_session_bindings` 段）并随重启恢复，旧文件缺该段按空加载；落盘失败返回 `503/field=data_file` 并回滚绑定与会话（内存与文件均不前移，该 `claim_id` 未被占用，稍后可重试成功）。
- `POST /v1/sessions/from-batch-claim`：凭一次成功的批量预钥领取，原子建立"每台领取设备一个"的会话集合（原单领会话与 `from-claim` 契约不变）。请求体含 `claim_id`、`initiator_device_id`（均须为非空字符串）与数组 `ephemeral_keys`；缺失/类型错误返回 `400`/对应 `field`。数组为非空数组，每项含唯一的非空字符串 `device_id` 与合法公钥 `ephemeral_key`（错误分别指明 `ephemeral_keys[i].device_id` / `ephemeral_keys[i].ephemeral_key`，重复 device_id 指向后一项的路径）；请求设备集合必须与领取快照严格相等——多出的设备返回 `400/field=ephemeral_keys[i].device_id`，缺少设备或集合不等返回 `400/field=ephemeral_keys`。`claim_id` 无对应领取记录返回 `404/field=claim_id`；该 id 指向单领记录或该批次已成功建过会话返回 `409/field=claim_id`（换发起设备或临时公钥同样拒绝）。发起设备未知 `404`/已撤销 `409`，`field=initiator_device_id`；发起设备本身也在领取快照中时整批返回 `400/field=initiator_device_id` 且不创建任何会话（该校验先于发起设备撤销判定）。领取快照中的接收设备或其冻结预钥在此后被撤销，分别返回 `409/field=recipient_device_id`（指明具体设备）或 `409/field=prekey_id`。成功 `201` 返回 `claim_id` 与 `sessions`；`sessions` 严格按领取快照的注册相对顺序列出，每个为现有八字段会话——`recipient_device_id`/`prekey_id`/`identity_key`/`public_key` 一律取领取时冻结值（设备此后轮换身份密钥不改变快照），`initiator_device_id` 与各 `ephemeral_key` 取请求值，`session_id`/`created_at` 按现有会话契约生成；请求数组自身的排列顺序不影响响应顺序。会话集合与单条批量绑定在存储同一把锁下共锁提交，全成或全败；落盘失败 `503/field=data_file` 并回滚全部会话与绑定，该批次仍可稍后重试成功。绑定写入 version=1 状态文件新增 `batch_claim_session_bindings` 段（含 `claim_id`/`initiator_device_id`/`created_at` 及有序 `entries`，每项含 `recipient_device_id`/`prekey_id`/`identity_key`/`public_key`/`session_id`），旧文件缺该段按空加载；恢复时校验批次引用、会话引用（不与单领绑定或其他批量绑定共用 session）、冻结值同时与领取记录和会话快照一致（冻结身份公钥只与领取记录及会话快照比对，不与轮换后的设备当前值比对）、发起设备已注册且不在领取设备中，且 entries 与领取快照的设备集合及顺序完全一致，矛盾拒启且不覆盖原文件；`prekey_batch_claims` 恢复时还校验设备列表保持注册相对顺序，乱序即拒启。
- `GET /v1/sessions/{session_id}`：返回与创建时一致的八字段快照；未知会话 `404/field=session_id`。快照创建后冻结，设备/预密钥撤销不改变它。
- 会话创建与设备/预密钥撤销共享同一把锁、线性化执行：撤销先行则创建得 `409` 且不写，创建先行则得 `201` 且会话保留，不存在中间态。
- 撤销与查询共享同一把锁、线性化执行：并发的 `GET` 只能看到某次撤销操作前或后的完整快照，不会观察到中间态。
- `POST /v1/messages`：向会话投递加密消息信封。请求体含 `session_id`、`sender_device_id`、`message_id`、`sequence`、`nonce`、`ciphertext`；`sequence` 从 1 开始逐条连续。未知会话 `404/field=session_id`；发送方设备不存在或已撤销 `409/field=sender_device_id`；`message_id` 在会话内重复 `409/field=message_id`；序号不连续 `409/field=sequence`；`nonce` 已在同一会话历史消息中使用（会话级重放）`409/field=nonce`，按原始字符串逐字节比较，同一 nonce 在不同会话可各自使用；字段缺失或类型错误 `400/field=对应字段`。成功 `201`，响应为完整信封（六入参回显）加 `created_at`（UTC ISO-8601，`+00:00`）。重复 nonce、重复 message_id、错序等检查与写入在存储同一把锁下按固定优先顺序（会话→发送方→message_id→sequence→nonce）原子线性化，任一失败不写入消息、不推进序号、不改变任何投递状态。
- `POST /v1/messages/submit`：幂等提交加密消息信封（原 `POST /v1/messages` 语义不变）。请求体为 `request_id` 加原六字段；`request_id` 非空字符串，否则 `400/field=request_id`。首次成功 `201` 返回 `request_id` 及原七字段；同 id 同六字段重放返回 `200` 及首次响应（发送方此后撤销亦如此）；同 id 任一字段改变或跨会话复用 `409/field=request_id`；失败不占用该 id。消息、序号、nonce 与幂等记录同锁提交，落盘失败 `503/field=data_file` 并全部回滚。记录存于 version=1 可选段 `message_submissions`，旧文件缺段按空加载；恢复校验 id 唯一、会话/消息引用与六字段一致，矛盾拒启且不覆盖原文件。
- `GET /v1/messages/{session_id}`：分页拉取会话消息。查询参数 `device_id` 必填，`after` 默认 0（须 ≥0），`limit` 默认 100（1..100）；未知会话 `404/field=session_id`，`device_id` 非活跃设备 `409/field=device_id`，参数缺失/非法 `400/field=对应参数`。返回 `messages`（与 POST 响应同构的信封数组，筛 `sequence > after` 升序）与 `next_after`（空页等于 `after`，否则为末条序号）。消息读取与设备撤销共享同一把锁，不观察中间态；已存消息在发送方被撤销后仍可读取。
- 命令行 `send-message` / `submit-message` / `pull-messages` 与上述接口一一对应，同样打印单行 JSON（`submit-message` 的 201 与 200 均为成功）。
- `POST /v1/messages/{session_id}/retry/{message_id}`：可靠投递重试。体含非空 `device_id`、`attempt_id`；接收方须活跃，未知会话/消息 `404`（`field=session_id`/`message_id`），设备不符、未知或已撤销 `409/field=device_id`，字段缺失或类型错误 `400`。首次尝试 `201`、`attempts=1`；相同 `attempt_id` 幂等 `200` 不计数，新 id `200` 且 `attempts+1`；acked 后重试仍 `200` 且保持 `acked`。返回 `session_id`/`message_id`/`status(pending|acked)`/`attempts`/`sequence`。
- `POST /v1/messages/{session_id}/acks`：体含 `device_id`、`message_id`、整数 `sequence`；权限/未知同上，序号不符 `409/field=sequence`。首次 `201` 置 acked，重复 `200` 幂等；响应同五字段（无重试时 `attempts=0`）。
- `GET /v1/messages/{session_id}/status/{message_id}?device_id=…`：`device_id` 缺失/为空/重复 `400/field=device_id`；未知 `404`，越权或接收方撤销 `409/field=device_id`；`200` 返回五字段投递状态（无记录时 `pending`/`0`）。
- 命令行 `retry-message` / `ack-message` / `message-status` 与三个接口一一对应；成功 stdout 单行 JSON（含 200/201），失败 stderr 单行 JSON 且非零退出。
- 可靠投递的校验、去重计数与确认在存储同一把锁下原子完成，失败不改变状态。
- 持久化：`serve --data-file PATH`（缺省取 `$E2EE_DATA_FILE`，再缺省为纯内存）。状态文件 `version=1` JSON，缺失自动创建，损坏/非对象/版本不符拒绝启动；每次变更临时文件 + fsync + `os.replace` 原子替换。重启恢复设备（含 `identity_key` 与 `rotated_at`）、预密钥（含新增与撤销标记）、会话、消息与序号游标、会话级已用 nonce 集合，以及投递去重集合、`attempts`、acked 与设备撤销状态。群组与冻结群会话快照、群会话轮换记录（含无分叉与冻结值校验）、每设备群会话同步游标（`group_sync_cursors`，含 `cursor`/`updated_at`）以及兼容 1:1/群组会话的统一同步游标（`message_sync_cursors`，结构与校验同上）同样恢复；重启后默认同步从各设备保存游标继续，显式 `after` 仍只查询。缺少已用 nonce 集合字段的旧 version=1 文件可正常加载并由历史消息重建该集合；该字段存在但结构畸形仍视为损坏并拒绝启动。缺少 `group_sync_cursors`/`message_sync_cursors` 字段的旧 version=1 文件按空加载（各设备游标视为 0）；任一字段存在但记录畸形、字段缺失/类型错误、cursor 越界（含空会话）、键重复或关联（会话/设备/参与者）对不上时拒绝启动，且不覆盖原文件。
- 持久化事务与故障恢复：同步默认推进、检查点前进、设备撤销等变更与落盘在同一存储锁事务内线性化；临时文件写入/fsync/`os.replace` 失败时返回 `503/field=data_file`，内存回滚到上一个已持久化状态、旧文件保留、临时文件清理，失败不推进内存或文件；与撤销并发只能得到成功（游标落盘）或 `409/field=device_id`。未启用持久化时保持既有行为，无 503。
- `GET /v1/persistence/integrity`：只读状态完整性探针（无参数、无请求体，均忽略）。纯内存模式 `409/field=data_file`；启用持久化后，在存储同一把锁（与一切写入及审计追加共锁）内读取正式 version=1 文件并做启动级语义校验，文件代次（commit_seq，旧文件缺字段按 0）须等于最后已提交代次，去 version/commit_seq 后的 17 段规范快照（devices、sessions、groups、group_sessions、messages、delivery、prekey_claims、prekey_batch_claims、claim_session_bindings、batch_claim_session_bindings、group_session_rotations、group_delivery、used_nonces、group_sync_cursors、message_sync_cursors、message_submissions、key_events，缺段补空）须与锁内内存快照完全一致。成功 `200` 返回键序 `commit_seq`、`state_hash`、`consistent`：commit_seq 为上一代非负整数；state_hash 为规范快照紧凑 JSON（ensure_ascii=false、UTF-8）的 SHA-256 小写 hex；consistent 恒 true。解析/版本/代次/语义错误、文件缺失不可读或快照不一致一律 `503/field=data_file`（错误体键序 message、field），且只读不改：内存、文件字节与 inode、游标、commit_seq 均不变；与并发写线性化，只观察完整已提交代次，重启后结果稳定。无 CLI 子命令。
- 提交完整性旁车：启用持久化后，每次提交在同一存储锁事务内除原子重写状态文件外，还维护同目录旁车 `<数据文件>.integrity`（内存模式不生成，其余入口不变）。旁车为单个紧凑 JSON（UTF-8、ensure_ascii=false、无多余空白/换行），顶层键序 `version`、`entries` 且 `version=1`；`entries` 按 `commit_seq` 升序，每项键序 `commit_seq`、`state_hash`、`prev_hash`、`hash`：commit_seq 为非负整数代次；state_hash 为该代去 version/commit_seq/标记后 17 段规范快照（与探针同款紧凑 UTF-8 JSON）的 SHA-256 小写 hex；首项 `prev_hash` 为空串、其后每项接上一项 `hash`；`hash` 为去掉 `hash` 字段后按键排序的同款紧凑 JSON 的 SHA-256 小写 hex。状态文件以 `integrity_log_version=1` 记录格式（位于 commit_seq 之后的信封字段，不参与 state_hash）。缺该标记且无旁车的旧文件可正常启动并保持纯状态文件行为，首次真实提交在同一事务内写入 `integrity_log_version=1` 并追加首项（链锚定于该提交代次，首项 prev_hash 为空），其后每代追加且幂等不重复追加。标记有而旁车无、旁车有而标记无、标记 ≠1，或旁车结构/哈希链/单项哈希/末项代次与 state_hash 不符，一律拒绝启动：stderr 单行 JSON（field=data_file）、退出码 1，状态文件与旁车均不改动。两文件经各自临时文件 fsync 后同事务 `os.replace` 落盘，任一侧写入/fsync/替换失败整体回滚（内存回滚、旧状态与旧旁车保留、临时文件清理、不消耗代次），请求 `503/field=data_file`；重试同代只追加一次。`GET /v1/persistence/integrity/history`（无参数、无请求体，均忽略）：无 `integrity_log_version` 标记（含纯内存模式与尚未首写的旧文件）一律 `409/field=data_file`；有标记时在存储锁内校验状态文件并读全旁车、核对链与末项，成功 `200` 返回键序 `commit_seq`、`entries`，commit_seq 同时等于状态文件代次与末项代次；解析/链/哈希/末项错误一律 `503/field=data_file` 且只读不改。无 CLI 子命令。
- 命令行 `encrypt-message` / `decrypt-message` 为纯本地 AES-256-GCM 加解密（不访问服务器）：`--session-id`、`--key`（base64 编码的 32 字节密钥）、`--plaintext`（UTF-8）→ 输出 `session_id`/`nonce`（12 字节，base64）/`ciphertext`（base64，末尾附 16 字节 GCM tag）；`decrypt-message` 额外接收 `--nonce`/`--ciphertext` → 输出 `session_id`/`plaintext`。`session_id` 的 UTF-8 字节作为 AAD 参与认证。任何失败在 stderr 打印带 `field` 的单行 JSON 并以非零码退出。
- 设备密钥审计链：注册、身份轮换、新增预密钥、预密钥撤销、设备撤销各自在提交时向该设备的审计链追加一条事件（链按 `device_id` 隔离；幂等重放与失败不追加）。事件含 `device_id`/`seq`/`type`/`payload`/`prev_hash`/`hash`/`created_at` 七字段；`seq` 从 1 连续，首事件 `prev_hash` 为空串，之后链接前一事件的 `hash`。`payload`：`registered` 为注册体的身份公钥与有序预密钥，`identity_rotated` 为 `old_identity_key`/`new_identity_key`，`prekey_added`/`prekey_revoked` 为 `key_id`+公钥，`device_revoked` 为空对象（一条事件全撤）。`hash` 为去掉 `hash` 字段后按键排序、紧凑分隔、Unicode 原样 JSON 的 UTF-8 SHA-256 小写 hex。事件与状态变更在同一存储锁事务内提交，落盘失败 `503/field=data_file` 并随整体状态回滚。`GET /v1/devices/{device_id}/key-events`：`after` 默认 0（须 ≥0），`limit` 默认 100（1..100），非法 `400/field=对应参数`，未知设备 `404/field=device_id`，已撤销设备仍可读；返回 `seq > after` 的前 `limit` 条及 `next_after`（空页等于 `after`）与 `has_more`。命令行 `key-events DEVICE_ID [--after N] [--limit N]` 与该接口对应。事件持久化于 version=1 可选段 `key_events`，旧文件缺段可正常加载（无任何变更时该段保持缺省）；段存在时恢复校验字段/设备引用/seq 连续与 prev_hash/hash 链，并重放每条链核对身份公钥、预密钥顺序/公钥/撤销标记与设备撤销状态，矛盾拒绝启动且不改动原文件。对缺段的旧文件采用惰性迁移：启用 `--data-file`/`E2EE_DATA_FILE` 时，任何会持久化的变更（消息、群组、同步、投递、撤销等）以及新增设备注册，先在同一存储锁事务内、提交本次变更之前，按注册顺序为所有无链已注册设备补建锚点——每台设备一条 `registered`（取当前 identity_key 与按顺序的 key_id/public_key，`prev_hash` 为空、`created_at` 固定为 `registered_at`），随后按预钥顺序为已撤销预钥各补一条 `prekey_revoked`，再为已撤销设备补一条 `device_revoked`（撤销设备缺省的预钥撤销标记在同一事务内归一并记入锚点，不重复也不遗漏撤销），最后再追加本次真实事件；迁移事件按同套规则重算 `seq`/`prev_hash`/`hash`。锚点与业务变更共用一次 version=1 原子落盘，任一写入/fsync/替换失败返回 `503/field=data_file`，内存、文件字节与 inode、业务状态与审计链一并回滚，旧文件保持缺段原样；迁移成功后 key-events 查询、重启恢复与幂等撤销/轮换/补钥沿用原字段与状态码。
- 服务端仅保存标识与公开密钥（identity key、signed pre-key、临时公钥均为公钥），不保存私钥、共享秘密或明文消息。存储线程安全；默认进程内，`serve --data-file`/`$E2EE_DATA_FILE` 时持久化到 version=1 JSON 文件并在重启后完整恢复。

## 安装依赖

需要 Python 3.10+。

```bash
python3 -m pip install -r requirements.txt
# 或者安装为包（提供 e2ee-backend 命令）
python3 -m pip install -e .
```

## 启动方式

```bash
# 直接运行 HTTP 服务（默认 127.0.0.1:8080；默认进程内存储）
python3 -m e2ee_backend serve --host 0.0.0.0 --port 8080

# 持久化到 JSON 状态文件（也可用环境变量 E2EE_DATA_FILE 指定；缺失自动创建）
python3 -m e2ee_backend serve --port 8080 --data-file /var/lib/e2ee/state.json
```

命令行调用（公钥支持 PEM 文本、base64/hex 编码的 DER，或 base64/hex 编码的 32 字节 X25519/Ed25519 原始点；也可用 `@路径` 从文件读取）：

```bash
# 注册（--prekey 可重复，格式为 KEY_ID:PUBLIC_KEY，或 @文件 读取 JSON 对象）
python3 -m e2ee_backend register \
  --user-id alice --device-id laptop \
  --identity-key  BASE64_OR_PEM_PUBLIC_KEY \
  --prekey 1:BASE64_OR_PEM_PUBLIC_KEY \
  --prekey 2:BASE64_OR_PEM_PUBLIC_KEY
# => {"device_id":"laptop","registered_at":"2026-09-19T10:10:50.232732+00:00"}

# 查询
python3 -m e2ee_backend show laptop
# => {"identity_key":"...","prekey_ids":["1","2"],"registered_at":"..."}

# 撤销单个预密钥（幂等）
python3 -m e2ee_backend revoke-prekey --device-id laptop --key-id 1
# => {"device_id":"laptop","key_id":"1","revoked":true}

# 撤销整台设备（幂等）
python3 -m e2ee_backend revoke-device --device-id laptop
# => {"device_id":"laptop","revoked":true}

# 轮换身份公钥（相同公钥幂等；不同公钥刷新 rotated_at）
python3 -m e2ee_backend rotate-identity-key \
  --device-id laptop --identity-key BASE64_OR_PEM_NEW_PUBLIC_KEY
# => {"device_id":"laptop","identity_key":"…","rotated_at":"2026-09-19T10:14:02.345678+00:00"}

# 补充预密钥（新 id 返回 201；同 id 同公钥幂等返回 200）
python3 -m e2ee_backend add-prekey \
  --device-id laptop --key-id 3 --public-key BASE64_OR_PEM_PUBLIC_KEY
# => {"device_id":"laptop","key_id":"3","public_key":"…"}

# 协商会话（ephemeral-key 同样支持 PEM/base64/hex 公钥与 @文件）
python3 -m e2ee_backend create-session \
  --initiator-device-id laptop --recipient-device-id phone \
  --prekey-id 1 --ephemeral-key BASE64_OR_PEM_PUBLIC_KEY
# => {"session_id":"…","initiator_device_id":"laptop","recipient_device_id":"phone",
#     "prekey_id":"1","ephemeral_key":"…","identity_key":"…","public_key":"…",
#     "created_at":"2026-09-19T10:12:33.802144+00:00"}

# 查询会话快照
python3 -m e2ee_backend show-session SESSION_ID
# => 同样的八个字段，单行 JSON

# 批量领取后，一次性为领取快照中的每台设备建立会话（--ephemeral-key 每台设备一个，可任意顺序）
python3 -m e2ee_backend create-batch-sessions \
  --claim-id BATCH_CLAIM_ID --initiator-device-id laptop \
  --ephemeral-key phone:BASE64_OR_PEM_PUBLIC_KEY \
  --ephemeral-key tablet:BASE64_OR_PEM_PUBLIC_KEY
# => {"claim_id":"…","sessions":[{八个字段，按领取顺序}, …]}

# 创建群组（--member-device-id 可重复；创建者自动成为首位成员）
python3 -m e2ee_backend group-create \
  --group-id team --creator-device-id laptop --member-device-id phone
# => {"group_id":"team","revision":1,"members":["laptop","phone"],
#     "created_at":"2026-09-21T…+00:00"}

# 查询群组
python3 -m e2ee_backend group-show team

# 新增成员（新成员 201；已在群内重复添加 200 幂等，revision 不变）
python3 -m e2ee_backend group-add-member \
  --group-id team --actor-device-id laptop --device-id tablet
# 移除成员（恒为 200；重复移除幂等）
python3 -m e2ee_backend group-remove-member \
  --group-id team --actor-device-id laptop --device-id phone

# 创建群组会话（成员表与 revision 在此刻冻结；重复提交总是新建）
python3 -m e2ee_backend create-group-session \
  --group-id team --initiator-device-id laptop --ephemeral-key BASE64_OR_PEM_PUBLIC_KEY
# => {"session_id":"…","group_id":"team","initiator_device_id":"laptop",
#     "ephemeral_key":"…","revision":2,"members":["laptop","tablet"],
#     "created_at":"2026-09-21T…+00:00"}

# 查询冻结的群组会话快照
python3 -m e2ee_backend show-group-session SESSION_ID
# => 同样的七个字段，单行 JSON；此后群组增删成员不改变该快照

# 轮换群组会话（冻结提交时的成员表与 revision；同 id 重放幂等）
python3 -m e2ee_backend rotate-group-session SESSION_ID \
  --rotation-id ROTATION_ID --actor-device-id laptop \
  --ephemeral-key BASE64_OR_PEM_PUBLIC_KEY --expected-revision 2
# => {"session_id":"…","group_id":"team","initiator_device_id":"laptop",
#     "ephemeral_key":"…","revision":2,"members":["laptop","tablet"],
#     "created_at":"2026-09-21T…+00:00",
#     "rotation_id":"ROTATION_ID","predecessor_session_id":"SESSION_ID"}

# 增量同步群组会话消息（省略 --after：使用并推进该设备的服务端游标）
python3 -m e2ee_backend sync-group-messages SESSION_ID \
  --device-id phone --limit 100
# => {"messages":[…],"next_cursor":3,"has_more":false}
# 显式 --after 只做一次性查询，不改动服务端游标
python3 -m e2ee_backend sync-group-messages SESSION_ID \
  --device-id phone --after 0 --limit 100

# 推进/确认该设备的同步检查点（前进 201，相同 200，倒退 409）
python3 -m e2ee_backend sync-checkpoint SESSION_ID \
  --device-id phone --cursor 3
# => {"session_id":"…","device_id":"phone","cursor":3,"updated_at":"…"}

# 兼容 1:1/群组会话的离线增量同步（省略 --after：使用并推进服务端游标）
python3 -m e2ee_backend sync-session-messages SESSION_ID \
  --device-id phone --limit 100
# => {"messages":[…],"next_cursor":3,"has_more":false}
# 显式 --after 只做一次性查询，不改动服务端游标
python3 -m e2ee_backend sync-session-messages SESSION_ID \
  --device-id phone --after 0 --limit 100
# 推进/确认该设备的统一同步检查点（前进 201，相同 200，倒退/越界 409）
python3 -m e2ee_backend sync-session-checkpoint SESSION_ID \
  --device-id phone --cursor 3
# => {"session_id":"…","device_id":"phone","cursor":3,"updated_at":"…"}

# 投递加密消息信封（sequence 从 1 开始逐条连续）
python3 -m e2ee_backend send-message \
  --session-id SESSION_ID --sender-device-id laptop \
  --message-id msg-1 --sequence 1 \
  --nonce BASE64_NONCE --ciphertext BASE64_CIPHERTEXT
# => {"session_id":"…","sender_device_id":"laptop","message_id":"msg-1",
#     "sequence":1,"nonce":"…","ciphertext":"…",
#     "created_at":"2026-09-19T10:13:01.123456+00:00"}

# 幂等提交（201 首次；同 request-id 同字段重放 200 并返回首次响应）
python3 -m e2ee_backend submit-message \
  --request-id req-uuid-1 --session-id SESSION_ID --sender-device-id laptop \
  --message-id msg-1 --sequence 1 \
  --nonce BASE64_NONCE --ciphertext BASE64_CIPHERTEXT
# => {"request_id":"req-uuid-1","session_id":"…","sender_device_id":"laptop",
#     "message_id":"msg-1","sequence":1,"nonce":"…","ciphertext":"…",
#     "created_at":"2026-09-19T10:13:01.123456+00:00"}

# 分页拉取会话消息（--after 默认 0，--limit 默认 100）
python3 -m e2ee_backend pull-messages SESSION_ID --device-id phone --after 0 --limit 100
# => {"messages":[…],"next_after":1}

# 可靠投递：首次尝试（201），相同 --attempt-id 幂等（200，不重复计数）
python3 -m e2ee_backend retry-message SESSION_ID MESSAGE_ID \
  --device-id phone --attempt-id attempt-uuid-1
# => {"session_id":"…","message_id":"…","status":"pending","attempts":1,"sequence":1}

# 显式确认（首次 201，重复 200）
python3 -m e2ee_backend ack-message SESSION_ID \
  --device-id phone --message-id MESSAGE_ID --sequence 1
# => {"session_id":"…","message_id":"…","status":"acked","attempts":1,"sequence":1}

# 查询投递状态
python3 -m e2ee_backend message-status SESSION_ID MESSAGE_ID --device-id phone
# => {"session_id":"…","message_id":"…","status":"acked","attempts":1,"sequence":1}

# 本地 AES-256-GCM 加解密（不访问服务器；--key 为 base64 编码的 32 字节密钥）
python3 -m e2ee_backend encrypt-message \
  --session-id SESSION_ID --key BASE64_32BYTE_KEY --plaintext "hello"
# => {"session_id":"…","nonce":"…","ciphertext":"…"}   # nonce 12B，ciphertext 附 16B tag
python3 -m e2ee_backend decrypt-message \
  --session-id SESSION_ID --key BASE64_32BYTE_KEY \
  --nonce BASE64_NONCE --ciphertext BASE64_CIPHERTEXT
# => {"session_id":"…","plaintext":"hello"}
```

默认服务地址为 `http://127.0.0.1:8080`，可用全局参数 `--base-url` 或环境变量 `E2EE_BASE_URL` 覆盖。

## 基础测试命令

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖：公钥解析、注册成功/409 冲突/各类 400（指明字段）、查询/404、`prekey_ids` 顺序稳定与撤销过滤、设备与单预密钥撤销（200、幂等、404 及对应 field、同用户设备隔离）、撤销与查询并发线性化、会话协商成功（八字段、四入参回显、接收方公钥、唯一 session_id、重复 POST 新建）、会话各类 400/404/409（对应 field、失败不写）、会话快照在撤销后不变、创建与撤销并发原子线性化（撤销先行 409/创建先行 201 两种顺序均被观察到）、消息投递（信封回显与 created_at、sequence 从 1 连续、重复 message_id/错序/发送方撤销/未知会话的 400/404/409 及对应 field、失败不推进序号、会话间序号独立、会话内重复 nonce 返回 409/nonce 且不写入不推进不改变投递状态、nonce 原始字符串比较、同 nonce 跨会话可复用、重复 id/错序/重放的固定优先顺序、相同信封并发只线性化写入一条）、消息拉取（分页升序、next_after 语义、参数校验、读取方撤销 409、发送方撤销后已存消息仍可读）、可靠投递（首次 retry 201/attempts+1、相同 attempt_id 200 不计数、新 attempt_id 200 计数、acked 后重试仍 200 且保持 acked、ack 首次 201/重复 200 幂等、无重试直接 ack、序号不符 409/sequence、未知会话/消息 404、设备不符/未知/撤销 409/device_id、status 缺参 400/device_id、失败不改变状态）、持久化（缺失创建 version=1、损坏/非对象/版本不符/载荷畸形拒绝启动、原子替换无残留临时文件、重启恢复尝试去重/attempts/acked、序号游标、已用 nonce 集合并拒绝重放、旧文件缺 nonce 字段由历史消息重建、nonce 字段畸形拒绝启动、设备与预密钥撤销）、`serve --data-file` 真实子进程重启恢复与损坏拒启、AES-256-GCM 加解密（UTF-8 回环、随机 nonce、AAD 绑定 session_id、密钥/nonce/密文长度与编码校验、篡改与错密钥认证失败）、HTTP 全链路（真实 socket）、CLI 子命令（真实子进程，含连接失败 `field=server`、非零退出、无 traceback）、群组与群组会话（创建四字段与成员去重、重复 group_id 409、创建者未知/已撤销 404/409、成员数组各类 400、查询 404、成员新增 201/重复 200、未知群组/设备与越权 404/409 及对应 field、已撤销设备不得加入、移除恒 200/幂等/创建者保留、revision 递增与冻结不变、群会话发起者规则 404/409、重复提交新建 session_id、冻结成员表与 revision 不受后续增删影响、消息读写限冻结成员（冻结后新加入不可、移出者仍可读、设备撤销仍 409）、创建与撤销并发线性化只有合法结局、群组与冻结快照持久化重启恢复）、群组 HTTP 全链路与群组 CLI 子进程（成功 stdout/失败 stderr/非零退出）、群组会话增量同步与检查点（after 省略取设备游标并锁内推进/空页不变、显式 after 不改游标、分页升序与 next_cursor/has_more、device_id/after/limit 各类 400、未知会话 404、设备未知/撤销/非冻结成员均 409、移出者可同步而冻结后加入者不可、checkpoint 前进 201 刷时间/相同 200 不变/倒退与超最大序号 409、失败不改游标、每设备游标独立、游标持久化重启恢复、旧文件缺 group_sync_cursors 按空加载而畸形（记录/类型/范围/重复键/悬空会话或设备/非冻结成员/空会话 cursor>0）拒绝启动且不覆盖原文件、重启后默认同步从保存 cursor 与 updated_at 继续而显式 after 仅查询、HTTP 全链路与 CLI 子进程成功 stdout/失败 stderr/非零退出）、持久化事务故障恢复（临时文件写入/fsync/os.replace 失败时默认同步推进、检查点前进、设备撤销均回滚内存、旧文件 inode 不变、临时文件清理并抛出 PersistenceUnavailable；HTTP 层返回 503/field=data_file；修复后重试从旧状态继续；检查点与设备撤销并发仅线性化为 201 落盘或 409/device_id；多线程写失败时内存与文件均不推进）、崩溃恢复最后边界（替换后 fsync 父目录，失败时凭替换前硬链接把旧文件字节与 inode 原子改回；启动时清理同目录 .state-*.tmp/.bak 遗留：正式文件有效则不被覆盖并清理遗留，正式缺失则在显式含 group_sync_cursors/message_sync_cursors/key_events 段（一次完整落盘事务的标志，空亦为列表）并通过 version=1 全量语义校验的快照中，带 commit_seq 者按代次取最高（最高代有两个及以上可验证候选则拒启、绝不按名或 mtime 选），全缺 commit_seq 的旧候选才按 mtime 取最新，将其 os.replace 原子恢复并 fsync 父目录；缺这些段的 legacy/部分快照即便语义合法也不得提升（避免丢失游标或审计链），无有效快照则清空遗留创建空状态，正式存在但损坏仍拒启；真实 serve 子进程从遗留快照恢复后 HTTP 同步继续）、持久化完整性探针（内存模式 409/data_file；空状态代次 0、成功 200 键序 commit_seq/state_hash/consistent 与 64 位小写 hex；代次与哈希随提交推进且探针不消耗代次；紧凑 JSON、ensure_ascii=false 的 UTF-8 直字哈希；顶层键重排哈希不变、旧文件缺可选段按空哈希；文件损坏/非对象/版本不符/缺 version、commit_seq 为布尔负数小数字符串、代次不符、跨实体语义错误、未知段、快照不一致（旧代字节）、文件缺失均 503/data_file 且文件字节与 inode、内存、游标、commit_seq 均不变；重启后哈希稳定；与并发写共锁线性化只观察完整已提交 (commit_seq,state_hash)；HTTP 全链路忽略参数/请求体、错误体键序 message/field；真实 serve 子进程内存 409、持久化 200、带外损坏 503 且不修复文件）。

## 代码结构

```
e2ee_backend/
  crypto.py       # cryptography 公钥解析/校验（PEM、DER、原始曲线点）与 AES-256-GCM 本地加解密
  models.py       # Device / SignedPreKey / KeyEvent / Session / Group / GroupSession / GroupSessionRotation / GroupSyncCursor / Message / MessageSubmission / MessageDelivery 数据模型
  storage.py      # 线程安全的进程内存储（插入顺序、撤销过滤、原子快照、会话原子创建、群组与冻结群会话、群会话轮换（幂等重放、无分叉）、消息原子追加与分页（群组会话限冻结成员）、幂等消息提交（request_id 重放/冲突）、投递去重/确认、群会话按设备同步分页与检查点游标、整体状态快照与恢复）
  persistence.py  # version=1 JSON 状态文件：缺失创建、损坏/版本不符拒启、临时文件+fsync+硬链接备份+os.replace+父目录 fsync（平台不支持时安全跳过）原子替换，不可判定失败后在存储锁内同进程自愈（校验并硬链接提升唯一 .bak、清理遗留、再提交当前请求；两个及以上可验证候选拒绝提升、绝不按名或 mtime 选；连正式路径都无法腾空时落 .state-*.block 标记进入阻断态、一切写 503 且不把残留正式文件当权威，路径腾出且唯一可验证备份时才硬链接提升解阻），启动清理/恢复崩溃遗留快照（同代多候选同样拒绝）
  locking.py      # --data-file 状态文件的进程级非阻塞独占锁：POSIX flock、Windows msvcrt.locking 字节区间锁，锁文件只创建不截断、不参与崩溃遗留扫描
  service.py      # 业务逻辑与字段校验（400/404/409，设备/预密钥/会话/群组/群会话/群会话轮换/消息/投递/群同步）
  http_app.py     # POST/GET 路由与 JSON 响应（注册、查询、两类撤销、会话协商与查询、群组创建/查询/成员增删、群组会话协商/查询/轮换、消息投递/幂等提交与拉取、重试/确认/状态、群会话同步与检查点、只读持久化完整性探针 /v1/persistence/integrity）
  cli.py          # register/show/key-events/revoke-*/rotate-identity-key/add-prekey/claim-prekey/claim-user-prekeys/create-session/create-session-from-claim/show-session/group-*/create-group-session/show-group-session/rotate-group-session/sync-group-messages/sync-checkpoint/send-message/submit-message/pull-messages/retry-message/ack-message/message-status/encrypt-message/decrypt-message/serve 命令行入口
tests/            # unittest 测试
```
