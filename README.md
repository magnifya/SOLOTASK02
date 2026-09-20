# SOLOTASK02 端到端加密通信后端

要实现端到端加密通信后端的设备注册与预密钥发布，用 Python 加 cryptography，同时提供 HTTP 服务和命令行入口。注册走 POST /v1/devices，请求体是 JSON，含 user_id、device_id、identity_key 三个字符串字段，以及数组 signed_prekeys，每个元素含 key_id 与 public_key；成功返回 201，响应含 device_id 与 registered_at。同一 user_id 下 device_id 已存在时返回 409。任一必填字段缺失或 signed_prekeys 元素结构不符时返回 400，并指明是哪个字段。查询走 GET /v1/devices/{device_id}，返回 identity_key、prekey_ids 数组与 registered_at；设备不存在返回 404；prekey_ids 只列出未被撤销的预密钥，同一请求重复返回的顺序完全一致。命令行提供 register 与 show 两个子命令，与上述两个接口一一对应，打印单行 JSON 且字段名与 HTTP 响应一致。服务端只保存公开密钥与标识，不保存明文消息或私钥。同一 user_id 下多台设备互不影响，一台设备的状态变化不影响另一台。

在此基础上新增会话协商与快照查询。POST /v1/sessions 接收 JSON：initiator_device_id、recipient_device_id、prekey_id、ephemeral_key 四个字段均须为非空字符串，ephemeral_key 使用与公钥相同的格式；缺失、类型错误或编码非法返回 400，错误体 field 指明对应字段。两台设备相同返回 400/field=recipient_device_id。服务端校验设备与预密钥归属：未知的发起方/接收方/预密钥分别返回 404/field=initiator_device_id、recipient_device_id、prekey_id；发起方设备、接收方设备或所用预密钥已撤销时分别返回 409/field=initiator_device_id、recipient_device_id、prekey_id，且失败时不写入任何会话。成功返回 201，响应含八个字段：四个入参原样回显，identity_key 为接收方身份公钥，public_key 为所用预密钥公钥，session_id 全局唯一，created_at 为 UTC ISO-8601（带 +00:00）。重复 POST 总是新建会话（新 session_id），不去重。GET /v1/sessions/{session_id} 返回同样的八个字段；未知会话返回 404/field=session_id。会话快照在创建时冻结，此后设备或预密钥撤销不改变快照。创建与撤销在同一把锁下原子线性化：撤销先行则创建 409 且不写，创建先行则 201 且会话保留。命令行的 create-session 与 show-session 与两个接口一一对应。服务端不保存私钥、共享秘密或明文消息。

在既有消息契约之上新增可靠投递：重试投递、显式确认与投递状态查询，全部向后兼容（原有 POST /v1/messages 与拉取接口语义不变）。POST /v1/messages/{session_id}/retry/{message_id} 的请求体含非空字符串 device_id、attempt_id；device_id 必须是该会话当前活跃的接收方设备——未知 session/message 返回 404/field 分别为 session_id/message_id，设备不是接收方、设备未知或已撤销返回 409/field=device_id，字段缺失或类型错误 400/对应 field。该会话该消息的首次尝试返回 201 且 attempts 计为 1；重复 attempt_id 幂等返回 200、不重复计数，新的 attempt_id 返回 200 且 attempts+1；消息被 ack 后再重试仍返回 200，status 保持 acked。响应含 session_id、message_id、status（pending|acked）、attempts、sequence。POST /v1/messages/{session_id}/acks 的请求体含 device_id、message_id 与整数 sequence；权限/未知错误同上，sequence 与该消息的实际序号不符返回 409/field=sequence；首次确认返回 201 并置 acked，重复确认幂等返回 200，响应字段与重试接口相同（未发生过重试时 attempts 为 0）。GET /v1/messages/{session_id}/status/{message_id}?device_id=…：device_id 缺失、为空或重复给出返回 400/field=device_id；未知会话/消息 404 对应 field；越权设备或接收方已撤销 409/field=device_id；200 返回五字段投递状态（无任何投递记录时 status=pending、attempts=0）。所有检查与计数都在存储的同一把锁下原子完成，失败不改变状态。命令行新增 retry-message、ack-message、message-status，与三个接口一一对应，成功打印单行 JSON 到 stdout，失败打印单行 JSON 到 stderr 并以非零码退出。

serve 支持持久化：--data-file 指定状态文件路径，缺省时取环境变量 E2EE_DATA_FILE，二者皆无则保持进程内存储。状态文件为 version=1 的单个 JSON 文档：文件缺失时自动创建；文件损坏、非 JSON 对象或版本不符时拒绝启动（不丢弃既有状态）。每次变更先写同目录临时文件、fsync 后 os.replace 原子替换，崩溃不会留下半写文件。重启后完整恢复：设备与预密钥（含撤销标记）、会话快照、消息与序号游标、每会话已用 nonce 集合（旧文件缺少该字段时由历史消息重建），以及可靠投递的尝试去重集合、attempts、acked 状态与撤销状态。

在设备契约之上新增身份轮换与预钥补充。POST /v1/devices/{device_id}/identity-key/rotate 的请求体含 identity_key，须为非空字符串且是合法公钥；缺失、类型错误或编码非法返回 400/field=identity_key，设备未知或已撤销分别返回 404/409/field=device_id。设备的 rotated_at 初值等于 registered_at；提交与当前相同的原始公钥返回 200 且 rotated_at 不变（幂等），提交不同公钥则更新 identity_key 并生成新的 UTC ISO-8601（带 +00:00）时间戳；响应含 device_id、identity_key、rotated_at。POST /v1/devices/{device_id}/prekeys 的请求体含 key_id、public_key，二者均须为非空字符串且 public_key 是合法公钥；字段缺失或类型错误返回 400（对应 field），公钥非法返回 400/field=public_key，设备未知或已撤销分别返回 404/409/field=device_id。全新 key_id 按顺序追加并返回 201，响应含 device_id、key_id、public_key；同 key_id、同原始公钥且未撤销时幂等返回 200、响应体相同；同 key_id 但公钥不同，或该 key_id 已撤销，返回 409/field=key_id。GET 返回最新 identity_key 与未撤销的 prekey_ids。身份轮换只影响此后新建的会话，既有会话快照保持冻结；撤销的 key 仍被禁用（无法用同 id 重新补充）。所有校验与写入都在存储的同一把锁下线性化完成，失败时不写入。命令行新增 rotate-identity-key 与 add-prekey，均带 --device-id，其余参数与请求字段一一对应；成功打印单行 JSON 到 stdout，失败打印单行 JSON 到 stderr 并以非零码退出。两项变更都写入 version=1 状态文件并在重启后完整恢复。

## 当前状态

接口已实现（纯标准库 HTTP 服务 + `cryptography` 校验公钥），并带有单元/集成测试。

- `POST /v1/devices`：注册设备并发布签名预密钥；成功 `201`，冲突 `409`，校验失败 `400`（错误体带 `field` 指明字段，数组元素使用 `signed_prekeys[i].key_id` 这样的路径）。
- `GET /v1/devices/{device_id}`：返回 `identity_key`、`prekey_ids`（仅未撤销，顺序与注册时一致且重复请求完全相同）、`registered_at`；不存在返回 `404`。
- `POST /v1/devices/{device_id}/revoke`：撤销设备及其全部预密钥。已存在设备返回 `200`，响应体 `{"device_id":...,"revoked":true}`；重复调用幂等；未知设备返回 `404`（`field=device_id`）。撤销后 `GET` 的 `prekey_ids` 为空，`identity_key` 与 `registered_at` 不变。
- `POST /v1/devices/{device_id}/prekeys/{key_id}/revoke`：撤销单个预密钥。成功 `200`，响应体 `{"device_id":...,"key_id":...,"revoked":true}`；重复调用幂等；未知设备 `404/field=device_id`，设备存在但 `key_id` 未知 `404/field=key_id`。仅排除目标 key，其他 key 与同用户的其他设备不受影响。
- `POST /v1/devices/{device_id}/identity-key/rotate`：轮换身份公钥。`identity_key` 须为非空合法公钥，否则 `400/field=identity_key`；设备未知/已撤销 `404/409/field=device_id`。`rotated_at` 初值等于 `registered_at`；相同原始公钥 `200` 且时间戳不变，不同公钥更新并生成新的 UTC ISO-8601（`+00:00`）。响应含 `device_id`/`identity_key`/`rotated_at`。轮换只影响此后新建的会话，既有会话快照冻结。
- `POST /v1/devices/{device_id}/prekeys`：补充预密钥。`key_id`、`public_key` 非空且公钥合法，否则 `400`（`public_key` 非法时 `field=public_key`）；设备未知/已撤销 `404/409/field=device_id`。新 `key_id` 顺序追加返回 `201`（`device_id`/`key_id`/`public_key`）；同 id 同原始公钥且未撤销幂等 `200`；同 id 异值或该 id 已撤销 `409/field=key_id`。撤销的 key 无法用同 id 重新补充。
- 命令行 `register` / `show` / `revoke-device` / `revoke-prekey` / `rotate-identity-key` / `add-prekey` / `create-session` / `show-session` 与接口一一对应，成功时在 stdout 打印单行 JSON，字段名与 HTTP 一致；失败时在 stderr 打印单行 JSON 错误并以非零码退出。连接失败或超时时，API 命令在 stderr 打印 `field` 为 `server` 的单行 JSON、非零退出，且不输出 traceback。
- `POST /v1/sessions`：协商会话。四个入参（`initiator_device_id`、`recipient_device_id`、`prekey_id`、`ephemeral_key`）须为非空字符串，`ephemeral_key` 须为合法公钥编码，否则 `400` 并以 `field` 指明；两台设备相同返回 `400/field=recipient_device_id`。未知设备/预密钥返回 `404`，已撤销返回 `409`，`field` 分别为 `initiator_device_id` / `recipient_device_id` / `prekey_id`；失败不写入。成功 `201` 返回八字段：四入参回显、`identity_key`（接收方身份公钥）、`public_key`（所用预密钥公钥）、唯一 `session_id`、`created_at`（UTC ISO-8601，`+00:00`）。重复 POST 总是新建会话。
- `GET /v1/sessions/{session_id}`：返回与创建时一致的八字段快照；未知会话 `404/field=session_id`。快照创建后冻结，设备/预密钥撤销不改变它。
- 会话创建与设备/预密钥撤销共享同一把锁、线性化执行：撤销先行则创建得 `409` 且不写，创建先行则得 `201` 且会话保留，不存在中间态。
- 撤销与查询共享同一把锁、线性化执行：并发的 `GET` 只能看到某次撤销操作前或后的完整快照，不会观察到中间态。
- `POST /v1/messages`：向会话投递加密消息信封。请求体含 `session_id`、`sender_device_id`、`message_id`、`sequence`、`nonce`、`ciphertext`；`sequence` 从 1 开始逐条连续。未知会话 `404/field=session_id`；发送方设备不存在或已撤销 `409/field=sender_device_id`；`message_id` 在会话内重复 `409/field=message_id`；序号不连续 `409/field=sequence`；同一会话历史消息已使用完全相同的 `nonce`（按原字符串比较）返回 `409/field=nonce`，同一 `nonce` 在不同会话可用；字段缺失或类型错误 `400/field=对应字段`。成功 `201`，响应为完整信封（六入参回显）加 `created_at`（UTC ISO-8601，`+00:00`）。重复 nonce、重复 message_id、错序等检查与写入在存储同一把锁下原子线性化，失败不写入任何消息，序号游标不前进，投递状态不变。
- `GET /v1/messages/{session_id}`：分页拉取会话消息。查询参数 `device_id` 必填，`after` 默认 0（须 ≥0），`limit` 默认 100（1..100）；未知会话 `404/field=session_id`，`device_id` 非活跃设备 `409/field=device_id`，参数缺失/非法 `400/field=对应参数`。返回 `messages`（与 POST 响应同构的信封数组，筛 `sequence > after` 升序）与 `next_after`（空页等于 `after`，否则为末条序号）。消息读取与设备撤销共享同一把锁，不观察中间态；已存消息在发送方被撤销后仍可读取。
- 命令行 `send-message` / `pull-messages` 与上述两个接口一一对应，同样打印单行 JSON。
- `POST /v1/messages/{session_id}/retry/{message_id}`：可靠投递重试。体含非空 `device_id`、`attempt_id`；接收方须活跃，未知会话/消息 `404`（`field=session_id`/`message_id`），设备不符、未知或已撤销 `409/field=device_id`，字段缺失或类型错误 `400`。首次尝试 `201`、`attempts=1`；相同 `attempt_id` 幂等 `200` 不计数，新 id `200` 且 `attempts+1`；acked 后重试仍 `200` 且保持 `acked`。返回 `session_id`/`message_id`/`status(pending|acked)`/`attempts`/`sequence`。
- `POST /v1/messages/{session_id}/acks`：体含 `device_id`、`message_id`、整数 `sequence`；权限/未知同上，序号不符 `409/field=sequence`。首次 `201` 置 acked，重复 `200` 幂等；响应同五字段（无重试时 `attempts=0`）。
- `GET /v1/messages/{session_id}/status/{message_id}?device_id=…`：`device_id` 缺失/为空/重复 `400/field=device_id`；未知 `404`，越权或接收方撤销 `409/field=device_id`；`200` 返回五字段投递状态（无记录时 `pending`/`0`）。
- 命令行 `retry-message` / `ack-message` / `message-status` 与三个接口一一对应；成功 stdout 单行 JSON（含 200/201），失败 stderr 单行 JSON 且非零退出。
- 可靠投递的校验、去重计数与确认在存储同一把锁下原子完成，失败不改变状态。
- 持久化：`serve --data-file PATH`（缺省取 `$E2EE_DATA_FILE`，再缺省为纯内存）。状态文件 `version=1` JSON，缺失自动创建，损坏/非对象/版本不符拒绝启动；每次变更临时文件 + fsync + `os.replace` 原子替换。重启恢复设备（含 `identity_key` 与 `rotated_at`）、预密钥（含新增与撤销标记）、会话、消息与序号游标、每会话已用 nonce 集合，以及投递去重集合、`attempts`、acked 与设备撤销状态。
- 命令行 `encrypt-message` / `decrypt-message` 为纯本地 AES-256-GCM 加解密（不访问服务器）：`--session-id`、`--key`（base64 编码的 32 字节密钥）、`--plaintext`（UTF-8）→ 输出 `session_id`/`nonce`（12 字节，base64）/`ciphertext`（base64，末尾附 16 字节 GCM tag）；`decrypt-message` 额外接收 `--nonce`/`--ciphertext` → 输出 `session_id`/`plaintext`。`session_id` 的 UTF-8 字节作为 AAD 参与认证。任何失败在 stderr 打印带 `field` 的单行 JSON 并以非零码退出。
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

# 投递加密消息信封（sequence 从 1 开始逐条连续）
python3 -m e2ee_backend send-message \
  --session-id SESSION_ID --sender-device-id laptop \
  --message-id msg-1 --sequence 1 \
  --nonce BASE64_NONCE --ciphertext BASE64_CIPHERTEXT
# => {"session_id":"…","sender_device_id":"laptop","message_id":"msg-1",
#     "sequence":1,"nonce":"…","ciphertext":"…",
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

测试覆盖：公钥解析、注册成功/409 冲突/各类 400（指明字段）、查询/404、`prekey_ids` 顺序稳定与撤销过滤、设备与单预密钥撤销（200、幂等、404 及对应 field、同用户设备隔离）、撤销与查询并发线性化、会话协商成功（八字段、四入参回显、接收方公钥、唯一 session_id、重复 POST 新建）、会话各类 400/404/409（对应 field、失败不写）、会话快照在撤销后不变、创建与撤销并发原子线性化（撤销先行 409/创建先行 201 两种顺序均被观察到）、消息投递（信封回显与 created_at、sequence 从 1 连续、重复 message_id/错序/发送方撤销/未知会话的 400/404/409 及对应 field、失败不推进序号、会话间序号独立）、消息拉取（分页升序、next_after 语义、参数校验、读取方撤销 409、发送方撤销后已存消息仍可读）、可靠投递（首次 retry 201/attempts+1、相同 attempt_id 200 不计数、新 attempt_id 200 计数、acked 后重试仍 200 且保持 acked、ack 首次 201/重复 200 幂等、无重试直接 ack、序号不符 409/sequence、未知会话/消息 404、设备不符/未知/撤销 409/device_id、status 缺参 400/device_id、失败不改变状态）、持久化（缺失创建 version=1、损坏/非对象/版本不符/载荷畸形拒绝启动、原子替换无残留临时文件、重启恢复尝试去重/attempts/acked、序号游标、设备与预密钥撤销）、`serve --data-file` 真实子进程重启恢复与损坏拒启、AES-256-GCM 加解密（UTF-8 回环、随机 nonce、AAD 绑定 session_id、密钥/nonce/密文长度与编码校验、篡改与错密钥认证失败）、HTTP 全链路（真实 socket）、CLI 子命令（真实子进程，含连接失败 `field=server`、非零退出、无 traceback）。

## 代码结构

```
e2ee_backend/
  crypto.py       # cryptography 公钥解析/校验（PEM、DER、原始曲线点）与 AES-256-GCM 本地加解密
  models.py       # Device / SignedPreKey / Session / Message / MessageDelivery 数据模型
  storage.py      # 线程安全的进程内存储（插入顺序、撤销过滤、原子快照、会话原子创建、消息原子追加与分页、投递去重/确认、整体状态快照与恢复）
  persistence.py  # version=1 JSON 状态文件：缺失创建、损坏/版本不符拒启、临时文件+fsync+os.replace 原子替换
  service.py      # 业务逻辑与字段校验（400/404/409，设备/预密钥/会话/消息/投递）
  http_app.py     # POST/GET 路由与 JSON 响应（注册、查询、两类撤销、会话协商与查询、消息投递与拉取、重试/确认/状态）
  cli.py          # register/show/revoke-*/rotate-identity-key/add-prekey/create-session/show-session/send-message/pull-messages/retry-message/ack-message/message-status/encrypt-message/decrypt-message/serve 命令行入口
tests/            # unittest 测试
```
