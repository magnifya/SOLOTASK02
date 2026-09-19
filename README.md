# SOLOTASK02 端到端加密通信后端

要实现端到端加密通信后端的设备注册与预密钥发布，用 Python 加 cryptography，同时提供 HTTP 服务和命令行入口。注册走 POST /v1/devices，请求体是 JSON，含 user_id、device_id、identity_key 三个字符串字段，以及数组 signed_prekeys，每个元素含 key_id 与 public_key；成功返回 201，响应含 device_id 与 registered_at。同一 user_id 下 device_id 已存在时返回 409。任一必填字段缺失或 signed_prekeys 元素结构不符时返回 400，并指明是哪个字段。查询走 GET /v1/devices/{device_id}，返回 identity_key、prekey_ids 数组与 registered_at；设备不存在返回 404；prekey_ids 只列出未被撤销的预密钥，同一请求重复返回的顺序完全一致。命令行提供 register 与 show 两个子命令，与上述两个接口一一对应，打印单行 JSON 且字段名与 HTTP 响应一致。服务端只保存公开密钥与标识，不保存明文消息或私钥。同一 user_id 下多台设备互不影响，一台设备的状态变化不影响另一台。

在此基础上新增会话协商与快照查询。POST /v1/sessions 接收 JSON：initiator_device_id、recipient_device_id、prekey_id、ephemeral_key 四个字段均须为非空字符串，ephemeral_key 使用与公钥相同的格式；缺失、类型错误或编码非法返回 400，错误体 field 指明对应字段。两台设备相同返回 400/field=recipient_device_id。服务端校验设备与预密钥归属：未知的发起方/接收方/预密钥分别返回 404/field=initiator_device_id、recipient_device_id、prekey_id；发起方设备、接收方设备或所用预密钥已撤销时分别返回 409/field=initiator_device_id、recipient_device_id、prekey_id，且失败时不写入任何会话。成功返回 201，响应含八个字段：四个入参原样回显，identity_key 为接收方身份公钥，public_key 为所用预密钥公钥，session_id 全局唯一，created_at 为 UTC ISO-8601（带 +00:00）。重复 POST 总是新建会话（新 session_id），不去重。GET /v1/sessions/{session_id} 返回同样的八个字段；未知会话返回 404/field=session_id。会话快照在创建时冻结，此后设备或预密钥撤销不改变快照。创建与撤销在同一把锁下原子线性化：撤销先行则创建 409 且不写，创建先行则 201 且会话保留。命令行的 create-session 与 show-session 与两个接口一一对应。服务端不保存私钥、共享秘密或明文消息。

## 当前状态

接口已实现（纯标准库 HTTP 服务 + `cryptography` 校验公钥），并带有单元/集成测试。

- `POST /v1/devices`：注册设备并发布签名预密钥；成功 `201`，冲突 `409`，校验失败 `400`（错误体带 `field` 指明字段，数组元素使用 `signed_prekeys[i].key_id` 这样的路径）。
- `GET /v1/devices/{device_id}`：返回 `identity_key`、`prekey_ids`（仅未撤销，顺序与注册时一致且重复请求完全相同）、`registered_at`；不存在返回 `404`。
- `POST /v1/devices/{device_id}/revoke`：撤销设备及其全部预密钥。已存在设备返回 `200`，响应体 `{"device_id":...,"revoked":true}`；重复调用幂等；未知设备返回 `404`（`field=device_id`）。撤销后 `GET` 的 `prekey_ids` 为空，`identity_key` 与 `registered_at` 不变。
- `POST /v1/devices/{device_id}/prekeys/{key_id}/revoke`：撤销单个预密钥。成功 `200`，响应体 `{"device_id":...,"key_id":...,"revoked":true}`；重复调用幂等；未知设备 `404/field=device_id`，设备存在但 `key_id` 未知 `404/field=key_id`。仅排除目标 key，其他 key 与同用户的其他设备不受影响。
- 命令行 `register` / `show` / `revoke-device` / `revoke-prekey` / `create-session` / `show-session` 与接口一一对应，成功时在 stdout 打印单行 JSON，字段名与 HTTP 一致；失败时在 stderr 打印单行 JSON 错误并以非零码退出。连接失败或超时时，API 命令在 stderr 打印 `field` 为 `server` 的单行 JSON、非零退出，且不输出 traceback。
- `POST /v1/sessions`：协商会话。四个入参（`initiator_device_id`、`recipient_device_id`、`prekey_id`、`ephemeral_key`）须为非空字符串，`ephemeral_key` 须为合法公钥编码，否则 `400` 并以 `field` 指明；两台设备相同返回 `400/field=recipient_device_id`。未知设备/预密钥返回 `404`，已撤销返回 `409`，`field` 分别为 `initiator_device_id` / `recipient_device_id` / `prekey_id`；失败不写入。成功 `201` 返回八字段：四入参回显、`identity_key`（接收方身份公钥）、`public_key`（所用预密钥公钥）、唯一 `session_id`、`created_at`（UTC ISO-8601，`+00:00`）。重复 POST 总是新建会话。
- `GET /v1/sessions/{session_id}`：返回与创建时一致的八字段快照；未知会话 `404/field=session_id`。快照创建后冻结，设备/预密钥撤销不改变它。
- 会话创建与设备/预密钥撤销共享同一把锁、线性化执行：撤销先行则创建得 `409` 且不写，创建先行则得 `201` 且会话保留，不存在中间态。
- 撤销与查询共享同一把锁、线性化执行：并发的 `GET` 只能看到某次撤销操作前或后的完整快照，不会观察到中间态。
- `POST /v1/messages`：向会话投递加密消息信封。请求体含 `session_id`、`sender_device_id`、`message_id`、`sequence`、`nonce`、`ciphertext`；`sequence` 从 1 开始逐条连续。未知会话 `404/field=session_id`；发送方设备不存在或已撤销 `409/field=sender_device_id`；`message_id` 在会话内重复 `409/field=message_id`；序号不连续 `409/field=sequence`；字段缺失或类型错误 `400/field=对应字段`。成功 `201`，响应为完整信封（六入参回显）加 `created_at`（UTC ISO-8601，`+00:00`）。失败不写入任何消息，序号游标不前进。
- `GET /v1/messages/{session_id}`：分页拉取会话消息。查询参数 `device_id` 必填，`after` 默认 0（须 ≥0），`limit` 默认 100（1..100）；未知会话 `404/field=session_id`，`device_id` 非活跃设备 `409/field=device_id`，参数缺失/非法 `400/field=对应参数`。返回 `messages`（与 POST 响应同构的信封数组，筛 `sequence > after` 升序）与 `next_after`（空页等于 `after`，否则为末条序号）。消息读取与设备撤销共享同一把锁，不观察中间态；已存消息在发送方被撤销后仍可读取。
- 命令行 `send-message` / `pull-messages` 与上述两个接口一一对应，同样打印单行 JSON。
- 命令行 `encrypt-message` / `decrypt-message` 为纯本地 AES-256-GCM 加解密（不访问服务器）：`--session-id`、`--key`（base64 编码的 32 字节密钥）、`--plaintext`（UTF-8）→ 输出 `session_id`/`nonce`（12 字节，base64）/`ciphertext`（base64，末尾附 16 字节 GCM tag）；`decrypt-message` 额外接收 `--nonce`/`--ciphertext` → 输出 `session_id`/`plaintext`。`session_id` 的 UTF-8 字节作为 AAD 参与认证。任何失败在 stderr 打印带 `field` 的单行 JSON 并以非零码退出。
- 服务端仅保存标识与公开密钥（identity key、signed pre-key、临时公钥均为公钥），不保存私钥、共享秘密或明文消息。存储为进程内、线程安全。

## 安装依赖

需要 Python 3.10+。

```bash
python3 -m pip install -r requirements.txt
# 或者安装为包（提供 e2ee-backend 命令）
python3 -m pip install -e .
```

## 启动方式

```bash
# 直接运行 HTTP 服务（默认 127.0.0.1:8080）
python3 -m e2ee_backend serve --host 0.0.0.0 --port 8080
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

测试覆盖：公钥解析、注册成功/409 冲突/各类 400（指明字段）、查询/404、`prekey_ids` 顺序稳定与撤销过滤、设备与单预密钥撤销（200、幂等、404 及对应 field、同用户设备隔离）、撤销与查询并发线性化、会话协商成功（八字段、四入参回显、接收方公钥、唯一 session_id、重复 POST 新建）、会话各类 400/404/409（对应 field、失败不写）、会话快照在撤销后不变、创建与撤销并发原子线性化（撤销先行 409/创建先行 201 两种顺序均被观察到）、消息投递（信封回显与 created_at、sequence 从 1 连续、重复 message_id/错序/发送方撤销/未知会话的 400/404/409 及对应 field、失败不推进序号、会话间序号独立）、消息拉取（分页升序、next_after 语义、参数校验、读取方撤销 409、发送方撤销后已存消息仍可读）、AES-256-GCM 加解密（UTF-8 回环、随机 nonce、AAD 绑定 session_id、密钥/nonce/密文长度与编码校验、篡改与错密钥认证失败）、HTTP 全链路（真实 socket）、CLI 子命令（真实子进程，含连接失败 `field=server`、非零退出、无 traceback）。

## 代码结构

```
e2ee_backend/
  crypto.py    # cryptography 公钥解析/校验（PEM、DER、原始曲线点）与 AES-256-GCM 本地加解密
  models.py    # Device / SignedPreKey / Session / Message 数据模型
  storage.py   # 线程安全的进程内存储（插入顺序、撤销过滤、原子快照、会话原子创建、消息原子追加与分页）
  service.py   # 业务逻辑与字段校验（400/404/409，设备/预密钥/会话/消息）
  http_app.py  # POST/GET 路由与 JSON 响应（注册、查询、两类撤销、会话协商与查询、消息投递与拉取）
  cli.py       # register/show/revoke-*/create-session/show-session/send-message/pull-messages/encrypt-message/decrypt-message/serve 命令行入口
tests/         # unittest 测试
```
