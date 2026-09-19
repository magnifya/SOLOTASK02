# SOLOTASK02 端到端加密通信后端

要实现端到端加密通信后端的设备注册与预密钥发布，用 Python 加 cryptography，同时提供 HTTP 服务和命令行入口。注册走 POST /v1/devices，请求体是 JSON，含 user_id、device_id、identity_key 三个字符串字段，以及数组 signed_prekeys，每个元素含 key_id 与 public_key；成功返回 201，响应含 device_id 与 registered_at。同一 user_id 下 device_id 已存在时返回 409。任一必填字段缺失或 signed_prekeys 元素结构不符时返回 400，并指明是哪个字段。查询走 GET /v1/devices/{device_id}，返回 identity_key、prekey_ids 数组与 registered_at；设备不存在返回 404；prekey_ids 只列出未被撤销的预密钥，同一请求重复返回的顺序完全一致。命令行提供 register 与 show 两个子命令，与上述两个接口一一对应，打印单行 JSON 且字段名与 HTTP 响应一致。服务端只保存公开密钥与标识，不保存明文消息或私钥。同一 user_id 下多台设备互不影响，一台设备的状态变化不影响另一台。

在此基础上新增会话协商与快照查询。POST /v1/sessions 接收 JSON：initiator_device_id、recipient_device_id、prekey_id、ephemeral_key 四个字段均须为非空字符串，ephemeral_key 使用与公钥相同的格式；缺失、类型错误或编码非法返回 400，错误体 field 指明对应字段。两台设备相同返回 400/field=recipient_device_id。服务端校验设备与预密钥归属：未知的发起方/接收方/预密钥分别返回 404/field=initiator_device_id、recipient_device_id、prekey_id；发起方设备、接收方设备或所用预密钥已撤销时分别返回 409/field=initiator_device_id、recipient_device_id、prekey_id，且失败时不写入任何会话。成功返回 201，响应含八个字段：四个入参原样回显，identity_key 为接收方身份公钥，public_key 为所用预密钥公钥，session_id 全局唯一，created_at 为 UTC ISO-8601（带 +00:00）。重复 POST 总是新建会话（新 session_id），不去重。GET /v1/sessions/{session_id} 返回同样的八个字段；未知会话返回 404/field=session_id。会话快照在创建时冻结，此后设备或预密钥撤销不改变快照。创建与撤销在同一把锁下原子线性化：撤销先行则创建 409 且不写，创建先行则 201 且会话保留。命令行的 create-session 与 show-session 与两个接口一一对应。服务端不保存私钥、共享秘密或明文消息。

在会话之上新增端到端密文消息的投递与拉取（服务端只转发密文，永远看不到明文或对称密钥）。POST /v1/messages 接收 JSON：session_id、sender_device_id、message_id、nonce、ciphertext 为非空字符串，sequence 为从 1 开始的正整数；任一必填字段缺失或类型错误返回 400/field（按声明顺序报告第一个出错字段，布尔值不算整数）。未知会话返回 404/field=session_id；发送方设备未注册或已撤销返回 409/field=sender_device_id；message_id 在该会话内重复返回 409/field=message_id；sequence 错序（必须等于已有条数 +1，间隙或重放都算错序）返回 409/field=sequence；任何失败都不写入。成功返回 201，响应体回显六个字段并附 created_at（UTC ISO-8601，+00:00）。GET /v1/messages/{session_id} 的 device_id 查询参数必填（缺失 400/field=device_id），after 默认 0、limit 默认 100（合法区间 1..100，越界或非整数 400 且 field 为对应参数名）；未知会话 404/field=session_id，device_id 未注册或已撤销 409/field=device_id。返回 {"messages":[...], "next_after": n}，messages 元素与 POST 响应同体，按 sequence 升序且只含 sequence > after 的至多 limit 条；next_after 在页为空时等于 after，否则等于本页最后一条的序号。命令行 send-message 与 pull-messages 与两个接口一一对应。

本地加解密由 encrypt-message / decrypt-message 两个子命令提供（纯本地运算，不联网）。encrypt-message 输入 session_id、key（base64，恰好 32 字节的 AES-256 密钥）、plaintext（UTF-8），使用 AES-256-GCM：随机 12 字节 nonce，AAD 为 session_id 的 UTF-8 字节，输出 session_id/nonce/ciphertext（ciphertext 为 base64，末尾含 16 字节 GCM tag）。decrypt-message 额外输入 nonce、ciphertext，返回 session_id/plaintext；会话绑定（AAD）不匹配、密钥错误或密文被篡改时认证失败，stderr 输出单行 JSON（field=ciphertext）并以非零码退出；key 不是合法 base64 或长度不是 32 字节报 field=key，nonce 解码或长度（12 字节）错误报 field=nonce，字段缺失或类型错误报对应 field。

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
- `POST /v1/messages`：投递密文消息。`session_id`、`sender_device_id`、`message_id`、`nonce`、`ciphertext` 须为非空字符串，`sequence` 须为从 1 起的正整数，否则 `400` 并以 `field` 指明（按声明顺序报告首个错误字段）。未知会话 `404/field=session_id`；发送方未注册或已撤销 `409/field=sender_device_id`；会话内重复 `message_id` `409/field=message_id`；`sequence` 不连续（间隙或重放）`409/field=sequence`；失败不写入。成功 `201`，回显六字段并附 `created_at`。
- `GET /v1/messages/{session_id}`：`device_id` 必填；`after` 默认 `0`，`limit` 默认 `100`（`1..100`，非法 `400/field=after|limit`）；未知会话 `404`，`device_id` 未注册或已撤销 `409`。返回 `{"messages":[...同 POST 响应体...],"next_after":n}`，仅含 `sequence > after` 的至多 `limit` 条、按序号升序；空页 `next_after` 等于 `after`，否则等于本页末序号。
- 消息追加与会话/设备撤销共享存储锁：并发追加被线性化，序号恰好为连续的 `1..N`，重复 id 不会漏过。
- 命令行 `send-message` / `pull-messages` 与消息接口一一对应；`encrypt-message` / `decrypt-message` 为本地 AES-256-GCM 加解密（12 字节随机 nonce，AAD=session_id 的 UTF-8 字节，密文 base64 且末尾带 16 字节 tag），成功 stdout 单行 JSON，失败 stderr 单行 JSON（`field` 指明字段，认证失败为 `ciphertext`）且非零退出。
- 服务端仅保存标识与公开密钥（identity key、signed pre-key、临时公钥均为公钥）以及消息密文，不保存私钥、共享秘密或明文消息。存储为进程内、线程安全。

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

# 本地加密（AES-256-GCM，12 字节随机 nonce，AAD=session_id）
KEY=$(python3 -c "import base64,os;print(base64.b64encode(os.urandom(32)).decode())")
python3 -m e2ee_backend encrypt-message \
  --session-id SESSION_ID --key "$KEY" --plaintext '你好'
# => {"session_id":"…","nonce":"…（12B base64）","ciphertext":"…（base64，末尾 16B tag）"}

# 本地解密（AAD 不匹配 / 密钥错误 / 密文被篡改 -> stderr JSON field=ciphertext，非零退出）
python3 -m e2ee_backend decrypt-message \
  --session-id SESSION_ID --key "$KEY" --nonce NONCE_B64 --ciphertext CIPHERTEXT_B64
# => {"session_id":"…","plaintext":"你好"}

# 投递密文消息（sequence 从 1 连续；重复 id / 错序 -> 409）
python3 -m e2ee_backend send-message \
  --session-id SESSION_ID --sender-device-id laptop \
  --message-id msg-1 --sequence 1 \
  --nonce NONCE_B64 --ciphertext CIPHERTEXT_B64
# => {六个字段回显 + "created_at":"…+00:00"}

# 拉取消息（after 默认 0，limit 默认 100；返回 messages 与 next_after）
python3 -m e2ee_backend pull-messages SESSION_ID --device-id phone --after 0 --limit 100
# => {"messages":[…],"next_after":1}
```

默认服务地址为 `http://127.0.0.1:8080`，可用全局参数 `--base-url` 或环境变量 `E2EE_BASE_URL` 覆盖。

## 基础测试命令

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖：公钥解析、注册成功/409 冲突/各类 400（指明字段）、查询/404、`prekey_ids` 顺序稳定与撤销过滤、设备与单预密钥撤销（200、幂等、404 及对应 field、同用户设备隔离）、撤销与查询并发线性化、会话协商成功（八字段、四入参回显、接收方公钥、唯一 session_id、重复 POST 新建）、会话各类 400/404/409（对应 field、失败不写）、会话快照在撤销后不变、创建与撤销并发原子线性化（撤销先行 409/创建先行 201 两种顺序均被观察到）、消息投递成功（七字段、`created_at`）、`sequence` 从 1 连续、重复 `message_id` 与错序（间隙/重放）409、发送方未注册/已撤销 409、未知会话 404、各类 400（含声明顺序首个错误字段、`sequence` 非正整数）、拉取分页（默认 after=0/limit=100、`next_after` 空页等于 after）、device_id 缺失/非活跃与参数非法、并发追加线性化得到连续 1..N、AES-256-GCM 信封（随机 nonce、UTF-8 往返、AAD/密钥/篡改认证失败 field=ciphertext、key/nonce 字段错误）、HTTP 全链路（真实 socket）、CLI 子命令（真实子进程，含 send/pull 与本地 encrypt/decrypt、连接失败 `field=server`、非零退出、无 traceback）。

## 代码结构

```
e2ee_backend/
  crypto.py    # cryptography 公钥解析/校验（PEM、DER、原始曲线点）
  envelope.py  # 本地 AES-256-GCM 信封 seal/open（12B nonce、AAD=session_id）
  models.py    # Device / SignedPreKey / Session / Message 数据模型
  storage.py   # 线程安全的进程内存储（插入顺序、撤销过滤、原子快照、会话/消息原子创建）
  service.py   # 业务逻辑与字段校验（400/404/409，设备/预密钥/会话/消息）
  http_app.py  # POST/GET 路由与 JSON 响应（注册、查询、撤销、会话、消息投递与拉取）
  cli.py       # register/show/revoke-*/create-/show-session/send-/pull-messages/encrypt-/decrypt-message/serve
tests/         # unittest 测试
```
