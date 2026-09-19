# SOLOTASK02 端到端加密通信后端

要实现端到端加密通信后端的设备注册与预密钥发布，用 Python 加 cryptography，同时提供 HTTP 服务和命令行入口。注册走 POST /v1/devices，请求体是 JSON，含 user_id、device_id、identity_key 三个字符串字段，以及数组 signed_prekeys，每个元素含 key_id 与 public_key；成功返回 201，响应含 device_id 与 registered_at。同一 user_id 下 device_id 已存在时返回 409。任一必填字段缺失或 signed_prekeys 元素结构不符时返回 400，并指明是哪个字段。查询走 GET /v1/devices/{device_id}，返回 identity_key、prekey_ids 数组与 registered_at；设备不存在返回 404；prekey_ids 只列出未被撤销的预密钥，同一请求重复返回的顺序完全一致。命令行提供 register 与 show 两个子命令，与上述两个接口一一对应，打印单行 JSON 且字段名与 HTTP 响应一致。服务端只保存公开密钥与标识，不保存明文消息或私钥。同一 user_id 下多台设备互不影响，一台设备的状态变化不影响另一台。

## 当前状态

接口已实现（纯标准库 HTTP 服务 + `cryptography` 校验公钥），并带有单元/集成测试。

- `POST /v1/devices`：注册设备并发布签名预密钥；成功 `201`，冲突 `409`，校验失败 `400`（错误体带 `field` 指明字段，数组元素使用 `signed_prekeys[i].key_id` 这样的路径）。
- `GET /v1/devices/{device_id}`：返回 `identity_key`、`prekey_ids`（仅未撤销，顺序与注册时一致且重复请求完全相同）、`registered_at`；不存在返回 `404`。
- `POST /v1/devices/{device_id}/revoke`：撤销设备及其全部预密钥。已存在设备返回 `200`，响应体 `{"device_id":...,"revoked":true}`；重复调用幂等；未知设备返回 `404`（`field=device_id`）。撤销后 `GET` 的 `prekey_ids` 为空，`identity_key` 与 `registered_at` 不变。
- `POST /v1/devices/{device_id}/prekeys/{key_id}/revoke`：撤销单个预密钥。成功 `200`，响应体 `{"device_id":...,"key_id":...,"revoked":true}`；重复调用幂等；未知设备 `404/field=device_id`，设备存在但 `key_id` 未知 `404/field=key_id`。仅排除目标 key，其他 key 与同用户的其他设备不受影响。
- 命令行 `register` / `show` / `revoke-device` / `revoke-prekey` 与接口一一对应，成功时在 stdout 打印单行 JSON，字段名与 HTTP 一致；失败时在 stderr 打印单行 JSON 错误并以非零码退出。连接失败或超时时，API 命令在 stderr 打印 `field` 为 `server` 的单行 JSON、非零退出，且不输出 traceback。
- `POST /v1/sessions`：协商会话。请求体含 `initiator_device_id`、`recipient_device_id`、`prekey_id`、`ephemeral_key` 四个非空字符串；`ephemeral_key` 必须是合法公钥（与注册相同的编码）。缺失/类型/编码错误返回 `400` 且 `field` 指明字段；发起方与接收方是同一设备返回 `400`/`field=recipient_device_id`。未知设备/预密钥返回 `404`（`field` 为对应项，预密钥必须属于接收方设备）；发起方、接收方或预密钥已撤销返回 `409`（`field` 为对应项）；任何失败都不写入。成功 `201`，返回四字段回显及 `session_id`（唯一）、`identity_key`（接收方身份公钥）、`public_key`（所用预密钥公钥）、`created_at`（UTC ISO-8601，`+00:00`）；重复 POST 每次新建会话。
- `GET /v1/sessions/{session_id}`：返回上述八项快照；未知会话返回 `404`/`field=session_id`。快照在创建时固化，之后撤销设备或预密钥不改变快照。创建与撤销在同一把锁下原子执行：撤销先行则 `409`，创建先行则 `201` 且会话保留。
- 命令行 `create-session` / `show-session` 与上述两个接口一一对应，行为与错误约定同其他子命令。
- 撤销与查询共享同一把锁、线性化执行：并发的 `GET` 只能看到某次撤销操作前或后的完整快照，不会观察到中间态。
- 服务端仅保存标识与公开密钥（identity key、signed pre-key、ephemeral key 均为公钥），不保存私钥、共享秘密或明文消息。存储为进程内、线程安全。

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

# 协商会话（--ephemeral-key 可用 @路径 从文件读取）
python3 -m e2ee_backend create-session \
  --initiator-device-id phone --recipient-device-id laptop \
  --prekey-id 1 --ephemeral-key BASE64_OR_PEM_PUBLIC_KEY
# => {"initiator_device_id":"phone","recipient_device_id":"laptop","prekey_id":"1","ephemeral_key":"...","session_id":"...","identity_key":"...","public_key":"...","created_at":"..."}

# 查询会话快照
python3 -m e2ee_backend show-session SESSION_ID
# => 同上八字段
```

默认服务地址为 `http://127.0.0.1:8080`，可用全局参数 `--base-url` 或环境变量 `E2EE_BASE_URL` 覆盖。

## 基础测试命令

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖：公钥解析、注册成功/409 冲突/各类 400（指明字段）、查询/404、`prekey_ids` 顺序稳定与撤销过滤、设备与单预密钥撤销（200、幂等、404 及对应 field、同用户设备隔离）、撤销与查询并发线性化、会话协商（201 八字段契约、四字段回显、400/404/409 各分支、失败不写入、重复 POST 新建、快照不受撤销影响）、HTTP 全链路（真实 socket）、CLI 子命令（真实子进程，含连接失败 `field=server`、非零退出、无 traceback）。

## 代码结构

```
e2ee_backend/
  crypto.py    # cryptography 公钥解析/校验（PEM、DER、原始曲线点）
  models.py    # Device / SignedPreKey / Session 数据模型
  storage.py   # 线程安全的进程内存储（插入顺序、撤销过滤、原子快照、设备隔离、会话创建原子性）
  service.py   # 业务逻辑与字段校验（400/404/409，设备/预密钥撤销，会话协商与快照）
  http_app.py  # POST/GET 路由与 JSON 响应（注册、查询、两类撤销、会话）
  cli.py       # register / show / revoke-device / revoke-prekey / create-session / show-session / serve 命令行入口
tests/         # unittest 测试
```
