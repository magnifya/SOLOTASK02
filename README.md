# SOLOTASK02 端到端加密通信后端

要实现端到端加密通信后端的设备注册与预密钥发布，用 Python 加 cryptography，同时提供 HTTP 服务和命令行入口。注册走 POST /v1/devices，请求体是 JSON，含 user_id、device_id、identity_key 三个字符串字段，以及数组 signed_prekeys，每个元素含 key_id 与 public_key；成功返回 201，响应含 device_id 与 registered_at。同一 user_id 下 device_id 已存在时返回 409。任一必填字段缺失或 signed_prekeys 元素结构不符时返回 400，并指明是哪个字段。查询走 GET /v1/devices/{device_id}，返回 identity_key、prekey_ids 数组与 registered_at；设备不存在返回 404；prekey_ids 只列出未被撤销的预密钥，同一请求重复返回的顺序完全一致。命令行提供 register 与 show 两个子命令，与上述两个接口一一对应，打印单行 JSON 且字段名与 HTTP 响应一致。服务端只保存公开密钥与标识，不保存明文消息或私钥。同一 user_id 下多台设备互不影响，一台设备的状态变化不影响另一台。

## 当前状态

接口已实现，代码位于 `e2e_backend/`：

- `store.py` — 线程安全的内存设备存储，只保存公开密钥与标识
- `server.py` — HTTP 服务（标准库 `http.server`），`POST /v1/devices` 与 `GET /v1/devices/{device_id}`
- `cli.py` — 命令行入口，`register` / `show` 子命令，输出单行 JSON
- `keyutil.py` — 基于 `cryptography` 的公钥解析与指纹工具

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
python3 -m e2e_backend.server --host 127.0.0.1 --port 8000
```

### 命令行用法

```bash
# 注册设备（服务地址可用 --server 或环境变量 E2E_SERVER 指定）
python3 -m e2e_backend.cli register --user-id alice --device-id phone \
    --identity-key IK_ALICE --prekey k1:PK1 --prekey k2:PK2

# 查询设备
python3 -m e2e_backend.cli show phone
```

### 运行测试

```bash
python3 -m unittest discover -s tests -v
```
