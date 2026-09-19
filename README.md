# SOLOTASK02

端到端加密通信后端的设备注册与预密钥发布。Python 实现，密码学部分使用 cryptography，对外提供 HTTP 服务与命令行入口，两者能力一致。

## 运行

    python -m app --port <port>      # 启动 HTTP 服务
    python -m app <subcommand>       # 命令行入口，输出单行 JSON

## 公开接口

POST /v1/devices
  请求：user_id、device_id、identity_key、signed_prekey[]
        每个 signed_prekey 含 key_id 与 public_key
  成功：201 -> device_id、registered_at

GET /v1/devices/{device_id}
  成功：200 -> identity_key、prekey key_id 列表、registered_at

## 约定

- 同一 user_id 下 device_id 必须唯一；重复注册：409
- 字段缺失：400，需指明缺少哪个字段
- 设备不存在：404
- 已撤销的预密钥不得出现在列表中
- 同一设备的列表顺序每次一致
- 服务端只保存公开密钥与标识，不保存明文消息或私钥
- 同一用户的多台设备相互独立，一台的状态变化不影响另一台
- 服务重启后已注册设备仍可查询；重复提交同一组预密钥不产生重复条目

## 当前状态

接口尚未实现；实现完成后需在此补充安装依赖、启动方式与基础测试命令。
