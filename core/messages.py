# core/messages.py
import sys
from typing import List

# 简化版，去掉对 common.settings/constants 的依赖
# 支持最基本的多轮 memory 功能

def index_name(uid: str):
    return f"memory_{uid}"

class MessageService:
    def __init__(self):
        self.store = {}  # {(uid, memory_id): [messages]}

    # 插入消息
    def insert_message(self, messages: List[dict], uid: str, memory_id: str):
        key = (uid, memory_id)
        if key not in self.store:
            self.store[key] = []
        for m in messages:
            m.setdefault("message_id", len(self.store[key]) + 1)
            m.setdefault("status", 1)
        self.store[key].extend(messages)
        return True

    # 获取最近消息
    def get_recent_messages(self, uid_list: List[str], memory_ids: List[str], agent_id=None, session_id=None, limit=10):
        key = (uid_list[0], memory_ids[0])
        if key not in self.store:
            return []
        msgs = self.store[key][-limit:]
        return msgs

    # 搜索消息（简单模拟）
    def search_message(self, memory_ids: List[str], condition_dict: dict, uid_list: List[str], match_expressions:list=None, top_n: int=10):
        key = (uid_list[0], memory_ids[0])
        if key not in self.store:
            return []
        return self.store[key][:top_n]

    # 计算消息大小（模拟）
    @staticmethod
    def calculate_message_size(message: dict):
        content = message.get("content","")
        return sys.getsizeof(content)

    # 删除消息（简单模拟）
    def delete_message(self, condition: dict, uid: str, memory_id: str):
        key = (uid, memory_id)
        if key in self.store:
            self.store[key] = []
        return True