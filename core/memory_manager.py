# core/memory_manager.py
from core.messages import MessageService  # 修正这里
class Memory:
    def __init__(self, uid="default_user", memory_id="default_mem"):
        self.uid = uid
        self.memory_id = memory_id
        self.service = MessageService()

    def insert(self, role, content):
        msg = {"role": role, "content": content}
        self.service.insert_message([msg], self.uid, self.memory_id)

    def get(self, limit=10):
        msgs = self.service.get_recent_messages([self.uid], [self.memory_id], None, None, limit)
        return "\n".join([f"{m['role']}: {m['content']}" for m in msgs])