import time

class NodeResult:
    def __init__(self, success, data=None, error=None):
        self.success = success
        self.data = data
        self.error = error

    @staticmethod
    def ok(data):
        return NodeResult(True, data=data)

    @staticmethod
    def fail(error):
        return NodeResult(False, error=error)

class RecoveryRunner:
    def __init__(self, max_retries=3, interval=1):
        self.max_retries = max_retries
        self.interval = interval

    def run(self, func, *args, **kwargs):
        retry = 0
        while retry <= self.max_retries:
            try:
                result = func(*args, **kwargs)
                return NodeResult.ok(result)
            except Exception as e:
                print(f"⚠️ 执行失败: {e} | retry {retry}/{self.max_retries}")
                if retry >= self.max_retries:
                    return NodeResult.fail(str(e))
                retry += 1
                time.sleep(self.interval)