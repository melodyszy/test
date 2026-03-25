from core.recovery import RecoveryRunner

class Node:
    def __init__(self, node_id, component, downstream=None, params=None):
        self.node_id = node_id
        self.component = component
        self.downstream = downstream if downstream else []
        self.params = params if params else {}

class DSLRunner:
    def __init__(self, nodes_json, component_map):
        self.nodes = {}
        # 兼容 workflow JSON
        if isinstance(nodes_json, dict) and "nodes" in nodes_json:
            node_items = nodes_json["nodes"].items()
            self.start_node = nodes_json.get("start_node")
        elif isinstance(nodes_json, dict):
            node_items = nodes_json.items()
            self.start_node = list(nodes_json.keys())[0]
        else:
            raise ValueError("workflow JSON 格式不支持")

        for nid, info in node_items:
            if isinstance(info, dict):
                component_name = info.get("component") or info.get("llm_id") or "answer_agent"
                downstream = info.get("downstream", [])
                params = info.get("params", {})
            else:
                component_name = "answer_agent"
                downstream = []
                params = {}

            component = component_map.get(component_name)
            if component is None:
                raise ValueError(f"component '{component_name}' 不在 component_map 中")

            self.nodes[nid] = Node(nid, component, downstream, params)

        self.recovery_runner = RecoveryRunner(max_retries=3)

    async def _run_node(self, node_id, inputs):
        node = self.nodes[node_id]
        result = self.recovery_runner.run(node.component.run, inputs)
        outputs = result.data if result.success else None
        for dn in node.downstream:
            await self._run_node(dn, outputs)
        return outputs

    def run(self, inputs):
        import asyncio
        return asyncio.run(self._run_node(self.start_node, inputs))