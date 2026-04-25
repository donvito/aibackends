from aibackends.core.registry import TaskSpec
from aibackends.tasks import BaseTask, create_task, register_task


class WordCountTask(BaseTask):
    name = "word-count"

    def run(self, input: str, **kwargs):
        options = self._resolve_options(**kwargs)
        if options.get("strip", False):
            input = input.strip()
        return {
            "characters": len(input),
            "words": len(input.split()),
        }


register_task(TaskSpec(name=WordCountTask.name, task_factory=WordCountTask))

task = create_task("word-count", strip=True)
result = task.run("AIBackends tasks can be objects too.")
print(result)
