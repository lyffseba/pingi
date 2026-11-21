"""Local Docker SandboxInterface for $0-cost Harbor runs on macOS/Linux.

Supports both prebuilt `docker_image` references (fast startup for `scale800-harbor`
and `termgrade-train-harbor`) and local `environment/Dockerfile` builds with per-image
async locking so parallel rollouts never race on `docker build`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shlex
import uuid
from dataclasses import dataclass

from pingy.datasets import PingyTask

logger = logging.getLogger(__name__)

_BUILD_LOCKS: dict[str, asyncio.Lock] = {}
_BUILT_IMAGES: set[str] = set()


def _lock_for(key: str) -> asyncio.Lock:
    lock = _BUILD_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _BUILD_LOCKS[key] = lock
    return lock


@dataclass
class LocalSandboxResult:
    exit_code: int
    stdout: str
    stderr: str = ""


class DockerSandbox:
    """Runs Harbor task environments in a local Docker container."""

    def __init__(self, container_id: str) -> None:
        self.container_id = container_id

    @classmethod
    async def _ensure_image(cls, task: PingyTask) -> str:
        # 1. Prefer prebuilt registry image when specified in task.toml
        if task.docker_image:
            image = task.docker_image
            async with _lock_for(f"pull:{image}"):
                if image not in _BUILT_IMAGES:
                    inspect = await asyncio.create_subprocess_exec(
                        "docker",
                        "image",
                        "inspect",
                        image,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    if await inspect.wait() != 0:
                        pull = await asyncio.create_subprocess_exec(
                            "docker",
                            "pull",
                            image,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        _, err = await pull.communicate()
                        if pull.returncode != 0:
                            err_text = err.decode(errors="replace")
                            if "no matching manifest" in err_text or "no match for platform" in err_text:
                                pull_amd64 = await asyncio.create_subprocess_exec(
                                    "docker",
                                    "pull",
                                    "--platform",
                                    "linux/amd64",
                                    image,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE,
                                )
                                _, err_amd64 = await pull_amd64.communicate()
                                if pull_amd64.returncode == 0:
                                    _BUILT_IMAGES.add(image)
                                    return image
                                err_text = err_amd64.decode(errors="replace")
                            if task.dockerfile_path is None:
                                raise RuntimeError(
                                    f"docker pull {image} failed: {err_text}"
                                )
                            logger.warning(
                                "docker pull %s failed (%s); falling back to Dockerfile",
                                image,
                                err_text.strip()[:200],
                            )
                        else:
                            _BUILT_IMAGES.add(image)
                            return image
                    else:
                        _BUILT_IMAGES.add(image)
                        return image
                else:
                    return image

        # 2. Build from environment/Dockerfile (locked so group_size rollouts share 1 build)
        dockerfile = task.dockerfile_path
        if dockerfile is None:
            raise RuntimeError(
                f"Task {task.task_name} has neither a pullable docker_image nor environment/Dockerfile."
            )
        env_dir = dockerfile.parent
        digest = hashlib.sha256(str(env_dir.resolve()).encode()).hexdigest()[:12]
        image_tag = f"pingy-harbor-{digest}"

        async with _lock_for(f"build:{image_tag}"):
            if image_tag not in _BUILT_IMAGES:
                build = await asyncio.create_subprocess_exec(
                    "docker",
                    "build",
                    "-q",
                    "-t",
                    image_tag,
                    "-f",
                    str(dockerfile),
                    str(env_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, err = await build.communicate()
                if build.returncode != 0:
                    raise RuntimeError(
                        f"docker build failed for {env_dir}: {err.decode(errors='replace')}"
                    )
                _BUILT_IMAGES.add(image_tag)
        return image_tag

    @classmethod
    async def create_for_task(cls, task: PingyTask, timeout: int = 600) -> DockerSandbox:
        del timeout
        image = await cls._ensure_image(task)
        name = f"pingy-{uuid.uuid4().hex[:12]}"
        run = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            image,
            "sleep",
            "infinity",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await run.communicate()
        if run.returncode != 0:
            raise RuntimeError(f"docker run failed for {image}: {err.decode(errors='replace')}")
        return cls(container_id=out.decode().strip())

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> LocalSandboxResult:
        cmd = [
            "docker",
            "exec",
            "-e",
            "DEBIAN_FRONTEND=noninteractive",
            "-e",
            "GIT_PAGER=cat",
            "-e",
            "PAGER=cat",
            "-e",
            "CI=true",
        ]
        if workdir:
            cmd += ["-w", workdir]
        cmd += [self.container_id, "bash", "-lc", command]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return LocalSandboxResult(
                exit_code=124, stdout="", stderr=f"Command timed out after {timeout}s"
            )

        if max_output_bytes is not None:
            stdout_b = stdout_b[:max_output_bytes]
            stderr_b = stderr_b[:max_output_bytes]
        return LocalSandboxResult(
            exit_code=proc.returncode if proc.returncode is not None else 1,
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
        )

    async def read_file(
        self,
        path: str,
        max_bytes: int | None = None,
        timeout: int = 60,
    ) -> LocalSandboxResult:
        return await self.run_command(
            f"cat {shlex.quote(path)}",
            workdir="/",
            timeout=timeout,
            max_output_bytes=max_bytes,
        )

    async def write_file(
        self,
        path: str,
        content: str | bytes,
        executable: bool = False,
        timeout: int = 60,
    ) -> LocalSandboxResult:
        raw = content.encode("utf-8") if isinstance(content, str) else content
        quoted_path = shlex.quote(path)
        chmod_part = f" && chmod +x {quoted_path}" if executable else ""
        script = f"mkdir -p $(dirname {quoted_path}) && cat > {quoted_path}{chmod_part}"
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            self.container_id,
            "bash",
            "-lc",
            script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=raw), timeout=timeout
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return LocalSandboxResult(
                exit_code=124, stdout="", stderr=f"write_file timed out after {timeout}s"
            )
        return LocalSandboxResult(
            exit_code=proc.returncode if proc.returncode is not None else 1,
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
        )

    async def send_heartbeat(self) -> None:
        return None

    async def cleanup(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            self.container_id,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()


async def docker_sandbox_factory(task: PingyTask, timeout: int) -> DockerSandbox:
    return await DockerSandbox.create_for_task(task=task, timeout=timeout)
