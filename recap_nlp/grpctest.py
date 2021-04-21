import asyncio
from typing import AsyncIterator

from grpclib.server import Server
from grpclib.utils import graceful_exit
from recap_schema.test.v1 import EchoBase, EchoResponse, EchoStreamResponse


class EchoService(EchoBase):
    async def echo(self, value: str, extra_times: int) -> "EchoResponse":
        return EchoResponse([value for _ in range(extra_times)])

    async def echo_stream(
        self, value: str, extra_times: int
    ) -> AsyncIterator["EchoStreamResponse"]:
        for _ in range(extra_times):
            yield EchoStreamResponse(value)


async def main(*, host="127.0.0.1", port=50051):
    server = Server([EchoService()])
    # Note: graceful_exit isn't supported in Windows
    with graceful_exit([server]):
        await server.start(host, port)
        print(f"Serving on {host}:{port}")
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
